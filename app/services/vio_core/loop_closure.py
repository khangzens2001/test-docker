"""BoW & PnP RANSAC Loop Detection Module (Task 10A)."""

import numpy as np
import cv2


class LoopDetector:
    """Loop Closure Detector using Bag of Words / Inverted Index & PnP RANSAC geometric verification."""

    def __init__(
        self,
        min_inliers: int = 12,
        min_kf_diff: int = 10,
        max_hamming_dist: int = 64,
        ratio_thresh: float = 0.8,
    ) -> None:
        """Initializes the Loop Detector.

        Args:
            min_inliers: Minimum number of PnP RANSAC inliers to confirm a loop closure.
            min_kf_diff: Minimum keyframe ID difference to consider candidate loops.
            max_hamming_dist: Maximum Hamming distance for descriptor matching.
            ratio_thresh: Lowe's ratio test threshold.
        """
        self.min_inliers = min_inliers
        self.min_kf_diff = min_kf_diff
        self.max_hamming_dist = max_hamming_dist
        self.ratio_thresh = ratio_thresh

        # Keyframe database: kf_id -> dict with 'descriptors', 'kps_3d', 'K', 'kps_2d'
        self.db: dict[int, dict] = {}
        # Inverted index for feature retrieval: word_id -> list of (kf_id, feature_idx)
        self.inverted_index: dict[int, list[tuple[int, int]]] = {}

    def _hash_descriptor(self, desc: np.ndarray) -> int:
        """Simple visual word hash from top 16 bits of descriptor for inverted index lookup."""
        if desc.dtype == np.uint8:
            return int(desc[0]) | (int(desc[1]) << 8)
        else:
            return int(hash(desc.tobytes()) & 0xFFFF)

    def add_keyframe(
        self,
        kf_id: int,
        descriptors: np.ndarray,
        kps_3d: np.ndarray,
        K: np.ndarray,
        kps_2d: np.ndarray | None = None,
    ) -> None:
        """Adds a keyframe to the descriptor database and inverted index.

        Args:
            kf_id: Keyframe ID.
            descriptors: (N, D) array of feature descriptors.
            kps_3d: (N, 3) 3D landmark positions in keyframe camera frame.
            K: 3x3 camera intrinsic matrix.
            kps_2d: Optional (N, 2) keypoint image coordinates.
        """
        if descriptors is None or len(descriptors) == 0:
            return

        self.db[kf_id] = {
            "descriptors": descriptors.copy(),
            "kps_3d": kps_3d.copy() if kps_3d is not None and len(kps_3d) > 0 else np.empty((0, 3), dtype=np.float64),
            "K": K.copy(),
            "kps_2d": kps_2d.copy() if kps_2d is not None else None,
        }

        # Index descriptors into inverted index
        for idx, desc in enumerate(descriptors):
            word_id = self._hash_descriptor(desc)
            if word_id not in self.inverted_index:
                self.inverted_index[word_id] = []
            self.inverted_index[word_id].append((kf_id, idx))

    def detect_loop(
        self,
        query_kf_id: int,
        query_descriptors: np.ndarray,
        query_kps_2d: np.ndarray,
        K: np.ndarray,
        min_kf_diff: int | None = None,
    ) -> tuple[int, int, np.ndarray, np.ndarray, int] | None:
        """Detects loop closure candidates and performs PnP RANSAC geometric verification.

        Args:
            query_kf_id: Query keyframe ID.
            query_descriptors: (M, D) array of query feature descriptors.
            query_kps_2d: (M, 2) query 2D keypoint projections.
            K: 3x3 camera intrinsic matrix.
            min_kf_diff: Optional override for min keyframe ID difference.

        Returns:
            tuple (keyframe_i, keyframe_j, R_ij, t_ij, inlier_count) if a loop is detected,
            or None if no candidate satisfies geometric verification.
        """
        diff_threshold = min_kf_diff if min_kf_diff is not None else self.min_kf_diff

        if query_descriptors is None or len(query_descriptors) == 0:
            return None

        # Query inverted index for candidate past keyframes sharing visual words
        word_votes: dict[int, int] = {}
        for desc in query_descriptors:
            word_id = self._hash_descriptor(desc)
            if word_id in self.inverted_index:
                for past_kf_id, _ in self.inverted_index[word_id]:
                    if abs(query_kf_id - past_kf_id) >= diff_threshold:
                        word_votes[past_kf_id] = word_votes.get(past_kf_id, 0) + 1

        min_votes = max(8, int(self.min_inliers * 0.6))
        candidate_ids = [
            kf_id for kf_id, votes in sorted(word_votes.items(), key=lambda item: item[1], reverse=True)[:5]
            if votes >= min_votes
        ]

        if not candidate_ids:
            return None

        is_binary = query_descriptors.dtype == np.uint8
        norm_type = cv2.NORM_HAMMING if is_binary else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)

        best_result = None
        max_inliers = 0

        for past_kf_id in candidate_ids:
            past_kf = self.db[past_kf_id]
            past_desc = past_kf["descriptors"]
            past_3d = past_kf["kps_3d"]
            past_2d = past_kf.get("kps_2d")

            if len(past_desc) < self.min_inliers:
                continue

            # Match features between past keyframe descriptors and query descriptors
            try:
                knn_matches = matcher.knnMatch(past_desc, query_descriptors, k=2)
            except cv2.error:
                continue

            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.ratio_thresh * n.distance:
                        if not is_binary or m.distance <= self.max_hamming_dist:
                            good_matches.append(m)
                elif len(match_pair) == 1:
                    m = match_pair[0]
                    if not is_binary or m.distance <= self.max_hamming_dist:
                        good_matches.append(m)

            if len(good_matches) < self.min_inliers:
                continue

            used_pnp = False
            # 1. 3D-2D PnP RANSAC geometric verification when 3D landmarks are available
            if len(past_3d) >= self.min_inliers:
                valid_3d = [
                    m for m in good_matches
                    if m.queryIdx < len(past_3d) and past_3d[m.queryIdx, 2] > 0.05 and m.trainIdx < len(query_kps_2d)
                ]
                if len(valid_3d) >= self.min_inliers:
                    obj_pts = np.array([past_3d[m.queryIdx] for m in valid_3d], dtype=np.float64)
                    img_pts = np.array([query_kps_2d[m.trainIdx] for m in valid_3d], dtype=np.float64)

                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        objectPoints=obj_pts,
                        imagePoints=img_pts,
                        cameraMatrix=K.astype(np.float64),
                        distCoeffs=None,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                        reprojectionError=4.0,
                        confidence=0.99,
                        iterationsCount=50,
                    )

                    if success and inliers is not None and len(inliers) >= self.min_inliers:
                        num_inliers = len(inliers)
                        if np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec)):
                            if (len(inliers) / max(len(valid_3d), 1)) >= 0.25:
                                R_ji, _ = cv2.Rodrigues(rvec)
                                t_ji = tvec.reshape(3, 1)
                                R_ij = R_ji.T
                                t_ij = (-R_ji.T @ t_ji).reshape(3)

                                t_norm = float(np.linalg.norm(t_ij))
                                rot_norm = float(np.linalg.norm(rvec))
                                if np.isfinite(t_norm) and t_norm <= 5.0 and np.isfinite(rot_norm) and rot_norm <= np.pi + 0.1:
                                    used_pnp = True
                                    if num_inliers > max_inliers:
                                        max_inliers = num_inliers
                                        best_result = (past_kf_id, query_kf_id, R_ij, t_ij, num_inliers)
                                        if num_inliers >= 20:
                                            break

            # 2. 2D-2D Essential Matrix verification when 3D points are lacking (zero parallax at start/end)
            if not used_pnp and past_2d is not None and len(past_2d) >= self.min_inliers and query_kps_2d is not None and len(query_kps_2d) >= self.min_inliers:
                valid_2d = [m for m in good_matches if m.queryIdx < len(past_2d) and m.trainIdx < len(query_kps_2d)]
                if len(valid_2d) >= self.min_inliers:
                    pts1 = np.array([past_2d[m.queryIdx] for m in valid_2d], dtype=np.float64)
                    pts2 = np.array([query_kps_2d[m.trainIdx] for m in valid_2d], dtype=np.float64)
                    E, mask = cv2.findEssentialMat(
                        pts1, pts2, K.astype(np.float64), method=cv2.RANSAC, prob=0.999, threshold=2.0
                    )
                    if mask is not None and np.sum(mask) >= self.min_inliers:
                        retval, R_loop_rel, t_loop_rel, pose_mask = cv2.recoverPose(
                            E, pts1, pts2, K.astype(np.float64), mask=mask
                        )
                        num_inliers = int(np.sum(pose_mask > 0)) if pose_mask is not None else int(np.sum(mask))
                        if num_inliers >= self.min_inliers and (num_inliers / max(len(valid_2d), 1)) >= 0.25:
                            R_ij = R_loop_rel.T
                            t_unit = (-R_loop_rel.T @ t_loop_rel).reshape(3)
                            t_unit_norm = float(np.linalg.norm(t_unit))
                            if t_unit_norm > 1e-6:
                                t_unit = t_unit / t_unit_norm
                            if num_inliers > max_inliers:
                                max_inliers = num_inliers
                                best_result = (past_kf_id, query_kf_id, R_ij, t_unit, num_inliers)
                                if num_inliers >= 20:
                                    break

        return best_result


# Alias for backward compatibility / specification naming
LoopClosure = LoopDetector
LoopClosureDetector = LoopDetector

