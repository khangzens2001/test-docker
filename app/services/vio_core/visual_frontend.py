"""Visual Frontend with CLAHE contrast enhancement, FAST/Shi-Tomasi detection,
ANMS-SSC grid bucketing feature selection, and Gyro-predictive KLT optical flow.
"""

from __future__ import annotations
import numpy as np
import cv2
from app.services.vio_core.feature_selector import select_anms_ssc


class VisualFrontend:
    MIN_FEATURES = 80
    MAX_FEATURES = 300
    DECIMATION_CAP = 150

    def __init__(
        self,
        intrinsics: np.ndarray,
        dist_coeffs: np.ndarray | None = None,
        max_features: int = 300,
        min_features: int = 80,
        decimation_cap: int = 150,
    ):
        self.K = intrinsics.astype(np.float64)
        self.K_inv = np.linalg.inv(self.K)
        self.dist_coeffs = dist_coeffs
        self.max_features = max_features
        self.min_features = min_features
        self.decimation_cap = decimation_cap

        # CLAHE Contrast Enhancement
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # FAST feature detector with non-max suppression
        self._fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

        # Lucas-Kanade optical flow parameters
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray = np.empty((0, 1, 2), dtype=np.float32)
        self._frame_id: int = -1
        self._tracks: list[list[tuple]] = []

    def process_frame(
        self, gray: np.ndarray, R_prev_curr: np.ndarray | None = None
    ) -> dict:
        """Process incoming grayscale frame, perform CLAHE, Gyro-predictive KLT tracking,
        and ANMS-SSC feature replenishment.

        Args:
            gray: Grayscale image uint8.
            R_prev_curr: 3x3 rotation matrix step from previous frame to current frame.

        Returns:
            Dictionary containing tracking results:
            - matched_pts_prev: (M, 2) previous feature locations
            - matched_pts_curr: (M, 2) current matched feature locations
            - num_tracked: int
            - new_features_injected: int
            - frame_id: int
        """
        self._frame_id += 1

        # Apply CLAHE contrast enhancement
        gray_prep = self._clahe.apply(gray)

        # Lens distortion correction if coefficients provided
        if self.dist_coeffs is not None:
            gray_prep = cv2.undistort(gray_prep, self.K, self.dist_coeffs)

        h, w = gray_prep.shape[:2]
        result = {
            "matched_pts_prev": np.empty((0, 2), dtype=np.float32),
            "matched_pts_curr": np.empty((0, 2), dtype=np.float32),
            "num_tracked": 0,
            "new_features_injected": 0,
            "frame_id": self._frame_id,
        }

        # Track features if previous frame exists
        if self._prev_gray is not None and len(self._prev_pts) > 0:
            if R_prev_curr is not None:
                # Gyro initial flow prediction: p_curr = K R K^-1 p_prev
                pts_prev_flat = self._prev_pts.reshape(-1, 2)
                pts_hom = np.hstack([pts_prev_flat, np.ones((len(pts_prev_flat), 1))])
                H_gyro = self.K @ R_prev_curr @ self.K_inv
                proj_hom = (H_gyro @ pts_hom.T).T
                z = np.clip(proj_hom[:, 2:], 1e-6, None)
                p_pred = (proj_hom[:, :2] / z).astype(np.float32)

                next_pts_guess = np.ascontiguousarray(p_pred.reshape(-1, 1, 2), dtype=np.float32)
                fwd, st_f, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray,
                    gray_prep,
                    self._prev_pts,
                    next_pts_guess,
                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                    **self._lk_params,
                )
            else:
                fwd, st_f, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray_prep, self._prev_pts, None, **self._lk_params
                )

            # Backward flow check for robust tracking validation
            bwd, st_b, _ = cv2.calcOpticalFlowPyrLK(
                gray_prep, self._prev_gray, fwd, None, **self._lk_params
            )
            fb_err = np.linalg.norm((self._prev_pts - bwd).reshape(-1, 2), axis=1)

            good = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (fb_err < 1.0)

            if np.sum(good) >= 8:
                pts_prev = self._prev_pts[good].reshape(-1, 2)
                pts_curr = fwd[good].reshape(-1, 2)

                # RANSAC essential matrix filter
                mask = self._ransac_filter(pts_prev, pts_curr)
                pts_prev = pts_prev[mask]
                pts_curr = pts_curr[mask]

                result["matched_pts_prev"] = pts_prev
                result["matched_pts_curr"] = pts_curr
                result["num_tracked"] = len(pts_curr)
                self._prev_pts = pts_curr.reshape(-1, 1, 2).astype(np.float32)
            else:
                self._prev_pts = np.empty((0, 1, 2), dtype=np.float32)

        # Check keyframe decimation cap (force replenishment / decimation every DECIMATION_CAP frames)
        is_decimation_frame = (self._frame_id > 0) and (self._frame_id % self.decimation_cap == 0)

        # Feature replenishment when tracked count drops below MIN_FEATURES or on decimation frame
        if len(self._prev_pts) < self.min_features or is_decimation_frame:
            n_before = len(self._prev_pts)
            existing_pts = self._prev_pts.reshape(-1, 2) if n_before > 0 else np.empty((0, 2), dtype=np.float32)

            new_pts = self._detect_and_select_anms(gray_prep, existing_pts)

            if len(existing_pts) > 0:
                combined = np.vstack([existing_pts, new_pts])
            else:
                combined = new_pts

            self._prev_pts = combined[: self.max_features].reshape(-1, 1, 2).astype(np.float32)
            result["new_features_injected"] = len(self._prev_pts) - n_before

        self._prev_gray = gray_prep.copy()
        return result

    def _detect_and_select_anms(
        self, gray: np.ndarray, existing_pts: np.ndarray
    ) -> np.ndarray:
        """Detect keypoints using FAST / Shi-Tomasi and select using ANMS-SSC grid bucketing."""
        h, w = gray.shape[:2]

        # Try FAST detector first
        kps = self._fast.detect(gray, None)
        if kps:
            pts = np.array([k.pt for k in kps], dtype=np.float32)
            responses = np.array([k.response for k in kps], dtype=np.float32)
        else:
            # Fallback to Shi-Tomasi corners
            corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=1000, qualityLevel=0.01, minDistance=5
            )
            if corners is None:
                return np.empty((0, 2), dtype=np.float32)
            pts = corners.reshape(-1, 2).astype(np.float32)
            responses = np.ones(len(pts), dtype=np.float32)

        # Apply ANMS-SSC grid bucketing selection (6x8 grid)
        selected_indices = select_anms_ssc(
            pts,
            responses,
            image_size=(w, h),
            max_features=self.max_features,
            min_features=self.min_features,
            num_grid_rows=6,
            num_grid_cols=8,
        )
        selected_pts = pts[selected_indices]

        # Distance filter against existing tracked features (min distance 10 px)
        if len(existing_pts) > 0 and len(selected_pts) > 0:
            dists = np.min(
                np.linalg.norm(
                    selected_pts[:, None, :] - existing_pts[None, :, :], axis=2
                ),
                axis=1,
            )
            selected_pts = selected_pts[dists > 10.0]

        quota = max(0, self.max_features - len(existing_pts))
        return selected_pts[:quota]

    def _ransac_filter(self, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """Perform RANSAC filtering using Essential Matrix to remove flow outliers."""
        if len(pts1) < 8:
            return np.ones(len(pts1), dtype=bool)

        _, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if mask is not None:
            return mask.ravel().astype(bool)
        return np.ones(len(pts1), dtype=bool)

    def get_tracks(self) -> list[list[tuple]]:
        return self._tracks

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_pts = np.empty((0, 1, 2), dtype=np.float32)
        self._frame_id = -1
        self._tracks = []
