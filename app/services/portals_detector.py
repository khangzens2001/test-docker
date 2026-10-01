"""AI Vision Keyframe Portal Detection Module (Doors & Windows).

Provides 2D-to-3D back-projection via camera ray casting onto planar walls,
architectural ground snapping, multi-view 1D IoU spatial fusion, and protocol interfaces.
"""

from __future__ import annotations

import builtins
import copy
import logging
import math
from typing import Any, Protocol, runtime_checkable

import numpy as np

# Export copy to builtins as a safeguard for test suites or callers that use copy without importing it
builtins.copy = copy

logger = logging.getLogger(__name__)


@runtime_checkable
class BasePortalDetector2D(Protocol):
    """Protocol for 2D keyframe portal (door/window) detectors."""

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        """Detect 2D portal bounding boxes from an image file path."""
        ...

    def detect_image(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect 2D portal bounding boxes from a numpy BGR/RGB image array."""
        ...


class MockPortalDetector2D:
    """Mock 2D detector for deterministic, offline testing without heavy neural weights."""

    def __init__(
        self,
        predictions: dict[str, list[dict[str, Any]]] | list[dict[str, Any]] | None = None,
        default_predictions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.predictions = predictions or {}
        self.default_predictions = default_predictions or []

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        if isinstance(self.predictions, dict):
            return self.predictions.get(image_path, self.default_predictions)
        if isinstance(self.predictions, list):
            return self.predictions
        return self.default_predictions

    def detect_image(self, image: np.ndarray) -> list[dict[str, Any]]:
        if isinstance(self.predictions, list):
            return self.predictions
        return self.default_predictions


def _resolve_wall_segments(
    walls: list[dict[str, Any]],
    vertices: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract start/end coordinates, length, tangent, and normal for each wall."""
    resolved = []
    n = len(walls)
    if n == 0:
        return resolved

    # Check if vertices is provided and walls have joints
    has_valid_vertices = False
    if vertices and isinstance(vertices, dict):
        has_valid_vertices = True

    # Check if individual walls already contain coordinates
    first_wall = walls[0]
    has_inline_coords = (
        ("start" in first_wall and "end" in first_wall)
        or ("start_xy" in first_wall and "end_xy" in first_wall)
        or ("p0" in first_wall and "p1" in first_wall)
    )

    if has_inline_coords:
        for i, w in enumerate(walls):
            p0 = np.asarray(w.get("start", w.get("start_xy", w.get("p0"))), dtype=float)[:2]
            p1 = np.asarray(w.get("end", w.get("end_xy", w.get("p1"))), dtype=float)[:2]
            _append_resolved(resolved, w, i, p0, p1)
        return resolved

    if has_valid_vertices:
        all_joints_found = True
        temp_pts = []
        for w in walls:
            joints = w.get("joints") or []
            if len(joints) >= 2 and joints[0] in vertices and joints[1] in vertices:
                p0 = np.asarray(vertices[joints[0]], dtype=float)[:2]
                p1 = np.asarray(vertices[joints[1]], dtype=float)[:2]
                temp_pts.append((p0, p1))
            else:
                all_joints_found = False
                break
        if all_joints_found and len(temp_pts) == n:
            for i, w in enumerate(walls):
                _append_resolved(resolved, w, i, temp_pts[i][0], temp_pts[i][1])
            return resolved

    # Fallback: chain walls orthogonally based on length_meters around a closed polygon
    curr = np.array([0.0, 0.0], dtype=float)
    heading = np.array([1.0, 0.0], dtype=float)  # Wall 0 runs along +X
    for i, w in enumerate(walls):
        length = float(w.get("length_meters", 1.0))
        p0 = curr.copy()
        p1 = p0 + length * heading
        _append_resolved(resolved, w, i, p0, p1)
        curr = p1
        # Turn +90 deg CCW: (hx, hy) -> (-hy, hx)
        heading = np.array([-heading[1], heading[0]], dtype=float)

    return resolved


def _append_resolved(
    resolved: list[dict[str, Any]],
    w: dict[str, Any],
    i: int,
    p0: np.ndarray,
    p1: np.ndarray,
) -> None:
    diff = p1 - p0
    length = float(np.hypot(diff[0], diff[1]))
    if length > 1e-6:
        tangent = diff / length
    else:
        length = float(w.get("length_meters", 1.0))
        tangent = np.array([1.0, 0.0], dtype=float)
    normal_2d = np.array([-tangent[1], tangent[0]], dtype=float)
    height_m = float(w.get("height_meters", 2.8))
    resolved.append({
        "wall_index": int(w.get("wall_index", i)),
        "wall_id": str(w.get("id", f"W{i}")),
        "raw_wall": w,
        "p0": p0,
        "p1": p1,
        "length_m": length,
        "tangent": tangent,
        "normal_2d": normal_2d,
        "height_m": height_m,
    })


def _ray_wall_intersection(
    origin_3d: np.ndarray,
    dir_3d: np.ndarray,
    wall_info: dict[str, Any],
) -> tuple[float, np.ndarray, float, float] | None:
    """Compute ray-wall intersection. Returns (lambda, hit_pt, s_along_wall, denom) or None."""
    n2 = wall_info["normal_2d"]
    norm_3d = np.array([n2[0], n2[1], 0.0], dtype=float)
    p0_3d = np.array([wall_info["p0"][0], wall_info["p0"][1], 0.0], dtype=float)
    d_plane = -float(np.dot(norm_3d, p0_3d))

    denom = float(np.dot(norm_3d, dir_3d))
    if abs(denom) < 1e-6:
        return None

    lam = -float(np.dot(norm_3d, origin_3d) + d_plane) / denom
    if lam <= 0.0:
        return None

    hit_pt = origin_3d + lam * dir_3d
    diff_hit = hit_pt[:2] - wall_info["p0"]
    s = float(np.dot(diff_hit, wall_info["tangent"]))
    return lam, hit_pt, s, denom


def project_2d_box_to_3d_wall(
    bbox_2d: list[float] | tuple[float, float, float, float] | np.ndarray,
    portal_class: str,
    confidence: float,
    camera_k: np.ndarray,
    camera_pose: dict[str, Any],
    wall: dict[str, Any],
    wall_index: int = 0,
    vertices: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Project 2D bounding box [u0, v0, u1, v1] onto 3D wall plane using ray-casting."""
    wall_segs = _resolve_wall_segments([wall], vertices=vertices)
    if not wall_segs:
        return None
    wall_info = wall_segs[0]

    cam_pos = np.asarray(camera_pose.get("position", [0.0, 0.0, 1.4]), dtype=float)
    rot = np.asarray(camera_pose.get("rotation_3x3", np.eye(3)), dtype=float)
    k_inv = np.linalg.inv(camera_k)

    u0, v0, u1, v1 = float(bbox_2d[0]), float(bbox_2d[1]), float(bbox_2d[2]), float(bbox_2d[3])
    corners_2d = [(u0, v0), (u1, v0), (u0, v1), (u1, v1)]

    s_coords = []
    z_coords = []

    for u, v in corners_2d:
        d_cam = k_inv @ np.array([u, v, 1.0], dtype=float)
        # Transform ray direction to world frame using rot.T
        d_world = rot.T @ d_cam
        res = _ray_wall_intersection(cam_pos, d_world, wall_info)
        if res is None:
            return None
        _, hit_pt, s, _ = res
        s_coords.append(s)
        z_coords.append(hit_pt[2])

    s_min = float(min(s_coords))
    s_max = float(max(s_coords))
    z_min = float(min(z_coords))
    z_max = float(max(z_coords))

    # Horizontal span clipping to wall bounds [0, L_wall]
    l_wall = wall_info["length_m"]
    s_min_clip = max(0.0, s_min)
    s_max_clip = min(l_wall, s_max)
    if s_max_clip <= s_min_clip:
        return None

    width_m = s_max_clip - s_min_clip
    p_type = str(portal_class).lower()

    # Enforce door architectural snapping:
    # if class == "door" or z_min <= 0.35m, snap sill_height_m = 0.0 and height_m = z_max
    if p_type == "door" or z_min <= 0.35:
        sill_height_m = 0.0
        height_m = max(0.0, z_max)
    else:
        sill_height_m = max(0.0, z_min)
        height_m = max(0.0, z_max - sill_height_m)

    p0 = wall_info["p0"]
    tang = wall_info["tangent"]
    start_xy = p0 + s_min_clip * tang
    end_xy = p0 + s_max_clip * tang
    center_3d = [
        float((start_xy[0] + end_xy[0]) / 2.0),
        float((start_xy[1] + end_xy[1]) / 2.0),
        float(sill_height_m + height_m / 2.0),
    ]

    return {
        "id": f"P{wall_index}",
        "type": p_type,
        "kind": p_type,
        "wall_index": int(wall.get("wall_index", wall_index)),
        "wall_id": str(wall.get("id", f"W{wall_index}")),
        "offset_along_wall_m": round(float(s_min_clip), 4),
        "width_m": round(float(width_m), 4),
        "height_m": round(float(height_m), 4),
        "sill_height_m": round(float(sill_height_m), 4),
        "start_xy": [round(float(start_xy[0]), 4), round(float(start_xy[1]), 4)],
        "end_xy": [round(float(end_xy[0]), 4), round(float(end_xy[1]), 4)],
        "center_3d": [round(float(c), 4) for c in center_3d],
        "confidence": round(float(confidence), 4),
        "source": "ai_vision",
    }


def compute_1d_iou(p1: dict[str, Any], p2: dict[str, Any]) -> float:
    """Compute 1D horizontal interval Intersection-over-Union along a wall."""
    s0_1 = float(p1.get("offset_along_wall_m", 0.0))
    s1_1 = s0_1 + float(p1.get("width_m", 0.0))
    s0_2 = float(p2.get("offset_along_wall_m", 0.0))
    s1_2 = s0_2 + float(p2.get("width_m", 0.0))

    inter = max(0.0, min(s1_1, s1_2) - max(s0_1, s0_2))
    union = max(s1_1, s1_2) - min(s0_1, s0_2)
    return inter / union if union > 0.0 else 0.0


def _fuse_portal_cluster(
    cluster: list[dict[str, Any]],
    wall_info: dict[str, Any],
) -> dict[str, Any]:
    """Merge a cluster of overlapping 1D detections using confidence-weighted averaging."""
    if len(cluster) == 1:
        return cluster[0]

    weights = [float(p.get("confidence", 0.9)) for p in cluster]
    w_sum = sum(weights)
    if w_sum <= 0:
        weights = [1.0] * len(cluster)
        w_sum = float(len(cluster))

    s_starts = [float(p.get("offset_along_wall_m", 0.0)) for p in cluster]
    widths = [float(p.get("width_m", 0.0)) for p in cluster]
    s_ends = [s + w for s, w in zip(s_starts, widths)]
    sills = [float(p.get("sill_height_m", 0.0)) for p in cluster]
    heights = [float(p.get("height_m", 0.0)) for p in cluster]

    merged_s_start = sum(w * s for w, s in zip(weights, s_starts)) / w_sum
    merged_s_end = sum(w * s for w, s in zip(weights, s_ends)) / w_sum
    merged_sill = sum(w * s for w, s in zip(weights, sills)) / w_sum
    merged_height = sum(w * h for w, h in zip(weights, heights)) / w_sum
    merged_width = max(0.1, merged_s_end - merged_s_start)

    # Multi-view confidence boost
    base_conf = max(weights)
    boosted_conf = min(1.0, base_conf + 0.03 * (len(cluster) - 1))

    p_type = cluster[0]["type"]
    w_idx = cluster[0]["wall_index"]
    w_id = cluster[0].get("wall_id", f"W{w_idx}")

    p0 = wall_info["p0"]
    tang = wall_info["tangent"]
    start_xy = p0 + merged_s_start * tang
    end_xy = p0 + merged_s_end * tang
    center_3d = [
        float((start_xy[0] + end_xy[0]) / 2.0),
        float((start_xy[1] + end_xy[1]) / 2.0),
        float(merged_sill + merged_height / 2.0),
    ]

    return {
        "id": cluster[0]["id"],
        "type": p_type,
        "kind": p_type,
        "wall_index": int(w_idx),
        "wall_id": w_id,
        "offset_along_wall_m": round(float(merged_s_start), 4),
        "width_m": round(float(merged_width), 4),
        "height_m": round(float(merged_height), 4),
        "sill_height_m": round(float(merged_sill), 4),
        "start_xy": [round(float(start_xy[0]), 4), round(float(start_xy[1]), 4)],
        "end_xy": [round(float(end_xy[0]), 4), round(float(end_xy[1]), 4)],
        "center_3d": [round(float(c), 4) for c in center_3d],
        "confidence": round(float(boosted_conf), 4),
        "source": "ai_vision",
    }


def detect_portals_from_keyframes(
    keyframes: list[dict[str, Any]],
    walls: list[dict[str, Any]] | dict[str, Any],
    camera_k: np.ndarray,
    detector: BasePortalDetector2D | None = None,
) -> list[dict[str, Any]]:
    """Detect 3D portals (doors/windows) on room walls from video keyframes."""
    if not keyframes:
        return []

    if isinstance(walls, dict):
        vertices = walls.get("vertices", {})
        walls_list = list(walls.get("walls", []))
    else:
        vertices = None
        walls_list = list(walls)

    if not walls_list:
        return []

    wall_segs = _resolve_wall_segments(walls_list, vertices=vertices)
    if not wall_segs:
        return []

    k_inv = np.linalg.inv(camera_k)
    raw_portals: list[dict[str, Any]] = []

    for kf in keyframes:
        pose = kf.get("camera_pose", {})
        cam_pos = np.asarray(pose.get("position", [0.0, 0.0, 1.4]), dtype=float)
        rot = np.asarray(pose.get("rotation_3x3", np.eye(3)), dtype=float)

        if detector is not None and "image_path" in kf:
            dets = detector.detect(kf["image_path"])
        else:
            dets = kf.get("detections", [])

        for det in dets:
            portal_class = str(det.get("class", det.get("type", "door"))).lower()
            confidence = float(det.get("confidence", 0.9))
            bbox_2d = det.get("bbox_2d", det.get("bbox", []))
            if len(bbox_2d) < 4:
                continue

            u0, v0, u1, v1 = float(bbox_2d[0]), float(bbox_2d[1]), float(bbox_2d[2]), float(bbox_2d[3])
            uc = (u0 + u1) / 2.0
            vc = (v0 + v1) / 2.0
            d_center_cam = k_inv @ np.array([uc, vc, 1.0], dtype=float)
            d_center_world = rot.T @ d_center_cam

            # Find best target wall via center ray
            best_wall_idx = None
            min_dist = float("inf")

            for i, w_info in enumerate(wall_segs):
                res = _ray_wall_intersection(cam_pos, d_center_world, w_info)
                if res is None:
                    continue
                lam, hit_pt, s, denom = res
                # Interior facing normal check: denom must be < 0
                if denom >= -1e-4:
                    continue
                # Center ray hits within wall bounds (with 0.5m margin for edge portals)
                if -0.5 <= s <= w_info["length_m"] + 0.5 and lam < min_dist:
                    min_dist = lam
                    best_wall_idx = i

            # If no wall met center bounds, pick the wall with minimum positive distance facing camera
            if best_wall_idx is None:
                for i, w_info in enumerate(wall_segs):
                    res = _ray_wall_intersection(cam_pos, d_center_world, w_info)
                    if res is None:
                        continue
                    lam, _, _, denom = res
                    if denom < -1e-4 and lam < min_dist:
                        min_dist = lam
                        best_wall_idx = i

            if best_wall_idx is None:
                continue

            # Back project to best wall
            proj = project_2d_box_to_3d_wall(
                bbox_2d=bbox_2d,
                portal_class=portal_class,
                confidence=confidence,
                camera_k=camera_k,
                camera_pose=pose,
                wall=walls_list[best_wall_idx],
                wall_index=best_wall_idx,
                vertices=vertices,
            )
            if proj is not None:
                raw_portals.append(proj)

    if not raw_portals:
        return []

    # 1D IoU Multi-view Fusion:
    # Cluster detections of the same class on the same wall whose intervals overlap with 1D IoU >= 0.40
    clusters: list[list[dict[str, Any]]] = []

    for p in raw_portals:
        matched_cluster = None
        for c in clusters:
            if c[0]["wall_index"] == p["wall_index"] and c[0]["type"] == p["type"]:
                # Check 1D IoU against any member of cluster
                if any(compute_1d_iou(p, member) >= 0.40 for member in c):
                    matched_cluster = c
                    break
        if matched_cluster is not None:
            matched_cluster.append(p)
        else:
            clusters.append([p])

    fused_portals = []
    wall_map = {w_info["wall_index"]: w_info for w_info in wall_segs}

    for i, c in enumerate(clusters):
        w_idx = c[0]["wall_index"]
        w_info = wall_map.get(w_idx, wall_segs[0])
        fused = _fuse_portal_cluster(c, w_info)
        fused["id"] = f"P{i}"
        fused_portals.append(fused)

    return fused_portals
