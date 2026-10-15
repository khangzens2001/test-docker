from __future__ import annotations

import logging
import os

import numpy as np
import open3d as o3d

import trimesh

from app.services.glb_export import write_scene_glb, write_triangle_mesh_glb
from app.services.visual_cloud import layout_height_m, layout_xy_polygon

logger = logging.getLogger(__name__)

ROOM_MODEL_RGB = (230, 230, 230)
WALL_THICKNESS_DEFAULT_M = 0.12
FIXTURE_RGB = (195, 208, 222)
FLOOR_RGB = (218, 214, 206)



def z_up_to_gltf_y_up(xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0]
    out[:, 1] = pts[:, 2]
    out[:, 2] = -pts[:, 1]
    return out


def _signed_area(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _point_in_triangle(p, a, b, c) -> bool:
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-18:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    return u >= -1e-9 and v >= -1e-9 and (u + v) <= 1.0 + 1e-9


def triangulate_floor_xy(xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int32)
    ccw = _signed_area(pts) > 0.0
    idx = list(range(n))
    faces: list[list[int]] = []
    guard = 0
    while len(idx) > 3 and guard < max(64, 4 * n):
        guard += 1
        m = len(idx)
        ear = None
        for k in range(m):
            i_prev, i, i_next = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = pts[i_prev], pts[i], pts[i_next]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            is_ccw = cross > 1e-12
            is_cw = cross < -1e-12
            if ccw and not is_ccw:
                continue
            if (not ccw) and not is_cw:
                continue
            occupied = False
            for j in idx:
                if j in (i_prev, i, i_next):
                    continue
                if _point_in_triangle(pts[j], a, b, c):
                    occupied = True
                    break
            if occupied:
                continue
            ear = k
            faces.append([i_prev, i, i_next])
            break
        if ear is None:
            break
        del idx[ear]
    if len(idx) == 3:
        faces.append(idx)
    return np.asarray(faces, dtype=np.int32)


def build_room_model_mesh(layout: dict) -> o3d.geometry.TriangleMesh:
    empty = o3d.geometry.TriangleMesh()
    xy = layout_xy_polygon(layout)
    if len(xy) < 4:
        return empty
    height_m = layout_height_m(layout)
    n = len(xy)
    verts = []
    faces = []
    portals = layout.get("portals") or []
    portals_by_wall: dict[int, list[dict]] = {}
    for p in portals:
        w_idx = p.get("wall_index")
        if w_idx is not None:
            portals_by_wall.setdefault(int(w_idx), []).append(p)

    def _add_panel(s0: float, s1: float, z0: float, z1: float, a_pt: np.ndarray, t_vec: np.ndarray) -> None:
        if s1 - s0 < 1e-4 or z1 - z0 < 1e-4:
            return
        p_bot_a = [float(a_pt[0] + s0 * t_vec[0]), float(a_pt[1] + s0 * t_vec[1]), float(z0)]
        p_bot_b = [float(a_pt[0] + s1 * t_vec[0]), float(a_pt[1] + s1 * t_vec[1]), float(z0)]
        p_top_b = [float(a_pt[0] + s1 * t_vec[0]), float(a_pt[1] + s1 * t_vec[1]), float(z1)]
        p_top_a = [float(a_pt[0] + s0 * t_vec[0]), float(a_pt[1] + s0 * t_vec[1]), float(z1)]
        idx = len(verts)
        verts.append(p_bot_a)
        verts.append(p_bot_b)
        verts.append(p_top_b)
        verts.append(p_top_a)
        faces.append([idx + 0, idx + 2, idx + 1])
        faces.append([idx + 0, idx + 3, idx + 2])

    for i in range(n):
        a = xy[i]
        b = xy[(i + 1) % n]
        diff = b - a
        length = float(np.hypot(diff[0], diff[1]))
        if length > 1e-6:
            tang = diff / length
        else:
            tang = np.array([1.0, 0.0], dtype=float)

        w_portals = portals_by_wall.get(i, [])
        if not w_portals:
            _add_panel(0.0, length, 0.0, height_m, a, tang)
        else:
            # Sort portals by horizontal offset along wall
            parsed_portals = []
            for p in w_portals:
                if "offset_along_wall_m" in p:
                    s0 = float(p["offset_along_wall_m"])
                elif "start_xy" in p:
                    s0 = float(np.dot(np.asarray(p["start_xy"][:2], dtype=float) - a, tang))
                else:
                    s0 = 0.0
                width = float(p.get("width_m", 0.9))
                s1 = s0 + width
                s0_clip = max(0.0, min(length, s0))
                s1_clip = max(0.0, min(length, s1))
                if s1_clip > s0_clip + 1e-4:
                    z_sill = max(0.0, min(height_m, float(p.get("sill_height_m", 0.0))))
                    p_h = float(p.get("height_m", 2.1))
                    z_top = max(z_sill, min(height_m, z_sill + p_h))
                    parsed_portals.append((s0_clip, s1_clip, z_sill, z_top))

            parsed_portals.sort(key=lambda item: item[0])

            curr_s = 0.0
            for s0_p, s1_p, z_sill, z_top in parsed_portals:
                # Left / intermediate panel
                if s0_p > curr_s + 1e-4:
                    _add_panel(curr_s, s0_p, 0.0, height_m, a, tang)
                # Sill panel (e.g. for window with elevated sill)
                if z_sill > 1e-4:
                    _add_panel(s0_p, s1_p, 0.0, z_sill, a, tang)
                # Lintel header panel
                if height_m > z_top + 1e-4:
                    _add_panel(s0_p, s1_p, z_top, height_m, a, tang)
                curr_s = max(curr_s, s1_p)

            # Right trailing panel
            if length > curr_s + 1e-4:
                _add_panel(curr_s, length, 0.0, height_m, a, tang)
    base = len(verts)
    for p in xy:
        verts.append([float(p[0]), float(p[1]), 0.0])
    for tri in triangulate_floor_xy(xy):
        faces.append([base + int(tri[0]), base + int(tri[1]), base + int(tri[2])])
    structural_vert_count = len(verts)

    obstacles = layout.get("obstacles") if isinstance(layout, dict) else None
    if isinstance(obstacles, list):
        for obs in obstacles:
            if not isinstance(obs, dict):
                continue
            bmin = obs.get("bbox_min")
            bmax = obs.get("bbox_max")
            if (
                not isinstance(bmin, (list, tuple))
                or not isinstance(bmax, (list, tuple))
                or len(bmin) < 3
                or len(bmax) < 3
            ):
                continue
            try:
                x0, y0, z0 = float(bmin[0]), float(bmin[1]), max(0.0, float(bmin[2]))
                x1, y1, z1 = float(bmax[0]), float(bmax[1]), min(height_m - 0.10, float(bmax[2]))
            except (TypeError, ValueError):
                continue
            if x1 - x0 < 0.05 or y1 - y0 < 0.05 or z1 - z0 < 0.05:
                continue
            ob = len(verts)
            verts.extend(
                [
                    [x0, y0, z0],
                    [x1, y0, z0],
                    [x1, y1, z0],
                    [x0, y1, z0],
                    [x0, y0, z1],
                    [x1, y0, z1],
                    [x1, y1, z1],
                    [x0, y1, z1],
                ]
            )
            faces.extend(
                [
                    # Front (+Y / -Y walls)
                    [ob + 0, ob + 1, ob + 5],
                    [ob + 0, ob + 5, ob + 4],
                    [ob + 1, ob + 2, ob + 6],
                    [ob + 1, ob + 6, ob + 5],
                    [ob + 2, ob + 3, ob + 7],
                    [ob + 2, ob + 7, ob + 6],
                    [ob + 3, ob + 0, ob + 4],
                    [ob + 3, ob + 4, ob + 7],
                    # Top face
                    [ob + 4, ob + 5, ob + 6],
                    [ob + 4, ob + 6, ob + 7],
                    # Bottom face
                    [ob + 0, ob + 2, ob + 1],
                    [ob + 0, ob + 3, ob + 2],
                ]
            )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    rgb = np.tile(np.array(ROOM_MODEL_RGB, dtype=float) / 255.0, (len(verts), 1))
    if len(verts) > structural_vert_count:
        fixture_rgb = np.array((195.0, 208.0, 222.0), dtype=float) / 255.0
        rgb[structural_vert_count:] = fixture_rgb
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)
    mesh.compute_vertex_normals()
    return mesh


def build_room_model_scene(
    layout: dict,
    wall_thickness_m: float = WALL_THICKNESS_DEFAULT_M,
) -> trimesh.Scene:
    """Build an Apple RoomPlan-grade multi-node glTF Scene with 3D thick walls,
    continuous floor slab, portal openings, and distinct interior obstacle nodes."""
    scene = trimesh.Scene()
    xy = layout_xy_polygon(layout)
    if len(xy) < 4:
        return scene
    height_m = layout_height_m(layout)
    n = len(xy)
    ccw = _signed_area(xy) > 0.0
    t_wall = max(0.0, float(wall_thickness_m))

    # 1. Floor node (continuous across entire room polygon, z=0)
    floor_verts = [[float(p[0]), float(p[1]), 0.0] for p in xy]
    floor_faces = triangulate_floor_xy(xy).tolist()
    if floor_verts and floor_faces:
        f_xyz = z_up_to_gltf_y_up(np.asarray(floor_verts, dtype=float))
        f_cols = np.tile(np.array([*FLOOR_RGB, 255], dtype=np.uint8), (len(f_xyz), 1))
        tm_floor = trimesh.Trimesh(
            vertices=f_xyz,
            faces=np.asarray(floor_faces, dtype=np.int64),
            vertex_colors=f_cols,
            process=False,
        )
        scene.add_geometry(tm_floor, node_name="floor", geom_name="floor")

    # 2. 3D Volumetric Walls node (inner face, outer face, top/bottom caps, jamb reveals)
    portals = layout.get("portals") or []
    portals_by_wall: dict[int, list[dict]] = {}
    for p in portals:
        w_idx = p.get("wall_index")
        if w_idx is not None:
            portals_by_wall.setdefault(int(w_idx), []).append(p)

    wall_verts: list[list[float]] = []
    wall_faces: list[list[int]] = []

    def _add_wall_block(
        s0: float,
        s1: float,
        z0: float,
        z1: float,
        a_pt: np.ndarray,
        t_vec: np.ndarray,
        n_out: np.ndarray,
        ext_s0: float = 0.0,
        ext_s1: float = 0.0,
    ) -> None:
        if s1 - s0 < 1e-4 or z1 - z0 < 1e-4:
            return
        p0_in = a_pt + s0 * t_vec
        p1_in = a_pt + s1 * t_vec
        if t_wall <= 1e-4:
            idx = len(wall_verts)
            wall_verts.extend(
                [
                    [float(p0_in[0]), float(p0_in[1]), float(z0)],
                    [float(p1_in[0]), float(p1_in[1]), float(z0)],
                    [float(p1_in[0]), float(p1_in[1]), float(z1)],
                    [float(p0_in[0]), float(p0_in[1]), float(z1)],
                ]
            )
            wall_faces.extend([[idx + 0, idx + 2, idx + 1], [idx + 0, idx + 3, idx + 2]])
            return
        p0_out = (a_pt + (s0 - ext_s0) * t_vec) + t_wall * n_out
        p1_out = (a_pt + (s1 + ext_s1) * t_vec) + t_wall * n_out
        idx = len(wall_verts)
        wall_verts.extend(
            [
                # 0..3: inner face (bottom-left, bottom-right, top-right, top-left)
                [float(p0_in[0]), float(p0_in[1]), float(z0)],
                [float(p1_in[0]), float(p1_in[1]), float(z0)],
                [float(p1_in[0]), float(p1_in[1]), float(z1)],
                [float(p0_in[0]), float(p0_in[1]), float(z1)],
                # 4..7: outer face (bottom-left, bottom-right, top-right, top-left)
                [float(p0_out[0]), float(p0_out[1]), float(z0)],
                [float(p1_out[0]), float(p1_out[1]), float(z0)],
                [float(p1_out[0]), float(p1_out[1]), float(z1)],
                [float(p0_out[0]), float(p0_out[1]), float(z1)],
            ]
        )
        wall_faces.extend(
            [
                # Inner face
                [idx + 0, idx + 1, idx + 2],
                [idx + 0, idx + 2, idx + 3],
                # Outer face
                [idx + 4, idx + 6, idx + 5],
                [idx + 4, idx + 7, idx + 6],
                # Top cap
                [idx + 3, idx + 2, idx + 6],
                [idx + 3, idx + 6, idx + 7],
                # Bottom cap
                [idx + 0, idx + 5, idx + 1],
                [idx + 0, idx + 4, idx + 5],
                # Left jamb / end cap
                [idx + 0, idx + 3, idx + 7],
                [idx + 0, idx + 7, idx + 4],
                # Right jamb / end cap
                [idx + 1, idx + 6, idx + 2],
                [idx + 1, idx + 5, idx + 6],
            ]
        )

    for i in range(n):
        a = xy[i]
        b = xy[(i + 1) % n]
        diff = b - a
        length = float(np.hypot(diff[0], diff[1]))
        if length <= 1e-6:
            continue
        tang = diff / length
        n_out = np.array([tang[1], -tang[0]], dtype=float) if ccw else np.array([-tang[1], tang[0]], dtype=float)
        w_portals = portals_by_wall.get(i, [])
        if not w_portals:
            _add_wall_block(0.0, length, 0.0, height_m, a, tang, n_out, ext_s0=t_wall, ext_s1=t_wall)
        else:
            parsed_portals = []
            for p in w_portals:
                if "offset_along_wall_m" in p:
                    s0 = float(p["offset_along_wall_m"])
                elif "start_xy" in p:
                    s0 = float(np.dot(np.asarray(p["start_xy"][:2], dtype=float) - a, tang))
                else:
                    s0 = 0.0
                width = float(p.get("width_m", 0.9))
                s1 = s0 + width
                s0_clip = max(0.0, min(length, s0))
                s1_clip = max(0.0, min(length, s1))
                if s1_clip > s0_clip + 1e-4:
                    z_sill = max(0.0, min(height_m, float(p.get("sill_height_m", 0.0))))
                    p_h = float(p.get("height_m", 2.1))
                    z_top = max(z_sill, min(height_m, z_sill + p_h))
                    parsed_portals.append((s0_clip, s1_clip, z_sill, z_top))
            parsed_portals.sort(key=lambda item: item[0])
            curr_s = 0.0
            for s0_p, s1_p, z_sill, z_top in parsed_portals:
                if s0_p > curr_s + 1e-4:
                    _add_wall_block(
                        curr_s,
                        s0_p,
                        0.0,
                        height_m,
                        a,
                        tang,
                        n_out,
                        ext_s0=(t_wall if curr_s <= 1e-4 else 0.0),
                        ext_s1=0.0,
                    )
                if z_sill > 1e-4:
                    _add_wall_block(s0_p, s1_p, 0.0, z_sill, a, tang, n_out)
                if height_m > z_top + 1e-4:
                    _add_wall_block(s0_p, s1_p, z_top, height_m, a, tang, n_out)
                curr_s = max(curr_s, s1_p)
            if length > curr_s + 1e-4:
                _add_wall_block(
                    curr_s,
                    length,
                    0.0,
                    height_m,
                    a,
                    tang,
                    n_out,
                    ext_s0=0.0,
                    ext_s1=t_wall,
                )

    if wall_verts and wall_faces:
        w_xyz = z_up_to_gltf_y_up(np.asarray(wall_verts, dtype=float))
        w_cols = np.tile(np.array([*ROOM_MODEL_RGB, 255], dtype=np.uint8), (len(w_xyz), 1))
        tm_walls = trimesh.Trimesh(
            vertices=w_xyz,
            faces=np.asarray(wall_faces, dtype=np.int64),
            vertex_colors=w_cols,
            process=False,
        )
        scene.add_geometry(tm_walls, node_name="walls", geom_name="walls")

    # 3. Interior Obstacles / Fixtures nodes (distinct scene nodes and materials)
    obstacles = layout.get("obstacles") if isinstance(layout, dict) else None
    if isinstance(obstacles, list):
        for obs_idx, obs in enumerate(obstacles):
            if not isinstance(obs, dict):
                continue
            bmin = obs.get("bbox_min")
            bmax = obs.get("bbox_max")
            if (
                not isinstance(bmin, (list, tuple))
                or not isinstance(bmax, (list, tuple))
                or len(bmin) < 3
                or len(bmax) < 3
            ):
                continue
            try:
                x0, y0, z0 = float(bmin[0]), float(bmin[1]), max(0.0, float(bmin[2]))
                x1, y1, z1 = float(bmax[0]), float(bmax[1]), min(height_m - 0.10, float(bmax[2]))
            except (TypeError, ValueError):
                continue
            if x1 - x0 < 0.05 or y1 - y0 < 0.05 or z1 - z0 < 0.05:
                continue
            obs_verts = np.array(
                [
                    [x0, y0, z0],
                    [x1, y0, z0],
                    [x1, y1, z0],
                    [x0, y1, z0],
                    [x0, y0, z1],
                    [x1, y0, z1],
                    [x1, y1, z1],
                    [x0, y1, z1],
                ],
                dtype=float,
            )
            obs_faces = np.array(
                [
                    [0, 1, 5],
                    [0, 5, 4],
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 0, 4],
                    [3, 4, 7],
                    [4, 5, 6],
                    [4, 6, 7],
                    [0, 2, 1],
                    [0, 3, 2],
                ],
                dtype=np.int64,
            )
            o_xyz = z_up_to_gltf_y_up(obs_verts)
            o_cols = np.tile(np.array([*FIXTURE_RGB, 255], dtype=np.uint8), (len(o_xyz), 1))
            node_id = str(obs.get("id") or f"obstacle_{obs_idx}")
            tm_obs = trimesh.Trimesh(
                vertices=o_xyz,
                faces=obs_faces,
                vertex_colors=o_cols,
                process=False,
            )
            scene.add_geometry(tm_obs, node_name=node_id, geom_name=node_id)

    return scene


def export_room_model_glb(layout: dict, session_dir: str) -> None:
    try:
        scene = build_room_model_scene(layout)
        if len(scene.geometry) == 0:
            return
        os.makedirs(session_dir, exist_ok=True)
        tmp = os.path.join(session_dir, "room_model.tmp.glb")
        final = os.path.join(session_dir, "room_model.glb")
        try:
            ok = write_scene_glb(scene, tmp)
            if not ok:
                mesh = build_room_model_mesh(layout)
                if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                    return
                xyz = z_up_to_gltf_y_up(np.asarray(mesh.vertices, dtype=float))
                mesh.vertices = o3d.utility.Vector3dVector(xyz)
                ok = write_triangle_mesh_glb(mesh, tmp)
            if not ok or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
                return
            os.replace(tmp, final)
        finally:
            if os.path.isfile(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
    except Exception:
        logger.exception("export_room_model_glb failed")

