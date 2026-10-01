from __future__ import annotations

import os

import numpy as np
import open3d as o3d
import trimesh


def write_triangle_mesh_glb(mesh: o3d.geometry.TriangleMesh, path: str) -> bool:
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return False
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    kwargs = {"vertices": vertices, "faces": faces, "process": False}
    if mesh.has_vertex_colors():
        cols = np.asarray(mesh.vertex_colors, dtype=np.float64)
        if cols.ndim == 2 and len(cols) == len(vertices):
            rgb = np.clip(np.round(cols[:, :3] * 255.0), 0, 255).astype(np.uint8)
            kwargs["vertex_colors"] = rgb
    try:
        tm = trimesh.Trimesh(**kwargs)
        tm.export(path, file_type="glb")
    except Exception:
        return False
    return os.path.isfile(path) and os.path.getsize(path) > 0
