import copy
import gc
import logging
import os
import numpy as np
import open3d as o3d

from app.services.glb_export import write_triangle_mesh_glb

logger = logging.getLogger(__name__)

VOXEL_LENGTH = 0.02
SDF_TRUNC = 0.12
DEPTH_TRUNC = 8.01
RADIAL_GAMMA = 1.0
RADIAL_WEIGHT_TAU = 0.50
QEM_TRIANGLE_LIMIT = 15000
POINT_CLOUD_SAMPLE_POINTS = 300000
ODOMETRY_RGB_MATCH_NS = 50_000_000
WHITEFLAT_RGB = (245, 245, 245)


def radial_confidence_weight(height: int, width: int, K: np.ndarray) -> np.ndarray:
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    r_max = 0.5 * np.sqrt(float(width) ** 2 + float(height) ** 2)
    r2 = (u - cx) ** 2 + (v - cy) ** 2
    weight = np.exp(-RADIAL_GAMMA * r2 / (r_max ** 2)).astype(np.float32)
    return weight


def apply_radial_gate(depth: np.ndarray, weight: np.ndarray) -> np.ndarray:
    if weight.shape != depth.shape:
        raise ValueError("Radial gate shape mismatch.")
    out = np.asarray(depth, dtype=np.float32).copy()
    out[weight < RADIAL_WEIGHT_TAU] = 0.0
    return out


def _decimate_if_needed(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) > QEM_TRIANGLE_LIMIT:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=QEM_TRIANGLE_LIMIT
        )
    return mesh


def create_tsdf_grid(
    poses: list[np.ndarray],
    depths: list[np.ndarray] | None = None,
    intrinsics: np.ndarray | None = None,
    bounds_min: np.ndarray | None = None,
    bounds_max: np.ndarray | None = None,
    padding: float = 0.4,
    voxel_length: float = VOXEL_LENGTH,
    sdf_trunc: float = SDF_TRUNC,
    max_grid_res: int = 384,
) -> o3d.pipelines.integration.UniformTSDFVolume:
    """Create a tight UniformTSDFVolume bounded by observed depth / trajectory,
    preserving ~1.5-2.5cm voxel resolution without arbitrary 5m inflation."""
    if bounds_min is not None and bounds_max is not None:
        p_min = np.asarray(bounds_min, dtype=float) - padding
        p_max = np.asarray(bounds_max, dtype=float) + padding
    else:
        sample_pts = []
        if (
            depths is not None
            and intrinsics is not None
            and len(depths) > 0
            and len(poses) == len(depths)
        ):
            fx = float(intrinsics[0, 0])
            fy = float(intrinsics[1, 1])
            cx = float(intrinsics[0, 2])
            cy = float(intrinsics[1, 2])
            n_frames = len(depths)
            stride_f = max(1, n_frames // 20)
            for f_idx in range(0, n_frames, stride_f):
                d_img = depths[f_idx]
                if d_img is None or d_img.size == 0:
                    continue
                pose = poses[f_idx]
                R = pose[0:3, 0:3]
                t = pose[0:3, 3]
                h_d, w_d = d_img.shape
                step = max(8, min(h_d, w_d) // 16)
                rv, cv = np.mgrid[step // 2 : h_d : step, step // 2 : w_d : step]
                rv_f = rv.ravel()
                cv_f = cv.ravel()
                z_vals = d_img[rv_f, cv_f]
                valid = (z_vals > 0.2) & (z_vals < DEPTH_TRUNC) & np.isfinite(z_vals)
                if np.any(valid):
                    z_v = z_vals[valid]
                    x_c = (cv_f[valid] - cx) * z_v / fx
                    y_c = (rv_f[valid] - cy) * z_v / fy
                    p_cam = np.column_stack([x_c, y_c, z_v])
                    p_world = (R @ p_cam.T).T + t
                    sample_pts.append(p_world)

        all_t = np.array([p[0:3, 3] for p in poses], dtype=float) if poses else np.zeros((0, 3))
        if sample_pts:
            pts_cat = np.vstack(sample_pts)
            p_min = np.percentile(pts_cat, 0.5, axis=0) - padding
            p_max = np.percentile(pts_cat, 99.5, axis=0) + padding
            if len(all_t) > 0:
                p_min = np.minimum(p_min, all_t.min(axis=0) - padding)
                p_max = np.maximum(p_max, all_t.max(axis=0) + padding)
        elif len(all_t) > 0:
            all_fwd = np.array(
                [p[0:3, 3] + p[0:3, :3] @ np.array([0.0, 0.0, 1.8]) for p in poses],
                dtype=float,
            )
            pts_box = np.vstack([all_t, all_fwd])
            p_min = pts_box.min(axis=0) - padding
            p_max = pts_box.max(axis=0) + padding
        else:
            p_min = np.array([-1.5, -1.5, -1.5])
            p_max = np.array([1.5, 1.5, 1.5])

    span = float(np.max(p_max - p_min))
    length = max(span, 1.0)
    center = 0.5 * (p_min + p_max)
    origin = center - 0.5 * length
    res = int(np.clip(round(length / voxel_length), 64, max_grid_res))
    return o3d.pipelines.integration.UniformTSDFVolume(
        length,
        res,
        sdf_trunc,
        o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin,
    )


class ReconstructionService:
    def integrate_tsdf(
        self,
        depths: list[np.ndarray],
        poses: list[np.ndarray],
        intrinsics: np.ndarray,
        colors: list[np.ndarray],
        bounds_min: np.ndarray | None = None,
        bounds_max: np.ndarray | None = None,
        max_grid_res: int = 384,
    ) -> o3d.geometry.TriangleMesh:
        if len(colors) != len(depths) or len(poses) != len(depths):
            raise ValueError("TSDF input lengths do not match.")
        if not depths:
            return o3d.geometry.TriangleMesh()

        height, width = depths[0].shape
        weight = radial_confidence_weight(height, width, intrinsics)
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])
        if not (0.0 <= cx < width) or not (0.0 <= cy < height):
            logger.warning("Principal point (cx, cy) is outside the depth map frame.")

        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )

        def _run_integrate(vol):
            for i, (depth, pose, color) in enumerate(zip(depths, poses, colors)):
                if depth.shape != (height, width):
                    raise ValueError("TSDF input shapes do not match.")
                if color.shape != (height, width, 3):
                    raise ValueError("TSDF input shapes do not match.")
                depth_gated = apply_radial_gate(depth, weight)
                color_img = o3d.geometry.Image(
                    np.ascontiguousarray(color, dtype=np.uint8)
                )
                depth_img = o3d.geometry.Image(
                    np.ascontiguousarray(depth_gated, dtype=np.float32)
                )
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_img,
                    depth_img,
                    depth_scale=1.0,
                    depth_trunc=DEPTH_TRUNC,
                    convert_rgb_to_intensity=False,
                )
                R = pose[0:3, 0:3]
                t = pose[0:3, 3]
                pose_inv = np.eye(4)
                pose_inv[0:3, 0:3] = R.T
                pose_inv[0:3, 3] = -R.T @ t
                vol.integrate(rgbd, o3d_intrinsic, pose_inv)
                del color_img, depth_img, rgbd
            m = vol.extract_triangle_mesh()
            vol.reset()
            return m

        volume = create_tsdf_grid(
            poses,
            depths=depths,
            intrinsics=intrinsics,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            max_grid_res=max_grid_res,
        )
        mesh = _run_integrate(volume)

        for i in range(len(depths)):
            depths[i] = None
            colors[i] = None
        gc.collect()
        return mesh

    def export_mesh_artifacts(
        self, mesh: o3d.geometry.TriangleMesh, session_dir: str
    ) -> None:
        glb_tmp = os.path.join(session_dir, "reconstructed.tmp.glb")
        ply_tmp = os.path.join(session_dir, "reconstructed.tmp.ply")
        glb_final = os.path.join(session_dir, "reconstructed.glb")
        ply_final = os.path.join(session_dir, "reconstructed.ply")
        try:
            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                raise ValueError("TSDF produced empty mesh.")
            export_mesh = copy.deepcopy(mesh)
            if not export_mesh.has_vertex_normals():
                export_mesh.compute_vertex_normals()
            export_mesh = _decimate_if_needed(export_mesh)
            ok_glb = write_triangle_mesh_glb(export_mesh, glb_tmp)
            o3d.utility.random.seed(42)
            pcd = export_mesh.sample_points_uniformly(
                number_of_points=POINT_CLOUD_SAMPLE_POINTS
            )
            if not pcd.has_colors():
                raise ValueError("TSDF produced empty mesh.")
            pcd.normals = o3d.utility.Vector3dVector()
            ok_ply = o3d.io.write_point_cloud(ply_tmp, pcd, write_ascii=False)
            if (
                not ok_glb
                or not ok_ply
                or not os.path.isfile(glb_tmp)
                or not os.path.isfile(ply_tmp)
                or os.path.getsize(glb_tmp) == 0
                or os.path.getsize(ply_tmp) == 0
            ):
                raise ValueError("Failed to write mesh artifacts.")
            os.replace(glb_tmp, glb_final)
            os.replace(ply_tmp, ply_final)
        finally:
            for tmp in (glb_tmp, ply_tmp):
                try:
                    if os.path.isfile(tmp):
                        os.unlink(tmp)
                except OSError:
                    pass

    def segment_planes(
        self,
        mesh: o3d.geometry.TriangleMesh,
        gravity: np.ndarray | None,
        trajectory: np.ndarray | None = None,
        *,
        vggt_prior=None,
        session_dir: str | None = None,
    ) -> tuple[dict, o3d.geometry.PointCloud]:
        from app.services.plane_segmentation import (
            MESH_SAMPLE_POINTS,
            empty_layout,
            is_usable_gravity,
            segment_point_cloud,
        )

        empty_pcd = o3d.geometry.PointCloud()
        if gravity is not None and not is_usable_gravity(gravity):
            return empty_layout(gravity), empty_pcd

        if isinstance(mesh, o3d.geometry.PointCloud):
            pcd = mesh
        elif hasattr(mesh, "triangles") and len(mesh.triangles) == 0 and len(mesh.vertices) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            if mesh.has_vertex_colors():
                pcd.colors = mesh.vertex_colors
            if mesh.has_vertex_normals():
                pcd.normals = mesh.vertex_normals
        else:
            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                return empty_layout(gravity), empty_pcd

            try:
                o3d.utility.random.seed(42)
                np.random.seed(42)
                mesh = _decimate_if_needed(mesh)
                pcd = mesh.sample_points_uniformly(number_of_points=MESH_SAMPLE_POINTS)
            except Exception:
                return empty_layout(gravity), empty_pcd

        if len(pcd.points) == 0:
            return empty_layout(gravity), empty_pcd

        return segment_point_cloud(
            pcd,
            gravity,
            trajectory=trajectory,
            vggt_prior=vggt_prior,
            session_dir=session_dir,
        )


    def export_svg(self, layout: dict, output_path: str) -> None:
        from app.services.floorplan import export_svg as export_floorplan_svg

        export_floorplan_svg(layout, output_path)

    def export_dxf(self, layout: dict, output_path: str) -> None:
        from app.services.floorplan import export_dxf as export_floorplan_dxf

        export_floorplan_dxf(layout, output_path)

    def export_whiteflat_ply(self, pcd: o3d.geometry.PointCloud, session_dir: str) -> None:
        if len(pcd.points) == 0:
            return
        os.makedirs(session_dir, exist_ok=True)
        painted = o3d.geometry.PointCloud(pcd)
        painted.normals = o3d.utility.Vector3dVector()
        n = len(painted.points)
        rgb = np.tile(np.array(WHITEFLAT_RGB, dtype=float) / 255.0, (n, 1))
        painted.colors = o3d.utility.Vector3dVector(rgb)
        tmp = os.path.join(session_dir, "reconstructed_whiteflat.tmp.ply")
        final = os.path.join(session_dir, "reconstructed_whiteflat.ply")
        try:
            ok = o3d.io.write_point_cloud(tmp, painted, write_ascii=False)
            if not ok or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
                return
            os.replace(tmp, final)
        finally:
            if os.path.isfile(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def export_visual_artifacts(
        self,
        mesh: o3d.geometry.TriangleMesh,
        layout: dict,
        gravity: np.ndarray | None,
        session_dir: str,
        number_of_points: int | None = None,
        vggt_prior=None,
    ) -> None:
        from app.services.visual_cloud import (
            build_visual_clouds,
            write_visual_ply,
            get_metric_vggt_cloud,
            build_poisson_mesh_from_cloud,
            layout_xy_polygon,
        )

        try:
            xy_poly = layout_xy_polygon(layout) if isinstance(layout, dict) else np.zeros((0, 2))
            skip_vggt = (
                isinstance(layout, dict)
                and isinstance(layout.get("diagnostics"), dict)
                and isinstance(layout["diagnostics"].get("vggt_prior"), dict)
                and layout["diagnostics"]["vggt_prior"].get("skip_reason") == "disabled_by_config"
            )
            if skip_vggt:
                for stale_name in ("reconstructed_vggt_dense.ply", "reconstructed_poisson.glb"):
                    stale_path = os.path.join(session_dir, stale_name)
                    if os.path.isfile(stale_path):
                        try:
                            os.unlink(stale_path)
                        except OSError:
                            pass
            if vggt_prior is not None and hasattr(vggt_prior, "prior"):
                vggt_prior = vggt_prior.prior
            if vggt_prior is None and not skip_vggt:
                try:
                    from app.services.vggt_prior import load_vggt_prior
                    res = load_vggt_prior(session_dir)
                    vggt_prior = res.prior
                except Exception:
                    vggt_prior = None

            if vggt_prior is not None and hasattr(vggt_prior, "points") and len(vggt_prior.points) >= 50:
                if isinstance(mesh, o3d.geometry.PointCloud):
                    ref_pts = np.asarray(mesh.points) if len(mesh.points) else None
                else:
                    ref_pts = np.asarray(mesh.vertices) if len(mesh.vertices) else None
                pcd_room, pcd_vio = get_metric_vggt_cloud(
                    vggt_prior, layout, gravity, reference_points=ref_pts
                )
                if len(pcd_room.points) > 0:
                    visual, dollhouse = build_visual_clouds(
                        mesh, layout, gravity, number_of_points=number_of_points, vggt_prior=vggt_prior
                    )
                    canonical_cloud = visual if len(visual.points) >= 100 else pcd_room
                    # 1. Export dense point clouds (keep TSDF reconstructed.ply intact)
                    write_visual_ply(canonical_cloud, session_dir, "reconstructed_vggt_dense.ply")
                    write_visual_ply(visual, session_dir, "reconstructed_visual.ply")
                    write_visual_ply(dollhouse, session_dir, "reconstructed_dollhouse.ply")
                    write_visual_ply(canonical_cloud, session_dir, "reconstructed_canonical.ply")

                    # 2. Build Poisson surface reconstruction mesh (saved as reconstructed_poisson.glb, keep TSDF reconstructed.glb intact)
                    poisson_mesh = build_poisson_mesh_from_cloud(canonical_cloud, layout=layout, depth=8)
                    if len(poisson_mesh.vertices) > 0 and len(poisson_mesh.triangles) > 0:
                        glb_poisson = os.path.join(session_dir, "reconstructed_poisson.glb")
                        write_triangle_mesh_glb(poisson_mesh, glb_poisson)
                        logger.info(
                            "Exported Poisson mesh (%d triangles) to reconstructed_poisson.glb for session %s",
                            len(poisson_mesh.triangles), session_dir,
                        )
                        glb_final = os.path.join(session_dir, "reconstructed.glb")
                        if not os.path.isfile(glb_final):
                            write_triangle_mesh_glb(poisson_mesh, glb_final)
                    ply_final = os.path.join(session_dir, "reconstructed.ply")
                    if not os.path.isfile(ply_final):
                        write_visual_ply(
                            canonical_cloud,
                            session_dir,
                            "reconstructed.ply",
                        )
                    return

            visual, dollhouse = build_visual_clouds(
                mesh, layout, gravity, number_of_points=number_of_points
            )
            write_visual_ply(visual, session_dir, "reconstructed_visual.ply")
            write_visual_ply(dollhouse, session_dir, "reconstructed_dollhouse.ply")
            write_visual_ply(visual, session_dir, "reconstructed_canonical.ply")
            ply_final = os.path.join(session_dir, "reconstructed.ply")
            if not os.path.isfile(ply_final):
                write_visual_ply(visual, session_dir, "reconstructed.ply")
        except Exception:
            logger.exception("export_visual_artifacts failed")

    export_canonical_visual_cloud = export_visual_artifacts

