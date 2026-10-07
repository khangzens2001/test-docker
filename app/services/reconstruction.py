import copy
import gc
import logging
import os
import numpy as np
import open3d as o3d

from app.services.glb_export import write_triangle_mesh_glb

logger = logging.getLogger(__name__)

VOXEL_LENGTH = 0.04
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


class ReconstructionService:
    def integrate_tsdf(
        self,
        depths: list[np.ndarray],
        poses: list[np.ndarray],
        intrinsics: np.ndarray,
        colors: list[np.ndarray],
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
        def _make_uniform_volume(poses_list):
            all_t = np.array([p[0:3, 3] for p in poses_list], dtype=float)
            all_fwd = np.array([p[0:3, 3] + p[0:3, :3] @ np.array([0.0, 0.0, 2.0]) for p in poses_list], dtype=float)
            all_back = np.array([p[0:3, 3] + p[0:3, :3] @ np.array([0.0, 0.0, -2.0]) for p in poses_list], dtype=float)
            pts_box = np.vstack([all_t, all_fwd, all_back])
            p_min = pts_box.min(axis=0) - 1.5
            p_max = pts_box.max(axis=0) + 1.5
            span = float(np.max(p_max - p_min))
            length = max(span, 6.0)
            origin = p_min - 0.5
            res = int(np.clip(length / 0.08, 64, 256))
            return o3d.pipelines.integration.UniformTSDFVolume(
                length,
                res,
                SDF_TRUNC,
                o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                origin,
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

        if len(depths) <= 2:
            volume = _make_uniform_volume(poses)
            mesh = _run_integrate(volume)
        else:
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=VOXEL_LENGTH,
                sdf_trunc=SDF_TRUNC,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )
            mesh = _run_integrate(volume)
            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                logger.warning("ScalableTSDFVolume returned empty mesh; falling back to UniformTSDFVolume.")
                vol_u = _make_uniform_volume(poses)
                mesh = _run_integrate(vol_u)

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
        )

        try:
            if vggt_prior is None:
                try:
                    from app.services.vggt_prior import load_vggt_prior
                    res = load_vggt_prior(session_dir)
                    vggt_prior = res.prior
                except Exception:
                    vggt_prior = None

            if vggt_prior is not None and hasattr(vggt_prior, "points") and len(vggt_prior.points) >= 50:
                ref_pts = np.asarray(mesh.vertices) if len(mesh.vertices) else None
                pcd_room, pcd_vio = get_metric_vggt_cloud(
                    vggt_prior, layout, gravity, reference_points=ref_pts
                )
                if len(pcd_room.points) > 0:
                    # 1. Export dense point clouds
                    write_visual_ply(pcd_room, session_dir, "reconstructed_vggt_dense.ply")

                    visual, dollhouse = build_visual_clouds(
                        mesh, layout, gravity, number_of_points=number_of_points, vggt_prior=vggt_prior
                    )
                    write_visual_ply(visual, session_dir, "reconstructed_visual.ply")
                    write_visual_ply(dollhouse, session_dir, "reconstructed_dollhouse.ply")

                    # 2. Build Poisson surface reconstruction mesh
                    cloud_for_mesh = visual if (visual is not None and len(visual.points) >= 100) else pcd_room
                    poisson_mesh = build_poisson_mesh_from_cloud(cloud_for_mesh, layout=layout, depth=8)
                    if len(poisson_mesh.vertices) > 0 and len(poisson_mesh.triangles) > 0:
                        glb_tmp = os.path.join(session_dir, "reconstructed.tmp.glb")
                        glb_final = os.path.join(session_dir, "reconstructed.glb")
                        ply_tmp = os.path.join(session_dir, "reconstructed.tmp.ply")
                        ply_final = os.path.join(session_dir, "reconstructed.ply")
                        ok_glb = write_triangle_mesh_glb(poisson_mesh, glb_tmp)
                        if ok_glb and os.path.isfile(glb_tmp) and os.path.getsize(glb_tmp) > 0:
                            os.replace(glb_tmp, glb_final)
                            logger.info(
                                "Replaced reconstructed.glb with Poisson mesh (%d triangles) for session %s",
                                len(poisson_mesh.triangles), session_dir,
                            )
                            # Export dense metric VGGT cloud in VIO coordinates for reconstructed.ply
                            if len(pcd_vio.points) > 0:
                                write_visual_ply(pcd_vio, session_dir, "reconstructed.ply")
                    return

            visual, dollhouse = build_visual_clouds(
                mesh, layout, gravity, number_of_points=number_of_points
            )
            write_visual_ply(visual, session_dir, "reconstructed_visual.ply")
            write_visual_ply(dollhouse, session_dir, "reconstructed_dollhouse.ply")
        except Exception:
            logger.exception("export_visual_artifacts failed")

