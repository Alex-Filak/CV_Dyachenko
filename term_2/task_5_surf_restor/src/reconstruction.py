from __future__ import annotations

from typing import Any

from common import ReconstructionChoice, ReconstructionResult, SegmentData


def _require_numpy():
    import numpy as np

    return np


def _require_open3d():
    import open3d as o3d

    return o3d


def _to_o3d_point_cloud(points: Any):
    o3d = _require_open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def _estimate_normals(pcd: Any, k_neighbors: int):
    o3d = _require_open3d()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max(4, int(k_neighbors)))
    )
    pcd.orient_normals_consistent_tangent_plane(k=max(4, int(k_neighbors)))
    return pcd


def _largest_components(mesh: Any, keep_ratio: float):
    o3d = _require_open3d()
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    if len(cluster_n_triangles) == 0:
        return mesh

    np = _require_numpy()
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    triangle_clusters = np.asarray(triangle_clusters)
    threshold = cluster_n_triangles.max() * keep_ratio
    keep_clusters = np.flatnonzero(cluster_n_triangles >= threshold)
    if len(keep_clusters) == 0:
        keep_clusters = np.asarray([int(cluster_n_triangles.argmax())])
    remove_mask = np.isin(triangle_clusters, keep_clusters, invert=True)
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def _run_poisson(segment: SegmentData, parameters: dict[str, Any]):
    o3d = _require_open3d()
    pcd = _to_o3d_point_cloud(segment.points)
    pcd = _estimate_normals(pcd, parameters["estimate_normals_k"])
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=int(parameters.get("poisson_depth", 8)),
        scale=float(parameters.get("poisson_scale", 1.1)),
    )
    np = _require_numpy()
    densities = np.asarray(densities)
    if len(densities):
        threshold = densities.mean() - 1.5 * densities.std()
        vertices_to_remove = densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
    return _largest_components(mesh, float(parameters.get("component_keep_ratio", 0.8)))


def _run_alpha_shapes(segment: SegmentData, parameters: dict[str, Any]):
    o3d = _require_open3d()
    pcd = _to_o3d_point_cloud(segment.points)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, float(parameters["alpha"])
    )
    return _largest_components(mesh, float(parameters.get("component_keep_ratio", 0.8)))


def _run_ball_pivoting(segment: SegmentData, parameters: dict[str, Any]):
    o3d = _require_open3d()
    pcd = _to_o3d_point_cloud(segment.points)
    pcd = _estimate_normals(pcd, parameters["estimate_normals_k"])
    radii = o3d.utility.DoubleVector([float(v) for v in parameters["ball_radii"]])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    return _largest_components(mesh, float(parameters.get("component_keep_ratio", 0.8)))


def _mesh_is_valid(mesh: Any) -> bool:
    return mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0


def reconstruct_segment(
    segment: SegmentData,
    choice: ReconstructionChoice,
) -> ReconstructionResult:
    runners = {
        "poisson": _run_poisson,
        "alpha_shapes": _run_alpha_shapes,
        "ball_pivoting": _run_ball_pivoting,
    }

    errors: list[str] = []
    for method_name in [choice.primary_method, *choice.fallback_methods]:
        try:
            mesh = runners[method_name](segment, choice.parameters)
            if _mesh_is_valid(mesh):
                return ReconstructionResult(
                    cloud_id=segment.cloud_id,
                    segment_id=segment.segment_id,
                    method_used=method_name,
                    mesh=mesh,
                    success=True,
                    metrics={
                        "vertex_count": len(mesh.vertices),
                        "triangle_count": len(mesh.triangles),
                    },
                )
            errors.append(f"{method_name}: produced an empty or invalid mesh")
        except Exception as exc:
            errors.append(f"{method_name}: {exc}")

    return ReconstructionResult(
        cloud_id=segment.cloud_id,
        segment_id=segment.segment_id,
        method_used=None,
        mesh=None,
        success=False,
        errors=errors,
    )
