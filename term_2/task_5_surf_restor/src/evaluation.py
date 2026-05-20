from __future__ import annotations

from typing import Any

from common import ReconstructionResult, SegmentData


def _require_numpy():
    import numpy as np

    return np


def _try_ckdtree():
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    return cKDTree


def _sample_mesh_points(mesh: Any, count: int):
    samples = mesh.sample_points_uniformly(number_of_points=count)
    np = _require_numpy()
    return np.asarray(samples.points)


def _nn_distances(source: Any, target: Any):
    np = _require_numpy()
    cKDTree = _try_ckdtree()
    if cKDTree is not None:
        tree = cKDTree(target)
        distances, _ = tree.query(source, k=1)
        return distances
    return np.linalg.norm(source[:, None, :] - target[None, :, :], axis=2).min(axis=1)


def evaluate_segment_reconstruction(
    segment: SegmentData,
    reconstruction: ReconstructionResult,
    sample_count: int = 2000,
) -> dict[str, float]:
    np = _require_numpy()
    if not reconstruction.success or reconstruction.mesh is None:
        return {"success": 0.0}

    count = min(sample_count, max(len(segment.points), 128))
    mesh_points = _sample_mesh_points(reconstruction.mesh, count)
    source_to_mesh = _nn_distances(segment.points, mesh_points)
    mesh_to_source = _nn_distances(mesh_points, segment.points)
    chamfer = float(source_to_mesh.mean() + mesh_to_source.mean())

    metrics = {
        "success": 1.0,
        "mean_point_to_mesh": float(source_to_mesh.mean()),
        "max_point_to_mesh": float(source_to_mesh.max()),
        "mean_mesh_to_point": float(mesh_to_source.mean()),
        "chamfer_like": chamfer,
        "coverage_at_2x_knn": float(
            (source_to_mesh <= max(2.0 * source_to_mesh.mean(), 1e-6)).mean()
        ),
        "vertex_count": float(len(reconstruction.mesh.vertices)),
        "triangle_count": float(len(reconstruction.mesh.triangles)),
    }

    triangle_clusters, cluster_n_triangles, _ = reconstruction.mesh.cluster_connected_triangles()
    if len(cluster_n_triangles):
        metrics["mesh_components"] = float(len(cluster_n_triangles))
        metrics["largest_component_triangles"] = float(max(cluster_n_triangles))
    else:
        metrics["mesh_components"] = 0.0
        metrics["largest_component_triangles"] = 0.0

    return metrics
