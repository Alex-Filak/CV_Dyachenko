from __future__ import annotations

import math
from typing import Any

from common import SegmentData, SegmentFeatures


def _require_numpy():
    import numpy as np

    return np


def _try_ckdtree():
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    return cKDTree


def _build_neighbor_search(points: Any):
    cKDTree = _try_ckdtree()
    if cKDTree is not None:
        return cKDTree(points)
    return None


def _knn(points: Any, k: int):
    np = _require_numpy()
    tree = _build_neighbor_search(points)
    if tree is not None:
        distances, indices = tree.query(points, k=min(k + 1, len(points)))
        return distances[:, 1:], indices[:, 1:]

    all_distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    indices = np.argsort(all_distances, axis=1)[:, 1 : k + 1]
    distances = np.take_along_axis(all_distances, indices, axis=1)
    return distances, indices


def _local_pca(points: Any, neighbor_indices: Any):
    np = _require_numpy()
    eigenvalues = []
    normals = []
    roughness = []
    curvatures = []
    for row, indices in enumerate(neighbor_indices):
        neighborhood = points[indices]
        centroid = neighborhood.mean(axis=0)
        centered = neighborhood - centroid
        covariance = centered.T @ centered / max(len(neighborhood) - 1, 1)
        vals, vecs = np.linalg.eigh(covariance)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        normal = vecs[:, -1]
        eigenvalues.append(vals)
        normals.append(normal)
        roughness.append(float(abs((points[row] - centroid).dot(normal))))
        denom = max(vals.sum(), 1e-12)
        curvatures.append(float(vals[-1] / denom))
    return np.asarray(eigenvalues), np.asarray(normals), np.asarray(roughness), np.asarray(curvatures)


def _connectivity_components(points: Any, radius: float) -> int:
    np = _require_numpy()
    if len(points) == 0:
        return 0

    tree = _build_neighbor_search(points)
    adjacency: list[list[int]] = []
    if tree is not None:
        for point in points:
            neighbors = tree.query_ball_point(point, r=radius)
            adjacency.append([idx for idx in neighbors if idx is not None])
    else:
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        for row in distances:
            adjacency.append(np.flatnonzero(row <= radius).tolist())

    visited = set()
    components = 0
    for node in range(len(points)):
        if node in visited:
            continue
        components += 1
        stack = [node]
        visited.add(node)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
    return components


def _fit_sphere(points: Any) -> tuple[float, float]:
    np = _require_numpy()
    if len(points) < 4:
        return float("inf"), 0.0
    a = np.column_stack((2.0 * points, np.ones(len(points))))
    b = (points**2).sum(axis=1)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    center = solution[:3]
    radius_sq = float(solution[3] + center.dot(center))
    if radius_sq <= 0:
        return float("inf"), 0.0
    radius = math.sqrt(radius_sq)
    radial_distances = np.linalg.norm(points - center, axis=1)
    error = float(np.mean(np.abs(radial_distances - radius)))
    return error, radius


def _fit_cylinder_proxy(points: Any) -> float:
    np = _require_numpy()
    if len(points) < 8:
        return float("inf")
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projections = centered @ axis
    radial_vectors = centered - np.outer(projections, axis)
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    radius = radial_distances.mean()
    return float(np.mean(np.abs(radial_distances - radius)))


def extract_segment_features(segment: SegmentData, k_neighbors: int = 16) -> SegmentFeatures:
    np = _require_numpy()
    points = segment.points
    point_count = len(points)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extents = bbox_max - bbox_min
    bbox_volume = float(np.prod(np.maximum(extents, 1e-9)))
    centroid = points.mean(axis=0)

    distances, indices = _knn(points, min(k_neighbors, max(point_count - 1, 1)))
    mean_knn_distance = float(distances.mean()) if distances.size else 0.0
    density = float(point_count / max(bbox_volume, 1e-9))

    eigenvalues, normals, roughness, curvatures = _local_pca(points, indices)
    lambda1 = float(eigenvalues[:, 0].mean()) if len(eigenvalues) else 0.0
    lambda2 = float(eigenvalues[:, 1].mean()) if len(eigenvalues) else 0.0
    lambda3 = float(eigenvalues[:, 2].mean()) if len(eigenvalues) else 0.0
    denom = max(lambda1, 1e-12)
    linearity = (lambda1 - lambda2) / denom
    planarity = (lambda2 - lambda3) / denom
    sphericity = lambda3 / denom
    anisotropy = (lambda1 - lambda3) / denom

    if len(normals) > 1:
        normal_dot = np.clip(np.abs(normals @ normals.T), 0.0, 1.0)
        upper = normal_dot[np.triu_indices(len(normals), k=1)]
        normal_consistency = float(upper.mean()) if len(upper) else 1.0
    else:
        normal_consistency = 1.0

    connectivity_radius = max(mean_knn_distance * 2.5, 1e-4)
    components = _connectivity_components(points, connectivity_radius)

    sphere_error, _ = _fit_sphere(points)
    cylinder_error = _fit_cylinder_proxy(points)

    centered = points - centroid
    radial_norms = np.linalg.norm(centered, axis=1)
    valid = radial_norms > 1e-9
    radial_alignment = None
    if valid.any():
        radial_dirs = centered[valid] / radial_norms[valid][:, None]
        radial_alignment = float(np.mean(np.abs((normals[valid] * radial_dirs).sum(axis=1))))

    return SegmentFeatures(
        cloud_id=segment.cloud_id,
        segment_id=segment.segment_id,
        point_count=point_count,
        bbox_extents=(float(extents[0]), float(extents[1]), float(extents[2])),
        bbox_volume=bbox_volume,
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        mean_knn_distance=mean_knn_distance,
        density=density,
        linearity=float(linearity),
        planarity=float(planarity),
        sphericity=float(sphericity),
        anisotropy=float(anisotropy),
        roughness_mean=float(roughness.mean()) if len(roughness) else 0.0,
        roughness_std=float(roughness.std()) if len(roughness) else 0.0,
        curvature_mean=float(curvatures.mean()) if len(curvatures) else 0.0,
        curvature_std=float(curvatures.std()) if len(curvatures) else 0.0,
        normal_consistency=normal_consistency,
        connectivity_components=int(components),
        cylinder_fit_error=float(cylinder_error),
        sphere_fit_error=float(sphere_error),
        radial_normal_alignment=radial_alignment,
    )
