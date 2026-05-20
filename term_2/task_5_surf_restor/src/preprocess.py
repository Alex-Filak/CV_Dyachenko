from __future__ import annotations

from typing import Any

from common import CloudData


def _require_numpy():
    import numpy as np

    return np


def _try_ckdtree():
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    return cKDTree


def validate_cloud(cloud: CloudData) -> list[str]:
    np = _require_numpy()
    issues: list[str] = []
    if cloud.points.ndim != 2 or cloud.points.shape[1] != 3:
        issues.append("Points array must have shape (N, 3).")
    if cloud.labels.ndim != 1:
        issues.append("Labels array must have shape (N,).")
    if len(cloud.points) != len(cloud.labels):
        issues.append("Points and labels must have equal length.")
    if len(cloud.points) == 0:
        issues.append("Cloud is empty.")
    if not np.isfinite(cloud.points).all():
        issues.append("Cloud contains NaN or Inf coordinates.")
    return issues


def remove_invalid_rows(cloud: CloudData) -> CloudData:
    np = _require_numpy()
    mask = np.isfinite(cloud.points).all(axis=1)
    points = cloud.points[mask]
    labels = cloud.labels[mask]
    removed = int((~mask).sum())
    metadata = dict(cloud.metadata)
    metadata["removed_invalid_rows"] = removed
    return CloudData(
        file_path=cloud.file_path,
        points=points,
        labels=labels,
        scale=cloud.scale,
        center=cloud.center,
        metadata=metadata,
    )


def statistical_outlier_filter(
    points: Any,
    nb_neighbors: int = 16,
    std_ratio: float = 2.0,
):
    np = _require_numpy()
    if len(points) <= nb_neighbors + 1:
        return points, np.ones(len(points), dtype=bool)

    cKDTree = _try_ckdtree()
    if cKDTree is None:
        return points, np.ones(len(points), dtype=bool)

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=nb_neighbors + 1)
    mean_distances = distances[:, 1:].mean(axis=1)
    threshold = mean_distances.mean() + std_ratio * mean_distances.std()
    mask = mean_distances <= threshold
    return points[mask], mask


def normalize_cloud(cloud: CloudData) -> CloudData:
    np = _require_numpy()
    center = cloud.points.mean(axis=0)
    shifted = cloud.points - center
    norms = np.linalg.norm(shifted, axis=1)
    scale = float(norms.max()) if len(norms) else 1.0
    if scale <= 0:
        scale = 1.0
    normalized = shifted / scale
    return CloudData(
        file_path=cloud.file_path,
        points=normalized,
        labels=cloud.labels.copy(),
        scale=scale,
        center=center,
        metadata=dict(cloud.metadata),
    )


def voxel_downsample(points: Any, labels: Any, voxel_size: float):
    np = _require_numpy()
    if voxel_size <= 0 or len(points) == 0:
        return points, labels

    keys = np.floor(points / voxel_size).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(keys):
        buckets.setdefault(tuple(int(v) for v in key), []).append(index)

    sampled_points = []
    sampled_labels = []
    for indices in buckets.values():
        bucket_points = points[indices]
        bucket_labels = labels[indices]
        sampled_points.append(bucket_points.mean(axis=0))
        values, counts = np.unique(bucket_labels, return_counts=True)
        sampled_labels.append(values[counts.argmax()])

    return np.asarray(sampled_points, dtype=float), np.asarray(sampled_labels, dtype=labels.dtype)


def preprocess_cloud(
    cloud: CloudData,
    remove_noise: bool = True,
    normalize: bool = True,
    downsample_voxel: float | None = None,
) -> CloudData:
    working = remove_invalid_rows(cloud)

    if remove_noise:
        filtered_points, mask = statistical_outlier_filter(working.points)
        working = CloudData(
            file_path=working.file_path,
            points=filtered_points,
            labels=working.labels[mask],
            scale=working.scale,
            center=working.center,
            metadata={**working.metadata, "noise_removed": int((~mask).sum())},
        )

    if downsample_voxel is not None:
        points, labels = voxel_downsample(working.points, working.labels, downsample_voxel)
        working = CloudData(
            file_path=working.file_path,
            points=points,
            labels=labels,
            scale=working.scale,
            center=working.center,
            metadata={**working.metadata, "downsample_voxel": downsample_voxel},
        )

    if normalize:
        working = normalize_cloud(working)

    return working
