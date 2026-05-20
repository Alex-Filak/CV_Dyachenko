from __future__ import annotations

from common import CloudData, SegmentData


def _require_numpy():
    import numpy as np

    return np


def _try_ckdtree():
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    return cKDTree


def _component_labels(points, k_neighbors: int = 8):
    np = _require_numpy()
    if len(points) == 0:
        return np.asarray([], dtype=np.int64), 0
    if len(points) == 1:
        return np.asarray([0], dtype=np.int64), 1

    cKDTree = _try_ckdtree()
    if cKDTree is not None:
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=min(k_neighbors + 1, len(points)))
        mean_knn = float(distances[:, 1:].mean()) if distances.size else 0.0
        radius = max(mean_knn * 2.5, 1e-4)
        adjacency = [tree.query_ball_point(point, r=radius) for point in points]
    else:
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        sorted_distances = np.sort(distances, axis=1)
        mean_knn = float(sorted_distances[:, 1 : min(k_neighbors + 1, len(points))].mean())
        radius = max(mean_knn * 2.5, 1e-4)
        adjacency = [np.flatnonzero(row <= radius).tolist() for row in distances]

    labels = np.full(len(points), -1, dtype=np.int64)
    component_id = 0
    for start in range(len(points)):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = component_id
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if labels[neighbor] == -1:
                    labels[neighbor] = component_id
                    stack.append(neighbor)
        component_id += 1

    return labels, component_id


def build_segments(
    cloud: CloudData,
    min_points: int = 64,
    min_component_ratio: float = 0.15,
) -> tuple[list[SegmentData], list[int]]:
    np = _require_numpy()
    segments: list[SegmentData] = []
    removed: list[int] = []
    cloud_id = cloud.file_path.stem

    for label in np.unique(cloud.labels):
        indices = np.flatnonzero(cloud.labels == label)
        if len(indices) < min_points:
            removed.append(int(label))
            continue
        points = cloud.points[indices]

        component_labels, component_count = _component_labels(points)
        if component_count > 1 and len(points) < max(min_points * 2, 96):
            removed.append(int(label))
            continue

        component_sizes = [
            int((component_labels == component_id).sum()) for component_id in range(component_count)
        ]
        largest_component = max(component_sizes) if component_sizes else len(points)

        kept_components = 0
        for component_id in range(component_count):
            local_mask = component_labels == component_id
            component_indices = indices[local_mask]
            component_points = cloud.points[component_indices]
            component_size = len(component_indices)

            if component_size < min_points:
                continue
            if component_count > 1 and component_size < largest_component * min_component_ratio:
                continue

            kept_components += 1
            segment_id = int(label) if component_count == 1 else int(label) * 100 + component_id
            segments.append(
                SegmentData(
                    cloud_id=cloud_id,
                    segment_id=segment_id,
                    points=component_points,
                    original_indices=component_indices,
                    metadata={
                        "point_count": int(component_size),
                        "cloud_center": cloud.center,
                        "cloud_scale": cloud.scale,
                        "original_label": int(label),
                        "component_index": int(component_id),
                        "component_count": int(component_count),
                    },
                )
            )

        if kept_components == 0:
            removed.append(int(label))

    return segments, removed
