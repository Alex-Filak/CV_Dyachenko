from __future__ import annotations

from common import ReconstructionResult


def _require_open3d():
    import open3d as o3d

    return o3d


def assemble_mesh(results: list[ReconstructionResult]):
    o3d = _require_open3d()
    merged = o3d.geometry.TriangleMesh()
    success_count = 0
    for result in results:
        if result.success and result.mesh is not None:
            merged += result.mesh
            success_count += 1
    if success_count == 0:
        return None
    merged.remove_duplicated_vertices()
    merged.remove_duplicated_triangles()
    merged.remove_degenerate_triangles()
    merged.remove_unreferenced_vertices()
    merged.compute_vertex_normals()
    return merged
