from __future__ import annotations

from pathlib import Path
from typing import Any

from assembly import assemble_mesh
from classification import classify_segment
from common import CloudRunResult
from evaluation import evaluate_segment_reconstruction
from geometry_analysis import extract_segment_features
from method_selection import choose_reconstruction_method
from point_cloud_io import load_clouds
from preprocess import preprocess_cloud, validate_cloud
from reconstruction import reconstruct_segment
from segmentation import build_segments


def _require_open3d():
    import open3d as o3d

    return o3d


def _progress(iterable, total: int | None = None, desc: str | None = None):
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _segment_color(segment_id: int) -> tuple[float, float, float]:
    palette = [
        (0.85, 0.20, 0.20),
        (0.15, 0.55, 0.85),
        (0.20, 0.70, 0.35),
        (0.90, 0.65, 0.15),
        (0.60, 0.35, 0.80),
        (0.15, 0.70, 0.70),
        (0.85, 0.40, 0.60),
        (0.55, 0.55, 0.20),
    ]
    return palette[segment_id % len(palette)]


def _restore_mesh_coordinates(mesh: Any, segment) -> Any:
    if mesh is None:
        return None
    scale = float(segment.metadata.get("cloud_scale", 1.0) or 1.0)
    center = segment.metadata.get("cloud_center")
    if scale != 1.0:
        mesh.scale(scale, center=(0.0, 0.0, 0.0))
    if center is not None:
        mesh.translate(center)
    return mesh


def _process_cloud(
    cloud,
    min_segment_points: int,
    downsample_voxel: float | None,
    output_dir: str | Path | None = None,
):
    issues = validate_cloud(cloud)
    if issues:
        return CloudRunResult(
            cloud_id=cloud.file_path.stem,
            segment_count=0,
            successful_segments=0,
            failed_segments=0,
            metrics={"validation_issues": issues},
        )

    processed = preprocess_cloud(cloud, downsample_voxel=downsample_voxel)
    segments, removed_segments = build_segments(processed, min_points=min_segment_points)

    segment_results = []
    class_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    cloud_dir = None
    if output_dir is not None:
        cloud_dir = Path(output_dir) / cloud.file_path.stem
        cloud_dir.mkdir(parents=True, exist_ok=True)

    for segment in segments:
        features = extract_segment_features(segment)
        classification = classify_segment(features)
        class_counts[classification.class_name] = class_counts.get(classification.class_name, 0) + 1

        choice = choose_reconstruction_method(features, classification)
        reconstruction = reconstruct_segment(segment, choice)
        reconstruction.metrics.update(evaluate_segment_reconstruction(segment, reconstruction))
        if reconstruction.method_used is not None:
            method_counts[reconstruction.method_used] = method_counts.get(reconstruction.method_used, 0) + 1
        if reconstruction.success and reconstruction.mesh is not None:
            reconstruction.mesh = _restore_mesh_coordinates(reconstruction.mesh, segment)
            reconstruction.mesh.paint_uniform_color(_segment_color(segment.segment_id))
        if cloud_dir is not None and reconstruction.success and reconstruction.mesh is not None:
            _write_segment_mesh(cloud_dir, reconstruction)
        segment_results.append(reconstruction)

    assembled_mesh = None
    assembly_error = None
    if segment_results:
        try:
            assembled_mesh = assemble_mesh(segment_results)
        except Exception as exc:
            assembly_error = str(exc)

    successful_segments = sum(1 for item in segment_results if item.success)
    failed_segments = len(segment_results) - successful_segments
    metrics: dict[str, Any] = {
        "removed_segments": removed_segments,
        "class_counts": class_counts,
        "method_counts": method_counts,
    }
    if assembly_error is not None:
        metrics["assembly_error"] = assembly_error

    cloud_result = CloudRunResult(
        cloud_id=cloud.file_path.stem,
        segment_count=len(segment_results),
        successful_segments=successful_segments,
        failed_segments=failed_segments,
        assembled_mesh=assembled_mesh,
        metrics=metrics,
        segment_results=segment_results,
    )
    if output_dir is not None:
        _write_assembled_mesh(cloud_result, output_dir)
        for item in cloud_result.segment_results:
            item.mesh = None
        cloud_result.assembled_mesh = None
    return cloud_result


def run_pipeline(
    data_dir: str | Path,
    max_clouds: int | None = None,
    min_segment_points: int = 64,
    downsample_voxel: float | None = None,
    output_dir: str | Path | None = None,
) -> list[CloudRunResult]:
    clouds = load_clouds(data_dir)
    if max_clouds is not None:
        clouds = clouds[:max_clouds]

    results: list[CloudRunResult] = []
    clouds_progress = _progress(clouds, total=len(clouds), desc="Clouds")
    for cloud in clouds_progress:
        result = _process_cloud(
            cloud,
            min_segment_points=min_segment_points,
            downsample_voxel=downsample_voxel,
            output_dir=output_dir,
        )
        results.append(result)
        if hasattr(clouds_progress, "set_postfix"):
            clouds_progress.set_postfix(
                segments=result.segment_count,
                ok=result.successful_segments,
                fail=result.failed_segments,
                refresh=True,
            )

    results.sort(key=lambda item: item.cloud_id)
    return results


def save_results(
    results: list[CloudRunResult],
    output_dir: str | Path,
):
    import json

    o3d = _require_open3d()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = []
    for cloud_result in results:
        cloud_dir = output_path / cloud_result.cloud_id
        cloud_dir.mkdir(exist_ok=True)

        for segment_result in cloud_result.segment_results:
            if segment_result.success and segment_result.mesh is not None:
                mesh_path = cloud_dir / f"segment_{segment_result.segment_id}.ply"
                o3d.io.write_triangle_mesh(str(mesh_path), segment_result.mesh)

        if cloud_result.assembled_mesh is not None:
            o3d.io.write_triangle_mesh(
                str(cloud_dir / f"{cloud_result.cloud_id}_assembled.ply"),
                cloud_result.assembled_mesh,
            )

        summary.append(
            {
                "cloud_id": cloud_result.cloud_id,
                "segment_count": cloud_result.segment_count,
                "successful_segments": cloud_result.successful_segments,
                "failed_segments": cloud_result.failed_segments,
                "metrics": cloud_result.metrics,
                "segments": [
                    {
                        "segment_id": item.segment_id,
                        "success": item.success,
                        "method_used": item.method_used,
                        "metrics": item.metrics,
                        "errors": item.errors,
                    }
                    for item in cloud_result.segment_results
                ],
            }
        )

    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _write_cloud_result(cloud_result: CloudRunResult, output_dir: str | Path):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cloud_dir = output_path / cloud_result.cloud_id
    cloud_dir.mkdir(exist_ok=True)

    for segment_result in cloud_result.segment_results:
        if segment_result.success and segment_result.mesh is not None:
            _write_segment_mesh(cloud_dir, segment_result)

    _write_assembled_mesh(cloud_result, output_dir)


def _write_segment_mesh(cloud_dir: str | Path, segment_result: Any):
    o3d = _require_open3d()
    cloud_path = Path(cloud_dir)
    cloud_path.mkdir(parents=True, exist_ok=True)
    mesh_path = cloud_path / f"segment_{segment_result.segment_id}.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), segment_result.mesh)


def _write_assembled_mesh(cloud_result: CloudRunResult, output_dir: str | Path):
    o3d = _require_open3d()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cloud_dir = output_path / cloud_result.cloud_id
    cloud_dir.mkdir(exist_ok=True)
    if cloud_result.assembled_mesh is not None:
        o3d.io.write_triangle_mesh(
            str(cloud_dir / f"{cloud_result.cloud_id}_assembled.ply"),
            cloud_result.assembled_mesh,
        )


def write_summary(
    results: list[CloudRunResult],
    output_dir: str | Path,
):
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = [
        {
            "cloud_id": cloud_result.cloud_id,
            "segment_count": cloud_result.segment_count,
            "successful_segments": cloud_result.successful_segments,
            "failed_segments": cloud_result.failed_segments,
            "metrics": cloud_result.metrics,
            "segments": [
                {
                    "segment_id": item.segment_id,
                    "success": item.success,
                    "method_used": item.method_used,
                    "metrics": item.metrics,
                    "errors": item.errors,
                }
                for item in cloud_result.segment_results
            ],
        }
        for cloud_result in results
    ]

    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
