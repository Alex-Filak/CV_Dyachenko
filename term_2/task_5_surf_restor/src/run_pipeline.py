from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crash-resistant adaptive surface reconstruction pipeline"
    )
    parser.add_argument("--data-dir", default="data", help="Directory with source .ply point clouds")
    parser.add_argument("--output-dir", default="outputs", help="Directory for reconstructed meshes")
    parser.add_argument("--max-clouds", type=int, default=None, help="Limit number of processed clouds")
    parser.add_argument(
        "--min-segment-points",
        type=int,
        default=64,
        help="Segments smaller than this are discarded",
    )
    parser.add_argument(
        "--downsample-voxel",
        type=float,
        default=None,
        help="Optional voxel size for preprocessing downsampling",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based index of first cloud to process",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-cloud timeout in seconds",
    )
    parser.add_argument(
        "--unsafe-direct",
        action="store_true",
        help="Run the legacy single-process pipeline directly instead of crash-resistant per-cloud subprocess mode",
    )
    return parser


def _progress(iterable, total: int | None = None, desc: str | None = None):
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _kill_process_tree(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _run_safe(args):
    repo_root = Path(__file__).resolve().parent
    data_dir = (repo_root / args.data_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.ply"))
    if args.start_index > 1:
        files = files[args.start_index - 1 :]
    if args.max_clouds is not None:
        files = files[: args.max_clouds]

    summary_rows = []
    progress = _progress(files, total=len(files), desc="Clouds")

    for file_path in progress:
        cloud_id = file_path.stem
        cloud_output_dir = output_dir / cloud_id
        cloud_output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix=f"{cloud_id}_", dir=output_dir) as temp_data_dir:
            temp_data_path = Path(temp_data_dir)
            temp_file_path = temp_data_path / file_path.name
            os.symlink(file_path, temp_file_path)

            cmd = [
                "python3",
                str(repo_root / "run_pipeline_worker.py"),
                "--data-dir",
                str(temp_data_path),
                "--output-dir",
                str(cloud_output_dir),
                "--max-clouds",
                "1",
                "--min-segment-points",
                str(args.min_segment_points),
            ]
            if args.downsample_voxel is not None:
                cmd.extend(["--downsample-voxel", str(args.downsample_voxel)])

            proc = subprocess.Popen(
                cmd,
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            timed_out = False
            try:
                stdout, stderr = proc.communicate(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_process_tree(proc)
                stdout = proc.stdout.read() if proc.stdout is not None else ""
                stderr = proc.stderr.read() if proc.stderr is not None else ""
            finally:
                _kill_process_tree(proc)

            returncode = proc.returncode if proc.returncode is not None else -9

        cloud_summary = {
            "cloud_id": cloud_id,
            "returncode": returncode,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "success": returncode == 0,
        }

        summary_file = cloud_output_dir / "summary.json"
        if summary_file.exists():
            try:
                cloud_summary["result"] = json.loads(summary_file.read_text(encoding="utf-8"))
            except Exception as exc:
                cloud_summary["summary_read_error"] = str(exc)

        if timed_out:
            cloud_summary["failure_type"] = "timeout"
        elif returncode != 0:
            cloud_summary["failure_type"] = (
                "segfault" if returncode < 0 or returncode == 139 else "command_error"
            )

        summary_rows.append(cloud_summary)

        success_count = sum(1 for row in summary_rows if row["success"])
        fail_count = len(summary_rows) - success_count
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(ok=success_count, fail=fail_count, refresh=True)

    batch_summary = {
        "clouds_requested": len(files),
        "successful_clouds": sum(1 for row in summary_rows if row["success"]),
        "failed_clouds": sum(1 for row in summary_rows if not row["success"]),
        "clouds": summary_rows,
    }

    with (output_dir / "batch_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(batch_summary, handle, indent=2)

    print(
        {
            "clouds_requested": batch_summary["clouds_requested"],
            "successful_clouds": batch_summary["successful_clouds"],
            "failed_clouds": batch_summary["failed_clouds"],
            "output_dir": str(output_dir),
        }
    )


def _run_unsafe_direct(args):
    from pipeline import run_pipeline, save_results

    results = run_pipeline(
        data_dir=args.data_dir,
        max_clouds=args.max_clouds,
        min_segment_points=args.min_segment_points,
        downsample_voxel=args.downsample_voxel,
        output_dir=args.output_dir,
    )
    save_results(results, args.output_dir)

    total_clouds = len(results)
    total_segments = sum(item.segment_count for item in results)
    successful_segments = sum(item.successful_segments for item in results)
    failed_segments = sum(item.failed_segments for item in results)
    print(
        {
            "clouds": total_clouds,
            "segments": total_segments,
            "successful_segments": successful_segments,
            "failed_segments": failed_segments,
            "output_dir": args.output_dir,
        }
    )


def main():
    args = build_parser().parse_args()
    if args.unsafe_direct:
        _run_unsafe_direct(args)
    else:
        _run_safe(args)


if __name__ == "__main__":
    main()
