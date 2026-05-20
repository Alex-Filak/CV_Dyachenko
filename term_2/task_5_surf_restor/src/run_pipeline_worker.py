from __future__ import annotations

import argparse

from pipeline import run_pipeline, save_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-cloud worker for the safe pipeline runner")
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
    return parser


def main():
    args = build_parser().parse_args()
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


if __name__ == "__main__":
    main()
