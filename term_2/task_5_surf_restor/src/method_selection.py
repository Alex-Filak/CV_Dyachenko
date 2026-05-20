from __future__ import annotations

from common import ClassificationResult, ReconstructionChoice, SegmentFeatures


def _alpha_multiplier(features: SegmentFeatures) -> float:
    if features.point_count < 900 or features.normal_consistency < 0.82:
        return 2.3
    if features.roughness_mean > 0.012 or features.curvature_mean > 0.02:
        return 2.8
    return 3.3


def _ball_radii(features: SegmentFeatures) -> list[float]:
    base = max(features.mean_knn_distance, 1e-4)
    if features.point_count > 700 and features.normal_consistency > 0.8:
        multipliers = (2.8, 3.8, 4.8)
    else:
        multipliers = (2.3, 3.3, 4.3)
    return [base * multiplier for multiplier in multipliers]


def choose_reconstruction_method(
    features: SegmentFeatures,
    classification: ClassificationResult,
) -> ReconstructionChoice:
    mean_dist = max(features.mean_knn_distance, 1e-4)
    rationale = list(classification.reasons)
    parameters = {
        "estimate_normals_k": min(max(16, features.point_count // 200), 48),
        "mean_knn_distance": mean_dist,
    }

    if classification.class_name == "planar":
        primary = "alpha_shapes"
        fallback = ["ball_pivoting", "poisson"]
        alpha_multiplier = _alpha_multiplier(features)
        parameters.update(
            {
                "alpha": mean_dist * alpha_multiplier,
                "component_keep_ratio": 0.9 if alpha_multiplier < 3.0 else 0.95,
            }
        )
        rationale.append("Thin open surface is handled by boundary-aware alpha shapes.")
    elif classification.class_name == "tubular":
        primary = "ball_pivoting"
        fallback = ["poisson", "alpha_shapes"]
        parameters.update(
            {
                "ball_radii": _ball_radii(features),
                "component_keep_ratio": 0.9,
            }
        )
        rationale.append("Tubular geometry benefits from rolling-ball triangulation.")
    elif classification.class_name == "spherical":
        if features.normal_consistency > 0.88:
            primary = "ball_pivoting"
            fallback = ["poisson", "alpha_shapes"]
            parameters.update(
                {
                    "ball_radii": [mean_dist * 1.4, mean_dist * 2.2],
                    "component_keep_ratio": 0.9,
                }
            )
            rationale.append("Smooth closed shell with stable normals suits ball pivoting.")
        else:
            primary = "poisson"
            fallback = ["ball_pivoting", "alpha_shapes"]
            parameters.update(
                {
                    "poisson_depth": 8,
                    "poisson_scale": 1.1,
                    "component_keep_ratio": 0.9,
                }
            )
            rationale.append("Sphere-like segment but normals are not fully stable, using Poisson.")
    else:
        if features.planarity > 0.45 and features.roughness_mean > 0.012:
            primary = "poisson"
            fallback = ["alpha_shapes", "ball_pivoting"]
            parameters.update(
                {
                    "estimate_normals_k": 16,
                    "poisson_depth": 7,
                    "poisson_scale": 1.02,
                    "component_keep_ratio": 0.92,
                }
            )
            rationale.append("Planar surface with local relief is reconstructed with a conservative Poisson setup.")
        else:
            primary = "poisson"
            fallback = ["alpha_shapes", "ball_pivoting"]
            depth = 7 if features.point_count < 2000 else 8 if features.point_count < 8000 else 9
            parameters.update(
                {
                    "poisson_depth": depth,
                    "poisson_scale": 1.1,
                    "component_keep_ratio": 0.9,
                }
            )
            rationale.append("Complex or uncertain geometry is routed to the most tolerant reconstructor.")

    return ReconstructionChoice(
        cloud_id=features.cloud_id,
        segment_id=features.segment_id,
        primary_method=primary,
        fallback_methods=fallback,
        parameters=parameters,
        rationale=rationale,
    )
