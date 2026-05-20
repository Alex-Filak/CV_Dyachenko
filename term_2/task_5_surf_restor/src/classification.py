from __future__ import annotations

from common import ClassificationResult, SegmentFeatures


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def classify_segment(features: SegmentFeatures) -> ClassificationResult:
    reasons: list[str] = []

    planar_score = 0.0
    if features.planarity > 0.45:
        planar_score += 0.35
        reasons.append("High planarity from PCA spectrum.")
    if features.sphericity < 0.12:
        planar_score += 0.15
        reasons.append("Low thickness relative to dominant axes.")
    if features.roughness_mean < 0.02:
        planar_score += 0.2
        reasons.append("Low local roughness.")
    if features.normal_consistency > 0.9:
        planar_score += 0.2
        reasons.append("Normals are locally consistent.")
    if features.roughness_mean > 0.02 or features.curvature_std > 0.025:
        planar_score -= 0.12

    tubular_score = 0.0
    if features.linearity > 0.55:
        tubular_score += 0.3
        reasons.append("Strong dominant axis suggests elongated structure.")
    if features.cylinder_fit_error is not None and features.cylinder_fit_error < 0.03:
        tubular_score += 0.3
        reasons.append("Low cylinder proxy fitting error.")
    if 0.05 < features.sphericity < 0.35:
        tubular_score += 0.15
        reasons.append("Cross-section thickness is compatible with tubular geometry.")
    if features.connectivity_components <= 2:
        tubular_score += 0.1
        reasons.append("Segment is mostly connected.")

    spherical_score = 0.0
    if features.sphere_fit_error is not None and features.sphere_fit_error < 0.03:
        spherical_score += 0.4
        reasons.append("Low sphere fitting error.")
    if features.radial_normal_alignment is not None and features.radial_normal_alignment > 0.8:
        spherical_score += 0.25
        reasons.append("Normals align with radial directions.")
    if features.sphericity > 0.3:
        spherical_score += 0.15
        reasons.append("PCA spectrum indicates volumetric compactness.")
    if features.normal_consistency > 0.8:
        spherical_score += 0.1
        reasons.append("Normals are stable enough for closed smooth geometry.")

    complex_score = 0.15
    if features.connectivity_components > 2:
        complex_score += 0.25
        reasons.append("Multiple connected components suggest complex topology.")
    if features.planarity > 0.45 and features.roughness_mean > 0.012:
        complex_score += 0.22
        reasons.append("Mostly planar segment contains local relief and is safer in the complex branch.")
    if features.roughness_mean > 0.03:
        complex_score += 0.2
        reasons.append("Roughness is high.")
    if features.curvature_std > 0.03:
        complex_score += 0.2
        reasons.append("Curvature varies significantly across the segment.")
    if features.normal_consistency < 0.75:
        complex_score += 0.2
        reasons.append("Normals are not sufficiently coherent.")

    scores = {
        "planar": _clamp(planar_score),
        "tubular": _clamp(tubular_score),
        "spherical": _clamp(spherical_score),
        "complex": _clamp(complex_score),
    }

    class_name = max(scores, key=scores.get)
    confidence = scores[class_name]

    if confidence < 0.45 and class_name != "complex":
        class_name = "complex"
        confidence = max(confidence, scores["complex"], 0.45)
        reasons.append("Confidence is low, routing to conservative complex class.")

    return ClassificationResult(
        cloud_id=features.cloud_id,
        segment_id=features.segment_id,
        class_name=class_name,
        confidence=float(confidence),
        reasons=reasons,
    )
