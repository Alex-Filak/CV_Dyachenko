from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CloudData:
    file_path: Path
    points: Any
    labels: Any
    scale: float = 1.0
    center: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentData:
    cloud_id: str
    segment_id: int
    points: Any
    original_indices: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentFeatures:
    cloud_id: str
    segment_id: int
    point_count: int
    bbox_extents: tuple[float, float, float]
    bbox_volume: float
    centroid: tuple[float, float, float]
    mean_knn_distance: float
    density: float
    linearity: float
    planarity: float
    sphericity: float
    anisotropy: float
    roughness_mean: float
    roughness_std: float
    curvature_mean: float
    curvature_std: float
    normal_consistency: float
    connectivity_components: int
    cylinder_fit_error: float | None = None
    sphere_fit_error: float | None = None
    radial_normal_alignment: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    cloud_id: str
    segment_id: int
    class_name: str
    confidence: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class ReconstructionChoice:
    cloud_id: str
    segment_id: int
    primary_method: str
    fallback_methods: list[str]
    parameters: dict[str, Any]
    rationale: list[str] = field(default_factory=list)


@dataclass
class ReconstructionResult:
    cloud_id: str
    segment_id: int
    method_used: str | None
    mesh: Any
    success: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class CloudRunResult:
    cloud_id: str
    segment_count: int
    successful_segments: int
    failed_segments: int
    assembled_mesh: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    segment_results: list[ReconstructionResult] = field(default_factory=list)
