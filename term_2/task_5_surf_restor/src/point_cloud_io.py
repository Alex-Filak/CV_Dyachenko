from __future__ import annotations

from pathlib import Path

from common import CloudData


def _require_numpy():
    import numpy as np

    return np


def parse_ascii_ply(file_path: str | Path) -> CloudData:
    np = _require_numpy()
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as handle:
        header = []
        vertex_count = None
        properties: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading header: {path}")
            line = line.strip()
            header.append(line)
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                properties.append(line.split()[-1])
            elif line == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"PLY file does not define vertex count: {path}")

        required = {"x", "y", "z"}
        if not required.issubset(set(properties)):
            raise ValueError(f"PLY file is missing coordinate properties: {path}")

        label_candidates = {"label", "Label", "scalar_Label", "segment", "segment_id"}
        label_name = None
        for candidate in label_candidates:
            if candidate in properties:
                label_name = candidate
                break
        if label_name is None:
            raise ValueError(f"PLY file is missing segment label property: {path}")

        x_idx = properties.index("x")
        y_idx = properties.index("y")
        z_idx = properties.index("z")
        label_idx = properties.index(label_name)

        rows = []
        for row_index in range(vertex_count):
            raw = handle.readline()
            if not raw:
                raise ValueError(f"Unexpected EOF in vertex section: {path}")
            parts = raw.strip().split()
            if len(parts) != len(properties):
                raise ValueError(
                    f"Invalid vertex row {row_index} in {path}: expected {len(properties)} values"
                )
            rows.append(
                (
                    float(parts[x_idx]),
                    float(parts[y_idx]),
                    float(parts[z_idx]),
                    int(float(parts[label_idx])),
                )
            )

    data = np.asarray(rows, dtype=float)
    points = data[:, :3]
    labels = data[:, 3].astype(np.int64)
    return CloudData(
        file_path=path,
        points=points,
        labels=labels,
        metadata={
            "vertex_count": vertex_count,
            "label_property": label_name,
            "header": header,
        },
    )


def load_clouds(data_dir: str | Path) -> list[CloudData]:
    path = Path(data_dir)
    files = sorted(path.glob("*.ply"))
    if not files:
        raise FileNotFoundError(f"No .ply files found in {path}")
    return [parse_ascii_ply(file_path) for file_path in files]
