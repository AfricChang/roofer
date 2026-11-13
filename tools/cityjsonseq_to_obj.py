#!/usr/bin/env python3
"""Convert CityJSONSeq building geometries to OBJ with semantic materials.

The converter reads CityJSON Sequence metadata and feature records, applies
the CityJSON transform, and exports selected geometry LoDs as OBJ polygon
faces. Ground, wall, roof, and ceiling surfaces are assigned separate
materials in a companion MTL file.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_LODS = ("1.3", "2.2")
MATERIAL_COLORS = {
    "GroundSurface": ("0.35", "0.65", "0.35"),
    "WallSurface": ("0.72", "0.72", "0.75"),
    "RoofSurface": ("0.85", "0.28", "0.18"),
    "OuterCeilingSurface": ("0.95", "0.68", "0.18"),
    "Other": ("0.60", "0.60", "0.60"),
}


def read_cityjsonseq(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}: invalid JSON on line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {line_number} is not a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: CityJSONSeq is empty")
    return records


def sequence_transform(records: Sequence[dict[str, Any]]) -> tuple[list[float], list[float]]:
    for record in records:
        transform = record.get("transform")
        if isinstance(transform, dict):
            scale = transform.get("scale", [1.0, 1.0, 1.0])
            translate = transform.get("translate", [0.0, 0.0, 0.0])
            if len(scale) == 3 and len(translate) == 3:
                return [float(value) for value in scale], [float(value) for value in translate]
    return [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]


def transformed_vertices(
    feature: dict[str, Any],
    scale: Sequence[float],
    translate: Sequence[float],
) -> list[tuple[float, float, float]]:
    vertices = feature.get("vertices", [])
    if not isinstance(vertices, list):
        raise ValueError("CityJSONFeature.vertices must be an array")
    result = []
    for index, vertex in enumerate(vertices):
        if not isinstance(vertex, list) or len(vertex) < 3:
            raise ValueError(f"CityJSONFeature vertex {index} is malformed")
        result.append(
            tuple(
                translate[axis] + float(vertex[axis]) * scale[axis]
                for axis in range(3)
            )
        )
    return result


def semantic_material(surface_type: str) -> str:
    if surface_type in MATERIAL_COLORS:
        return surface_type
    if surface_type in {"ClosureSurface", "InteriorWallSurface"}:
        return "WallSurface"
    return "Other"


def surface_type(
    geometry: dict[str, Any],
    shell_index: int,
    surface_index: int,
) -> str:
    semantics = geometry.get("semantics", {})
    if not isinstance(semantics, dict):
        return "Other"
    semantic_surfaces = semantics.get("surfaces", [])
    values = semantics.get("values", [])
    if not isinstance(semantic_surfaces, list) or not isinstance(values, list):
        return "Other"

    if geometry.get("type") == "Solid":
        shell_values = values[shell_index] if shell_index < len(values) else []
        semantic_index = (
            shell_values[surface_index]
            if isinstance(shell_values, list) and surface_index < len(shell_values)
            else None
        )
    else:
        semantic_index = values[surface_index] if surface_index < len(values) else None

    if not isinstance(semantic_index, int):
        return "Other"
    if semantic_index < 0 or semantic_index >= len(semantic_surfaces):
        return "Other"
    semantic = semantic_surfaces[semantic_index]
    if not isinstance(semantic, dict):
        return "Other"
    return semantic_material(str(semantic.get("type", "Other")))


def iter_surface_rings(
    geometry: dict[str, Any],
) -> Iterable[tuple[list[int], str]]:
    geometry_type = geometry.get("type")
    boundaries = geometry.get("boundaries", [])
    if not isinstance(boundaries, list):
        raise ValueError("geometry.boundaries must be an array")

    if geometry_type == "Solid":
        for shell_index, shell in enumerate(boundaries):
            if not isinstance(shell, list):
                continue
            for surface_index, surface in enumerate(shell):
                if not isinstance(surface, list):
                    continue
                material = surface_type(geometry, shell_index, surface_index)
                for ring in surface:
                    if isinstance(ring, list):
                        yield ring, material
        return

    if geometry_type == "MultiSurface":
        for surface_index, surface in enumerate(boundaries):
            if not isinstance(surface, list):
                continue
            material = surface_type(geometry, 0, surface_index)
            for ring in surface:
                if isinstance(ring, list):
                    yield ring, material
        return

    raise ValueError(f"unsupported CityJSON geometry type: {geometry_type!r}")


def geometry_records(
    records: Sequence[dict[str, Any]],
    lod: str,
    scale: Sequence[float],
    translate: Sequence[float],
) -> tuple[list[tuple[float, float, float]], list[tuple[str, list[int], str]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[str, list[int], str]] = []

    for record in records:
        if record.get("type") != "CityJSONFeature":
            continue
        feature_vertices = transformed_vertices(record, scale, translate)
        vertex_offset = len(vertices)
        vertices.extend(feature_vertices)
        for object_id, city_object in record.get("CityObjects", {}).items():
            if not isinstance(city_object, dict):
                continue
            for geometry in city_object.get("geometry", []):
                if not isinstance(geometry, dict) or str(geometry.get("lod")) != lod:
                    continue
                for ring, material in iter_surface_rings(geometry):
                    if len(ring) < 3:
                        continue
                    absolute_ring = [vertex_offset + int(index) + 1 for index in ring]
                    faces.append((str(object_id), absolute_ring, material))
    return vertices, faces


def write_mtl(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Materials generated from CityJSON semantics\n")
        for name, (red, green, blue) in MATERIAL_COLORS.items():
            stream.write(f"newmtl {name}\n")
            stream.write("Ka 0.1 0.1 0.1\n")
            stream.write(f"Kd {red} {green} {blue}\n")
            stream.write("Ks 0.0 0.0 0.0\n")
            stream.write("d 1.0\n\n")


def write_obj(
    path: Path,
    mtl_name: str,
    lod: str,
    vertices: Sequence[tuple[float, float, float]],
    faces: Sequence[tuple[str, list[int], str]],
) -> Counter[str]:
    material_counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"# CityJSONSeq LoD {lod} exported to OBJ\n")
        stream.write(f"mtllib {mtl_name}\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        current_object = None
        current_material = None
        for object_id, ring, material in faces:
            if object_id != current_object:
                stream.write(f"\no {object_id}_lod{lod.replace('.', '')}\n")
                current_object = object_id
                current_material = None
            if material != current_material:
                stream.write(f"usemtl {material}\n")
                current_material = material
            stream.write("f " + " ".join(str(index) for index in ring) + "\n")
            material_counts[material] += 1
    return material_counts


def default_name(path: Path) -> str:
    name = path.name
    if name.endswith(".city.jsonl"):
        return name[: -len(".city.jsonl")]
    if name.endswith(".jsonl"):
        return name[: -len(".jsonl")]
    return path.stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert CityJSONSeq LoD geometries to OBJ and semantic MTL."
    )
    parser.add_argument("cityjsonseq", type=Path, help="Input CityJSON Sequence (.city.jsonl).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for OBJ, MTL, and export summary files.",
    )
    parser.add_argument(
        "--name",
        help="Output name prefix; defaults to the input filename without .city.jsonl.",
    )
    parser.add_argument(
        "--lod",
        action="append",
        dest="lods",
        choices=("1.2", "1.3", "2.2"),
        help="LoD to export. Repeat this option; defaults to 1.3 and 2.2.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        records = read_cityjsonseq(args.cityjsonseq)
        scale, translate = sequence_transform(records)
        lods = tuple(args.lods or DEFAULT_LODS)
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        name = args.name or default_name(args.cityjsonseq)
        mtl_path = output_dir / f"{name}_semantic.mtl"
        write_mtl(mtl_path)

        summary: dict[str, object] = {}
        for lod in lods:
            vertices, faces = geometry_records(records, lod, scale, translate)
            obj_path = output_dir / f"{name}_lod{lod.replace('.', '')}.obj"
            material_counts = write_obj(
                obj_path,
                mtl_name=mtl_path.name,
                lod=lod,
                vertices=vertices,
                faces=faces,
            )
            summary[lod] = {
                "obj": str(obj_path),
                "vertices": len(vertices),
                "faces": len(faces),
                "materials": dict(sorted(material_counts.items())),
                "bytes": obj_path.stat().st_size,
            }

        summary_path = output_dir / f"{name}_obj_export_summary.json"
        with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
