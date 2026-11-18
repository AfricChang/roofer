#!/usr/bin/env python3
"""Convert CityJSONSeq building geometries to OBJ with semantic materials.

The converter reads CityJSON Sequence metadata and feature records, applies
the CityJSON transform, and exports selected geometry LoDs as OBJ faces.
Only surfaces with holes are triangulated; other polygons retain their source
vertex order. Every OBJ face records its original CityJSON surface identity.
Ground, wall, roof, and ceiling surfaces use separate semantic materials.

Hole-free exports need only the standard library. Triangulating a surface with
holes requires Shapely >= 2.1 with GEOS >= 3.10.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import quote


DEFAULT_LODS = ("2.2",)
OBJ_COORDINATE_DECIMALS = 6
MATERIAL_COLORS = {
    "GroundSurface": ("0.35", "0.65", "0.35"),
    "WallSurface": ("0.72", "0.72", "0.75"),
    "RoofSurface": ("0.85", "0.28", "0.18"),
    "OuterCeilingSurface": ("0.95", "0.68", "0.18"),
    "Other": ("0.60", "0.60", "0.60"),
}


@dataclass(frozen=True)
class SourceFace:
    """原始面身份；各层索引均从零开始，记录号包含元数据记录。"""

    record_index: int
    feature_id: str | None
    object_id: str
    geometry_index: int
    geometry_type: str
    lod: str
    shell_index: int
    surface_index: int
    ring_count: int

    @property
    def source_face_id(self) -> str:
        # 对象名转义后，跨记录、几何及壳层的原始面身份不会冲突。
        return (
            f"r{self.record_index}/o{quote(self.object_id, safe='')}"
            f"/g{self.geometry_index}/s{self.shell_index}/f{self.surface_index}"
        )


@dataclass(frozen=True)
class ExportFace:
    """OBJ 面及其原始顶点索引；三角化不重新按坐标搜索整个顶点表。"""

    source: SourceFace
    ring: list[int]
    source_vertex_indices: list[int]
    material: str
    triangulated: bool
    triangle_index: int | None

    @property
    def object_id(self) -> str:
        return self.source.object_id

    @property
    def source_face_id(self) -> str:
        return self.source.source_face_id


def face_provenance(face: ExportFace, index: int) -> dict[str, Any]:
    return {
        **asdict(face.source),
        "source_face_id": face.source_face_id,
        "semantic_feature_id": (
            f"{face.object_id}:{face.source.lod}:{face.source.surface_index}"
        ),
        "export_face_index": index,
        "obj_face_number": index + 1,
        "source_vertex_indices": face.source_vertex_indices,
        "obj_vertex_indices": face.ring,
        "material": face.material,
        "triangulated": face.triangulated,
        "triangle_index": face.triangle_index,
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
        coordinates = tuple(
            translate[axis] + float(vertex[axis]) * scale[axis]
            for axis in range(3)
        )
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"CityJSONFeature vertex {index} is not finite")
        result.append(coordinates)
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


def iter_surface_faces(
    geometry: dict[str, Any],
) -> Iterable[tuple[int, int, list[list[int]], str]]:
    """按原始 CityJSON 面返回完整环集合，不拆开外环和洞环。"""

    geometry_type = geometry.get("type")
    boundaries = geometry.get("boundaries", [])
    if not isinstance(boundaries, list):
        raise ValueError("geometry.boundaries must be an array")

    if geometry_type == "Solid":
        for shell_index, shell in enumerate(boundaries):
            if not isinstance(shell, list):
                raise ValueError(f"CityJSON shell {shell_index} is not an array")
            for surface_index, surface in enumerate(shell):
                if not isinstance(surface, list):
                    raise ValueError(f"CityJSON surface {surface_index} is not an array")
                yield shell_index, surface_index, surface, surface_type(
                    geometry, shell_index, surface_index)
        return

    if geometry_type == "MultiSurface":
        for surface_index, surface in enumerate(boundaries):
            if not isinstance(surface, list):
                raise ValueError(f"CityJSON surface {surface_index} is not an array")
            yield 0, surface_index, surface, surface_type(
                geometry, 0, surface_index)
        return

    raise ValueError(f"unsupported CityJSON geometry type: {geometry_type!r}")


def triangulate_surface(
    rings: Sequence[Sequence[int]],
    vertices: Sequence[tuple[float, float, float]],
) -> list[list[int]]:
    """主法向投影后的约束三角化；返回原始索引，不填洞、不补新点。"""

    if len(rings) <= 1:
        return [list(rings[0])]

    try:
        from shapely import constrained_delaunay_triangles
        from shapely.errors import GEOSException, UnsupportedGEOSVersionError
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from shapely.strtree import STRtree
    except ImportError as error:
        raise ValueError(
            "surfaces with holes require Shapely >= 2.1 and GEOS >= 3.10"
        ) from error

    # 只移除同一索引的显式闭合点，不合并几何重合但身份不同的顶点。
    open_rings = [
        list(ring[:-1] if ring[0] == ring[-1] else ring)
        for ring in rings
    ]
    origin = vertices[open_rings[0][0]]
    outer = [
        tuple(vertices[index][axis] - origin[axis] for axis in range(3))
        for index in open_rings[0]
    ]
    normal = [
        math.fsum(
            point[(axis + 1) % 3] * outer[(index + 1) % len(outer)][(axis + 2) % 3]
            - point[(axis + 2) % 3] * outer[(index + 1) % len(outer)][(axis + 1) % 3]
            for index, point in enumerate(outer)
        )
        for axis in range(3)
    ]
    drop_axis = max(range(3), key=lambda axis: abs(normal[axis]))
    if not all(math.isfinite(value) for value in normal) or normal[drop_axis] == 0:
        raise ValueError("CityJSON surface with holes has a degenerate outer ring")
    axes = ((drop_axis + 1) % 3, (drop_axis + 2) % 3)
    projected_rings = []
    coordinate_indices: dict[tuple[float, float], int] = {}
    for ring in open_rings:
        projected = []
        for index in ring:
            point = tuple(vertices[index][axis] - origin[axis] for axis in axes)
            if point in coordinate_indices and coordinate_indices[point] != index:
                raise ValueError("distinct source vertices share a surface projection")
            coordinate_indices[point] = index
            projected.append(point)
        projected_rings.append(projected)

    polygon = Polygon(projected_rings[0], projected_rings[1:])
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise ValueError("cannot triangulate invalid CityJSON surface with holes")

    try:
        constrained = constrained_delaunay_triangles(polygon)
    except (GEOSException, UnsupportedGEOSVersionError) as error:
        raise ValueError(f"constrained hole triangulation failed: {error}") from error
    triangle_geometries = list(constrained.geoms)
    triangles: list[list[int]] = []
    for triangle in triangle_geometries:
        if (
            triangle.geom_type != "Polygon" or triangle.is_empty
            or len(triangle.exterior.coords) != 4 or triangle.interiors
            or triangle.area <= 0 or not polygon.covers(triangle)
        ):
            raise ValueError("constrained triangulation returned an invalid or outside triangle")
        points = list(triangle.exterior.coords)[:-1]
        try:
            ring = [coordinate_indices[(float(x), float(y))] for x, y in points]
        except KeyError as error:
            raise ValueError("triangulation introduced a non-source vertex") from error
        a, b, c = points
        signed_twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if (signed_twice_area > 0) != (normal[drop_axis] > 0):
            ring.reverse()
        start = ring.index(min(ring))
        ring = ring[start:] + ring[:start]
        triangles.append(ring)

    # 拓扑校验不设面积/距离容差，不以面积相等掩盖重叠或缺口。
    if not unary_union(triangle_geometries).equals(polygon):
        raise ValueError("triangulation did not cover the original surface exactly")
    tree = STRtree(triangle_geometries)
    for index, triangle in enumerate(triangle_geometries):
        for other in tree.query(triangle):
            if other > index and triangle.relate_pattern(triangle_geometries[other], "T********"):
                raise ValueError("triangulation contains overlapping triangle interiors")
    boundary_edges = Counter(
        tuple(sorted((ring[index], ring[(index + 1) % len(ring)])))
        for ring in open_rings for index in range(len(ring))
    )
    triangle_edges = Counter(
        tuple(sorted((ring[index], ring[(index + 1) % 3])))
        for ring in triangles for index in range(3)
    )
    if (
        any(count not in (1, 2) for count in triangle_edges.values())
        or Counter({edge: count for edge, count in triangle_edges.items() if count == 1})
        != boundary_edges
    ):
        raise ValueError("triangulation did not preserve the original boundary edges")
    return sorted(triangles)


def geometry_records(
    records: Sequence[dict[str, Any]],
    lod: str,
    scale: Sequence[float],
    translate: Sequence[float],
) -> tuple[list[tuple[float, float, float]], list[ExportFace]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[ExportFace] = []

    for record_index, record in enumerate(records):
        if record.get("type") != "CityJSONFeature":
            continue
        feature_vertices = transformed_vertices(record, scale, translate)
        vertex_offset = len(vertices)
        vertices.extend(feature_vertices)
        for object_id, city_object in record.get("CityObjects", {}).items():
            if not isinstance(city_object, dict):
                continue
            for geometry_index, geometry in enumerate(city_object.get("geometry", [])):
                if not isinstance(geometry, dict) or str(geometry.get("lod")) != lod:
                    continue
                for shell_index, surface_index, surface, material in iter_surface_faces(geometry):
                    source = SourceFace(
                        record_index=record_index,
                        feature_id=str(record["id"]) if record.get("id") is not None else None,
                        object_id=str(object_id),
                        geometry_index=geometry_index,
                        geometry_type=str(geometry.get("type")),
                        lod=lod,
                        shell_index=shell_index,
                        surface_index=surface_index,
                        ring_count=len(surface),
                    )
                    try:
                        if not surface:
                            raise ValueError("surface has no outer ring")
                        for ring in surface:
                            if not isinstance(ring, list) or len(ring) < 3:
                                raise ValueError("surface ring has fewer than 3 vertices")
                            if any(
                                type(index) is not int or not 0 <= index < len(feature_vertices)
                                for index in ring
                            ):
                                raise ValueError("surface references an invalid vertex index")
                        triangulated = len(surface) > 1
                        output_rings = (
                            triangulate_surface(surface, feature_vertices)
                            if triangulated else [list(surface[0])]
                        )
                        for triangle_index, ring in enumerate(output_rings):
                            faces.append(ExportFace(
                                source=source,
                                ring=[vertex_offset + index + 1 for index in ring],
                                source_vertex_indices=ring,
                                material=material,
                                triangulated=triangulated,
                                triangle_index=triangle_index if triangulated else None,
                            ))
                    except ValueError as error:
                        raise ValueError(f"{source.source_face_id}: {error}") from error
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
    faces: Sequence[ExportFace],
) -> Counter[str]:
    material_counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"# CityJSONSeq LoD {lod} exported to OBJ\n")
        stream.write(f"mtllib {mtl_name}\n")
        for x, y, z in vertices:
            stream.write(
                f"v {x:.{OBJ_COORDINATE_DECIMALS}f}"
                f" {y:.{OBJ_COORDINATE_DECIMALS}f} {z:.{OBJ_COORDINATE_DECIMALS}f}\n"
            )

        current_object = None
        current_material = None
        for index, face in enumerate(faces):
            if face.object_id != current_object:
                stream.write(f"\no {face.object_id}_lod{lod.replace('.', '')}\n")
                current_object = face.object_id
                current_material = None
            if face.material != current_material:
                stream.write(f"usemtl {face.material}\n")
                current_material = face.material
            stream.write("# source_face " + json.dumps(
                face_provenance(face, index), ensure_ascii=True, separators=(",", ":")
            ) + "\n")
            stream.write("f " + " ".join(str(index) for index in face.ring) + "\n")
            material_counts[face.material] += 1
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
        choices=("2.2",),
        help="LoD to export. The integration exports LoD2.2 only.",
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
                "source_cityjsonseq": str(args.cityjsonseq),
                "source_sha256": hashlib.sha256(args.cityjsonseq.read_bytes()).hexdigest().upper(),
                "provenance_schema": "cityjson-obj-face-provenance-v1",
                "source_indices_zero_based": True,
                "obj_vertex_indices_one_based": True,
                "obj_coordinate_decimals": OBJ_COORDINATE_DECIMALS,
                "triangulation_policy": "only_surfaces_with_holes",
                "triangulation_method": "constrained_delaunay_dominant_axis",
                "vertices": len(vertices),
                "faces": len(faces),
                "materials": dict(sorted(material_counts.items())),
                "original_face_count": len({
                    face.source_face_id for face in faces
                }),
                "triangulated_face_count": len({
                    face.source_face_id for face in faces if face.triangulated
                }),
                "triangulated_triangle_count": sum(
                    1 for face in faces if face.triangulated
                ),
                "face_provenance": [face_provenance(face, index) for index, face in enumerate(faces)],
                "bytes": obj_path.stat().st_size,
            }

        summary_path = output_dir / f"{name}_obj_export_summary.json"
        with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        # 完整逐面映射只写伴随文件，避免旁路运行日志随三角形数膨胀。
        printed_summary = {
            lod: {key: value for key, value in item.items() if key != "face_provenance"}
            for lod, item in summary.items()
        }
        print(json.dumps(
            {"summary": str(summary_path), "lods": printed_summary},
            ensure_ascii=False, indent=2,
        ))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
