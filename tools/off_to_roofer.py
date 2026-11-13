#!/usr/bin/env python3
"""Convert a triangulated OFF mesh into Roofer-compatible LAS and GeoJSON.

This utility is intended for local preparation of a mesh-derived point cloud:

* vertices outside the supplied polygon become LAS ground points (class 2);
* roof samples are generated on an XY grid from non-vertical mesh triangles;
* each grid cell keeps the highest sampled roof elevation (class 6);
* small gaps can be filled from neighbouring roof cells;
* the supplied OFF polygon is written as a 2D GeoJSON building footprint.

The implementation uses only the Python standard library. Coordinates are
assumed to be projected metres, matching Roofer's input requirements.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


EPSILON = 1.0e-9


@dataclass(frozen=True)
class Vertex:
    x: float
    y: float
    z: float


@dataclass
class OffMesh:
    vertices: list[Vertex]
    faces: list[list[int]]


def _next_data_line(lines: Iterable[str], path: Path) -> str:
    for raw_line in lines:
        line = raw_line.strip()
        if line and not line.startswith("#"):
            return line
    raise ValueError(f"{path}: unexpected end of OFF file")


def read_off(path: Path) -> OffMesh:
    """Read an ASCII OFF mesh without loading the whole file as tokens."""

    with path.open("r", encoding="utf-8") as stream:
        header = _next_data_line(stream, path)
        if header.split()[0].upper() != "OFF":
            raise ValueError(f"{path}: expected an OFF header, got {header!r}")

        counts = _next_data_line(stream, path).split()
        if len(counts) < 2:
            raise ValueError(f"{path}: OFF counts line must contain vertices and faces")
        vertex_count = int(counts[0])
        face_count = int(counts[1])

        vertices: list[Vertex] = []
        for index in range(vertex_count):
            values = _next_data_line(stream, path).split()
            if len(values) < 3:
                raise ValueError(f"{path}: vertex {index} has fewer than 3 coordinates")
            vertices.append(Vertex(float(values[0]), float(values[1]), float(values[2])))

        faces: list[list[int]] = []
        for index in range(face_count):
            values = _next_data_line(stream, path).split()
            if not values:
                continue
            ring_size = int(values[0])
            if ring_size < 3 or len(values) < ring_size + 1:
                raise ValueError(f"{path}: face {index} is malformed")
            faces.append([int(value) for value in values[1 : ring_size + 1]])

    return OffMesh(vertices=vertices, faces=faces)


def select_polygon(mesh: OffMesh, path: Path) -> list[tuple[float, float]]:
    """Select the largest face as the footprint ring from a polygon OFF file."""

    if not mesh.faces:
        if len(mesh.vertices) < 3:
            raise ValueError(f"{path}: polygon OFF has fewer than 3 vertices")
        ring = list(range(len(mesh.vertices)))
    else:
        ring = max(mesh.faces, key=len)

    if len(ring) < 3:
        raise ValueError(f"{path}: footprint ring has fewer than 3 vertices")

    try:
        polygon = [(mesh.vertices[index].x, mesh.vertices[index].y) for index in ring]
    except IndexError as error:
        raise ValueError(f"{path}: footprint references an invalid vertex") from error

    return polygon


def polygon_bounds(polygon: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _point_on_segment(
    x: float,
    y: float,
    a: tuple[float, float],
    b: tuple[float, float],
) -> bool:
    cross = (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])
    if abs(cross) > 1.0e-8:
        return False
    return (
        min(a[0], b[0]) - 1.0e-8 <= x <= max(a[0], b[0]) + 1.0e-8
        and min(a[1], b[1]) - 1.0e-8 <= y <= max(a[1], b[1]) + 1.0e-8
    )


def point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Return True for points inside or on the boundary of a simple ring."""

    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(x, y, previous, current):
            return True

        x1, y1 = previous
        x2, y2 = current
        crosses_scanline = (y1 > y) != (y2 > y)
        if crosses_scanline:
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def _triangle_normal(
    a: Vertex,
    b: Vertex,
    c: Vertex,
) -> tuple[float, float, float]:
    abx, aby, abz = b.x - a.x, b.y - a.y, b.z - a.z
    acx, acy, acz = c.x - a.x, c.y - a.y, c.z - a.z
    return (
        aby * acz - abz * acy,
        abz * acx - abx * acz,
        abx * acy - aby * acx,
    )


def _barycentric_contains(
    x: float,
    y: float,
    a: Vertex,
    b: Vertex,
    c: Vertex,
) -> tuple[bool, float, float, float]:
    denominator = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y)
    if abs(denominator) <= EPSILON:
        return False, 0.0, 0.0, 0.0

    wa = ((b.y - c.y) * (x - c.x) + (c.x - b.x) * (y - c.y)) / denominator
    wb = ((c.y - a.y) * (x - c.x) + (a.x - c.x) * (y - c.y)) / denominator
    wc = 1.0 - wa - wb
    inside = wa >= -EPSILON and wb >= -EPSILON and wc >= -EPSILON
    return inside, wa, wb, wc


def _cell_index(value: float, origin: float, cell_size: float) -> int:
    return math.floor((value - origin) / cell_size)


def _cell_center(index: int, origin: float, cell_size: float) -> float:
    return origin + (index + 0.5) * cell_size


def rasterize_roof(
    mesh: OffMesh,
    polygon: Sequence[tuple[float, float]],
    grid_size: float,
    normal_z_threshold: float,
    fill_passes: int,
    fill_max_range: float,
) -> tuple[dict[tuple[int, int], float], int]:
    """Sample roof triangles and fill small gaps on the same regular grid."""

    polygon_min_x, polygon_min_y, polygon_max_x, polygon_max_y = polygon_bounds(polygon)
    grid: dict[tuple[int, int], float] = {}

    min_i = _cell_index(polygon_min_x, polygon_min_x, grid_size) - 1
    max_i = _cell_index(polygon_max_x, polygon_min_x, grid_size) + 1
    min_j = _cell_index(polygon_min_y, polygon_min_y, grid_size) - 1
    max_j = _cell_index(polygon_max_y, polygon_min_y, grid_size) + 1

    for face in mesh.faces:
        if len(face) < 3:
            continue
        anchor = mesh.vertices[face[0]]
        for offset in range(1, len(face) - 1):
            triangle = (anchor, mesh.vertices[face[offset]], mesh.vertices[face[offset + 1]])
            a, b, c = triangle
            nx, ny, nz = _triangle_normal(a, b, c)
            normal_length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if normal_length <= EPSILON:
                continue
            if abs(nz) / normal_length < normal_z_threshold:
                continue

            triangle_min_x = min(a.x, b.x, c.x)
            triangle_max_x = max(a.x, b.x, c.x)
            triangle_min_y = min(a.y, b.y, c.y)
            triangle_max_y = max(a.y, b.y, c.y)
            start_i = max(min_i, _cell_index(triangle_min_x, polygon_min_x, grid_size) - 1)
            end_i = min(max_i, _cell_index(triangle_max_x, polygon_min_x, grid_size) + 1)
            start_j = max(min_j, _cell_index(triangle_min_y, polygon_min_y, grid_size) - 1)
            end_j = min(max_j, _cell_index(triangle_max_y, polygon_min_y, grid_size) + 1)

            for i in range(start_i, end_i + 1):
                x = _cell_center(i, polygon_min_x, grid_size)
                for j in range(start_j, end_j + 1):
                    y = _cell_center(j, polygon_min_y, grid_size)
                    if not point_in_polygon(x, y, polygon):
                        continue
                    inside, wa, wb, wc = _barycentric_contains(x, y, a, b, c)
                    if not inside:
                        continue
                    z = wa * a.z + wb * b.z + wc * c.z
                    key = (i, j)
                    if z > grid.get(key, -math.inf):
                        grid[key] = z

    filled_points = 0
    if fill_passes > 0 and grid:
        polygon_cell_range = (
            range(min_i, max_i + 1),
            range(min_j, max_j + 1),
        )
        for _ in range(fill_passes):
            additions: dict[tuple[int, int], float] = {}
            for i in polygon_cell_range[0]:
                x = _cell_center(i, polygon_min_x, grid_size)
                for j in polygon_cell_range[1]:
                    key = (i, j)
                    if key in grid:
                        continue
                    y = _cell_center(j, polygon_min_y, grid_size)
                    if not point_in_polygon(x, y, polygon):
                        continue
                    neighbours = [
                        grid[neighbour_key]
                        for neighbour_key in (
                            (i - 1, j),
                            (i + 1, j),
                            (i, j - 1),
                            (i, j + 1),
                        )
                        if neighbour_key in grid
                    ]
                    if len(neighbours) < 3:
                        continue
                    if max(neighbours) - min(neighbours) > fill_max_range:
                        continue
                    additions[key] = sum(neighbours) / len(neighbours)

            if not additions:
                break
            grid.update(additions)
            filled_points += len(additions)

    return grid, filled_points


def _format_grid_suffix(grid_size: float) -> str:
    scaled = round(grid_size * 100)
    return f"{scaled:03d}"


def write_geojson(
    path: Path,
    polygon: Sequence[tuple[float, float]],
    building_id: str,
) -> None:
    ring = [[x, y] for x, y in polygon]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"building_id": building_id},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_las(
    path: Path,
    roof_points: Sequence[Vertex],
    ground_points: Sequence[Vertex],
    roof_class: int,
    ground_class: int,
) -> None:
    points = [(point, roof_class) for point in roof_points]
    points.extend((point, ground_class) for point in ground_points)
    if not points:
        raise ValueError("cannot write an empty LAS file")

    scale = (0.001, 0.001, 0.001)
    minimum = tuple(min(getattr(point, axis) for point, _ in points) for axis in ("x", "y", "z"))
    offsets = tuple(math.floor(value / step) * step for value, step in zip(minimum, scale))

    integer_points: list[tuple[int, int, int, int]] = []
    for point, classification in points:
        coordinates = tuple(
            int(round((value - offset) / step))
            for value, offset, step in zip((point.x, point.y, point.z), offsets, scale)
        )
        if any(abs(value) > 2_147_483_647 for value in coordinates):
            raise ValueError("point coordinate does not fit LAS 1.2 integer storage")
        integer_points.append((*coordinates, classification))

    header_size = 227
    point_record_length = 20
    header = bytearray(header_size)
    struct.pack_into("<4s", header, 0, b"LASF")
    struct.pack_into("<H", header, 4, 0)
    struct.pack_into("<H", header, 6, 0)
    struct.pack_into("<BB", header, 24, 1, 2)
    system_identifier = b"roofer OFF converter"
    generating_software = b"tools/off_to_roofer.py"
    header[26 : 26 + len(system_identifier)] = system_identifier
    header[58 : 58 + len(generating_software)] = generating_software
    today = date.today()
    struct.pack_into("<HH", header, 90, today.timetuple().tm_yday, today.year)
    struct.pack_into("<H", header, 94, header_size)
    struct.pack_into("<I", header, 96, header_size)
    struct.pack_into("<I", header, 100, 0)
    struct.pack_into("<B", header, 104, 0)
    struct.pack_into("<H", header, 105, point_record_length)
    struct.pack_into("<I", header, 107, len(integer_points))

    by_return = [len(integer_points), 0, 0, 0, 0]
    struct.pack_into("<5I", header, 111, *by_return)
    struct.pack_into("<3d", header, 131, *scale)
    struct.pack_into("<3d", header, 155, *offsets)
    quantized_minimum = tuple(
        min(coordinates[index] * scale[index] + offsets[index] for coordinates in integer_points)
        for index in range(3)
    )
    quantized_maximum = tuple(
        max(coordinates[index] * scale[index] + offsets[index] for coordinates in integer_points)
        for index in range(3)
    )
    struct.pack_into("<d", header, 179, quantized_maximum[0])
    struct.pack_into("<d", header, 187, quantized_minimum[0])
    struct.pack_into("<d", header, 195, quantized_maximum[1])
    struct.pack_into("<d", header, 203, quantized_minimum[1])
    struct.pack_into("<d", header, 211, quantized_maximum[2])
    struct.pack_into("<d", header, 219, quantized_minimum[2])

    with path.open("wb") as stream:
        stream.write(header)
        for x, y, z, classification in integer_points:
            stream.write(struct.pack("<iiiHBBBBH", x, y, z, 0, 1, classification, 0, 0, 0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a mesh OFF plus footprint OFF into Roofer LAS and GeoJSON inputs."
    )
    parser.add_argument("mesh_off", type=Path, help="Triangulated source mesh in ASCII OFF format.")
    parser.add_argument("footprint_off", type=Path, help="Footprint polygon in ASCII OFF format.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory. Files are written below <output>/input/.",
    )
    parser.add_argument(
        "--building-id",
        help="Building identifier stored in GeoJSON; defaults to the parent directory name.",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=0.25,
        help="Roof sampling cell size in projected units (default: 0.25).",
    )
    parser.add_argument(
        "--normal-z-threshold",
        type=float,
        default=0.2,
        help="Keep a triangle when abs(normal.z)/length >= this value (default: 0.2).",
    )
    parser.add_argument(
        "--fill-passes",
        type=int,
        default=3,
        help="Maximum small-hole fill passes (default: 3).",
    )
    parser.add_argument(
        "--fill-max-range",
        type=float,
        default=2.0,
        help="Maximum neighbour elevation range for a fill (default: 2.0).",
    )
    parser.add_argument("--roof-class", type=int, default=6, help="LAS roof/building class (default: 6).")
    parser.add_argument("--ground-class", type=int, default=2, help="LAS ground class (default: 2).")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.grid_size <= 0:
        raise ValueError("--grid-size must be greater than zero")
    if not 0.0 <= args.normal_z_threshold <= 1.0:
        raise ValueError("--normal-z-threshold must be between 0 and 1")
    if args.fill_passes < 0:
        raise ValueError("--fill-passes must be non-negative")
    if args.fill_max_range < 0:
        raise ValueError("--fill-max-range must be non-negative")
    for name in ("roof_class", "ground_class"):
        value = getattr(args, name)
        if not 0 <= value <= 255:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 255")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
        started = time.perf_counter()
        mesh = read_off(args.mesh_off)
        footprint_mesh = read_off(args.footprint_off)
        polygon = select_polygon(footprint_mesh, args.footprint_off)
        building_id = args.building_id or args.mesh_off.parent.name or args.mesh_off.stem

        roof_grid, filled_points = rasterize_roof(
            mesh=mesh,
            polygon=polygon,
            grid_size=args.grid_size,
            normal_z_threshold=args.normal_z_threshold,
            fill_passes=args.fill_passes,
            fill_max_range=args.fill_max_range,
        )
        polygon_min_x, polygon_min_y, _, _ = polygon_bounds(polygon)
        ground_points = [
            vertex
            for vertex in mesh.vertices
            if not point_in_polygon(vertex.x, vertex.y, polygon)
        ]
        roof_points = [
            Vertex(
                _cell_center(i, polygon_min_x, args.grid_size),
                _cell_center(j, polygon_min_y, args.grid_size),
                elevation,
            )
            for (i, j), elevation in sorted(roof_grid.items())
        ]

        output_dir = args.output
        input_dir = output_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        suffix = _format_grid_suffix(args.grid_size)
        las_path = input_dir / f"{building_id}_roof_grid{suffix}.las"
        geojson_path = input_dir / f"{building_id}_P.geojson"
        stats_path = input_dir / "stats.json"

        write_las(
            las_path,
            roof_points=roof_points,
            ground_points=ground_points,
            roof_class=args.roof_class,
            ground_class=args.ground_class,
        )
        write_geojson(geojson_path, polygon, building_id)

        stats = {
            "source_vertices": len(mesh.vertices),
            "source_faces": len(mesh.faces),
            "ground_points": len(ground_points),
            "roof_grid_points": len(roof_grid) - filled_points,
            "filled_points": filled_points,
            "total_points": len(roof_points) + len(ground_points),
            "grid": args.grid_size,
            "normal_filter": args.normal_z_threshold,
            "hole_fill_max_height_range": args.fill_max_range,
            "fill_passes": args.fill_passes,
            "building_id": building_id,
            "las": str(las_path),
            "geojson": str(geojson_path),
            "seconds": round(time.perf_counter() - started, 3),
        }
        with stats_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(stats, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, IndexError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
