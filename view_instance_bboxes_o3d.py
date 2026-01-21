"""
Visualize per-instance AABBs produced by semantic_octree_blocks_purity90.py (instance mode).

Typical usage:
  .venv\\Scripts\\python.exe view_instance_bboxes_o3d.py ^
    --pcd data\\training_10_classes\\Lille1_1.ply ^
    --bbox_jsonl Lille1_1_instbboxes.jsonl ^
    --downsample 0.20 --max_instances 200

Focus on one instance (optionally crop points inside bbox):
  .venv\\Scripts\\python.exe view_instance_bboxes_o3d.py ^
    --pcd data\\training_10_classes\\Lille1_1.ply ^
    --bbox_jsonl Lille1_1_instbboxes.jsonl ^
    --focus_instance 3 --crop_points --downsample 0.20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
from plyfile import PlyData


def _tab20_colors(values: np.ndarray) -> np.ndarray:
    u, inv = np.unique(values.astype(np.int64, copy=False), return_inverse=True)
    rng = np.random.default_rng(0)
    colors = (rng.random((u.size, 3)) * 0.85 + 0.15).astype(np.float64)
    return colors[inv]


def _pick_field(names: set[str], cands: Tuple[str, ...]) -> Optional[str]:
    for k in cands:
        if k in names:
            return k
    return None


def read_bboxes_jsonl(path: Path) -> Tuple[Optional[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    instance_ids = []
    bmin = []
    bmax = []
    ratio = []
    target_classes = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            instance_ids.append(int(rec["instance_id"]))
            bmin.append(rec["bbox_min"])
            bmax.append(rec["bbox_max"])
            ratio.append(float(rec.get("target_ratio", 0.0)))
            if "target_class" in rec:
                target_classes.append(int(rec["target_class"]))
    if not instance_ids:
        return (
            None,
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float32),
        )
    target_class: Optional[int] = None
    if target_classes:
        uniq = sorted(set(target_classes))
        if len(uniq) != 1:
            raise ValueError(f"bbox_jsonl contains multiple target_class values: {uniq}")
        target_class = int(uniq[0])
    return (
        target_class,
        np.asarray(instance_ids, dtype=np.int32),
        np.asarray(bmin, dtype=np.float64),
        np.asarray(bmax, dtype=np.float64),
        np.asarray(ratio, dtype=np.float32),
    )


def read_ply_xyz_class(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path), mmap="r")
    v = ply["vertex"].data
    names = set(v.dtype.names or ())

    kx = _pick_field(names, ("x", "X", "x_coord"))
    ky = _pick_field(names, ("y", "Y", "y_coord"))
    kz = _pick_field(names, ("z", "Z", "z_coord"))
    kc = _pick_field(names, ("class", "Class", "label", "semantic", "sem_class"))
    if not (kx and ky and kz and kc):
        raise ValueError(f"PLY missing required fields, got: {sorted(names)}")

    xyz = np.vstack([v[kx], v[ky], v[kz]]).T.astype(np.float64, copy=False)
    cls = np.asarray(v[kc]).astype(np.int64, copy=False)
    return xyz, cls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", type=str, default=None, help="Optional point cloud PLY to show as background")
    ap.add_argument("--bbox_jsonl", type=str, required=True, help="Instance bbox JSONL produced by semantic_octree_blocks_purity90.py")
    ap.add_argument("--downsample", type=float, default=0.20, help="Voxel downsample size for --pcd (meters)")
    ap.add_argument("--max_points", type=int, default=None, help="Optional random subsample after filtering")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for --max_points")
    ap.add_argument("--max_instances", type=int, default=200, help="Max number of bboxes to draw (after filtering)")
    ap.add_argument("--ratio_min", type=float, default=0.0, help="Only draw instances with target_ratio >= ratio_min")
    ap.add_argument("--focus_instance", type=int, default=None, help="Only draw one instance id")
    ap.add_argument("--crop_points", action="store_true", help="If set with --focus_instance, only show points inside that bbox")
    ap.add_argument("--target_only", action="store_true", help="Only show points whose class == target_class (requires class field in --pcd PLY)")
    ap.add_argument("--target_class", type=int, default=None, help="Used with --target_only; if omitted, read from bbox_jsonl target_class")
    ap.add_argument("--export_ply", type=str, default=None, help="If set with --focus_instance, export cropped points to this PLY path")
    ap.add_argument("--no_view", action="store_true", help="Do not open a viewer window (useful with --export_ply)")
    args = ap.parse_args()

    bbox_path = Path(args.bbox_jsonl)
    if not bbox_path.exists():
        raise SystemExit(f"Not found: {bbox_path}")

    target_class_jsonl, inst, bmin, bmax, ratio = read_bboxes_jsonl(bbox_path)
    if inst.size == 0:
        raise SystemExit(f"No instances in: {bbox_path}")

    keep = ratio >= float(args.ratio_min)
    if args.focus_instance is not None:
        keep &= inst == int(args.focus_instance)
    inst = inst[keep]
    bmin = bmin[keep]
    bmax = bmax[keep]
    ratio = ratio[keep]

    if inst.size == 0:
        raise SystemExit("No instances after filtering")

    if inst.size > int(args.max_instances):
        order = np.argsort(ratio)[::-1][: int(args.max_instances)]
        inst = inst[order]
        bmin = bmin[order]
        bmax = bmax[order]

    geoms = []

    if args.pcd is not None:
        p = Path(args.pcd)
        if not p.exists():
            raise SystemExit(f"Not found: {p}")

        # For "target_only", we must read the class field; Open3D's default PLY reader drops custom fields.
        if args.target_only:
            target_class = int(args.target_class) if args.target_class is not None else target_class_jsonl
            if target_class is None:
                raise SystemExit("--target_only requires --target_class or bbox_jsonl containing target_class")

            xyz, cls = read_ply_xyz_class(p)
            mask = cls == int(target_class)
            xyz = xyz[mask]

            if args.focus_instance is not None and inst.size == 1:
                in_box = np.all((xyz >= bmin[0]) & (xyz <= bmax[0]), axis=1)
                xyz = xyz[in_box]

            if args.max_points is not None and xyz.shape[0] > int(args.max_points):
                rng = np.random.default_rng(int(args.seed))
                sel = rng.choice(xyz.shape[0], size=int(args.max_points), replace=False)
                xyz = xyz[sel]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
            if float(args.downsample) > 0:
                pcd = pcd.voxel_down_sample(float(args.downsample))
            # Color points to match the focused instance (if any), else use a consistent target color.
            if args.focus_instance is not None and inst.size == 1:
                color = _tab20_colors(np.asarray([int(args.focus_instance)], dtype=np.int32))[0]
                pcd.paint_uniform_color(color.tolist())
            else:
                pcd.paint_uniform_color([0.95, 0.75, 0.15])
        else:
            pcd = o3d.io.read_point_cloud(str(p))
            if float(args.downsample) > 0:
                pcd = pcd.voxel_down_sample(float(args.downsample))
            pcd.paint_uniform_color([0.75, 0.75, 0.78])

        cropped_pcd: Optional[o3d.geometry.PointCloud] = None
        if (args.crop_points or args.export_ply) and args.focus_instance is not None and inst.size == 1:
            aabb = o3d.geometry.AxisAlignedBoundingBox(bmin[0], bmax[0])
            cropped_pcd = pcd.crop(aabb)
            if args.crop_points:
                pcd = cropped_pcd
        geoms.append(pcd)

        if args.export_ply is not None:
            if args.focus_instance is None or inst.size != 1:
                raise SystemExit("--export_ply requires --focus_instance that matches exactly one bbox after filtering")
            if cropped_pcd is None:
                aabb = o3d.geometry.AxisAlignedBoundingBox(bmin[0], bmax[0])
                cropped_pcd = pcd.crop(aabb)
            color = _tab20_colors(np.asarray([int(args.focus_instance)], dtype=np.int32))[0]
            cropped_pcd.paint_uniform_color(color.tolist())
            out = Path(args.export_ply)
            o3d.io.write_point_cloud(str(out), cropped_pcd, write_ascii=True)
            print(f"[DONE] Wrote: {out}")

    colors = _tab20_colors(inst)
    for i in range(inst.size):
        aabb = o3d.geometry.AxisAlignedBoundingBox(bmin[i], bmax[i])
        lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
        lines.paint_uniform_color(colors[i].tolist())
        geoms.append(lines)

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0))
    if not args.no_view:
        o3d.visualization.draw_geometries(geoms, window_name="Instance AABBs", width=1280, height=800)


if __name__ == "__main__":
    main()
