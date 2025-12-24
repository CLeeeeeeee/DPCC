"""
Semantic octree blocks (purity-threshold leaves + bottom-up sibling merge).

For a labeled PLY (x/y/z + class), this script:
1) Chooses a root cube covering the whole cloud.
2) Builds a fine uniform grid at depth D (or leaf_size), i.e. 2^D cubes per axis.
3) For each occupied leaf cube, computes semantic purity = max_class_count / total_count.
4) Keeps cubes with purity >= threshold and assigns their semantic class = argmax class.
5) Bottom-up merges siblings (8 children) with the same semantic class to form larger semantic blocks.

This “fine grid + merge” approach is equivalent to adaptive splitting until purity is satisfied
when D (leaf_size) is small enough that leaves become pure.

Example:
  .venv\\Scripts\\python.exe semantic_octree_blocks_purity90.py data\\training_10_classes\\Lille1_1.ply ^
    --purity 0.90 --leaf_size 0.20 --out_jsonl Lille1_1_semblocks.jsonl --out_ply Lille1_1_semblocks_centers.ply
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from plyfile import PlyData, PlyElement


def _pick_field(names: set[str], cands: Iterable[str]) -> Optional[str]:
    for k in cands:
        if k in names:
            return k
    return None


def read_ply_xyz_class(
    path: Path, mmap: str = "r"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path), mmap=mmap)
    v = ply["vertex"].data
    names = set(v.dtype.names or ())

    kx = _pick_field(names, ("x", "X", "x_coord"))
    ky = _pick_field(names, ("y", "Y", "y_coord"))
    kz = _pick_field(names, ("z", "Z", "z_coord"))
    kcls = _pick_field(names, ("class", "Class", "label", "semantic", "sem_class"))
    if not (kx and ky and kz and kcls):
        raise ValueError(f"Missing required fields in PLY vertex: have={sorted(names)}")

    x = np.asarray(v[kx], dtype=np.float32)
    y = np.asarray(v[ky], dtype=np.float32)
    z = np.asarray(v[kz], dtype=np.float32)
    cls_raw = np.asarray(v[kcls])
    return x, y, z, cls_raw


def root_cube_from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, float]:
    xyz_min = np.array([x.min(), y.min(), z.min()], dtype=np.float64)
    xyz_max = np.array([x.max(), y.max(), z.max()], dtype=np.float64)
    center = (xyz_min + xyz_max) / 2.0
    span = float(np.max(xyz_max - xyz_min))
    edge = float(span) if float(span) > 0 else 1.0
    cube_min = (center - edge / 2.0).astype(np.float64)
    return cube_min, edge


def _split_by_2bits_3way(x: np.ndarray) -> np.ndarray:
    # Spread lower 21 bits so that there are 2 zeros between each bit.
    x = x.astype(np.uint64) & np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def morton3d_encode(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    x = _split_by_2bits_3way(ix)
    y = _split_by_2bits_3way(iy) << np.uint64(1)
    z = _split_by_2bits_3way(iz) << np.uint64(2)
    return x | y | z


def _compact_3way_to_1(x: np.ndarray) -> np.ndarray:
    # Inverse of _split_by_2bits_3way.
    x = x.astype(np.uint64) & np.uint64(0x1249249249249249)
    x = (x ^ (x >> np.uint64(2))) & np.uint64(0x10C30C30C30C30C3)
    x = (x ^ (x >> np.uint64(4))) & np.uint64(0x100F00F00F00F00F)
    x = (x ^ (x >> np.uint64(8))) & np.uint64(0x1F0000FF0000FF)
    x = (x ^ (x >> np.uint64(16))) & np.uint64(0x1F00000000FFFF)
    x = (x ^ (x >> np.uint64(32))) & np.uint64(0x1FFFFF)
    return x.astype(np.uint32)


def morton3d_decode(code: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    code = code.astype(np.uint64)
    ix = _compact_3way_to_1(code)
    iy = _compact_3way_to_1(code >> np.uint64(1))
    iz = _compact_3way_to_1(code >> np.uint64(2))
    return ix, iy, iz


def depth_from_leaf_size(edge: float, leaf_size: float, max_depth: int = 21) -> int:
    if leaf_size <= 0:
        raise ValueError("leaf_size must be > 0")
    d = int(math.ceil(math.log2(edge / leaf_size))) if edge > leaf_size else 0
    return int(min(max(d, 0), max_depth))


def _bits_for_k(k: int) -> int:
    return int(math.ceil(math.log2(max(2, k))))


@dataclass(frozen=True)
class BlocksResult:
    cube_min: np.ndarray  # (3,) float64
    cube_edge: float
    class_ids: np.ndarray  # (K,) raw class ids corresponding to label_idx
    threshold: float
    leaf_depth: int
    occupied_leaf_voxels: int
    kept_leaf_voxels: int
    # Final semantic blocks after merge (varying depth):
    depths: np.ndarray  # (B,) uint8
    codes: np.ndarray  # (B,) uint64 morton code at that depth
    labels: np.ndarray  # (B,) uint16 label_idx in [0..K)
    n_points: np.ndarray  # (B,) uint32 points covered (from leaves)


def build_semantic_blocks(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls_raw: np.ndarray,
    *,
    depth: int,
    purity: float = 0.90,
    chunk_points: int = 2_000_000,
) -> BlocksResult:
    if not (0.0 < purity <= 1.0):
        raise ValueError("purity must be in (0, 1]")
    if not (0 <= depth <= 21):
        raise ValueError("depth must be in [0, 21] for this morton implementation")
    if x.shape[0] != cls_raw.shape[0] or y.shape[0] != cls_raw.shape[0] or z.shape[0] != cls_raw.shape[0]:
        raise ValueError("x/y/z and cls_raw must have same length")

    cube_min, edge = root_cube_from_xyz(x, y, z)

    class_ids = np.unique(cls_raw)
    class_ids_sorted = np.sort(class_ids)
    k = int(class_ids_sorted.shape[0])
    if k < 1:
        raise ValueError("No classes found")

    # Map raw class ids to 0..K-1, using searchsorted (fast, vectorized).
    cls_idx = np.searchsorted(class_ids_sorted, cls_raw).astype(np.uint16, copy=False)

    bits_k = _bits_for_k(k)
    if 3 * depth + bits_k > 64:
        raise ValueError(
            f"depth={depth} too large for packing (3*depth+bits_k={3*depth+bits_k} > 64). "
            f"Try smaller depth/leaf_size."
        )
    label_mask = np.uint64((1 << bits_k) - 1)

    n = cls_raw.shape[0]
    # Create packed key per point: (morton_code << bits_k) | label_idx.
    packed = np.empty(n, dtype=np.uint64)
    scale = (2**depth) / edge
    max_coord = np.uint32((2**depth) - 1) if depth > 0 else np.uint32(0)

    # Chunked generation to avoid holding ix/iy/iz for all points at once.
    for s in range(0, n, chunk_points):
        e = min(n, s + chunk_points)
        # Use float64 in the normalization to reduce boundary artifacts.
        tx = (x[s:e].astype(np.float64, copy=False) - cube_min[0]) * scale
        ty = (y[s:e].astype(np.float64, copy=False) - cube_min[1]) * scale
        tz = (z[s:e].astype(np.float64, copy=False) - cube_min[2]) * scale
        if depth > 0:
            ix = np.floor(tx).astype(np.int64)
            iy = np.floor(ty).astype(np.int64)
            iz = np.floor(tz).astype(np.int64)
            ix = np.clip(ix, 0, int(max_coord)).astype(np.uint32, copy=False)
            iy = np.clip(iy, 0, int(max_coord)).astype(np.uint32, copy=False)
            iz = np.clip(iz, 0, int(max_coord)).astype(np.uint32, copy=False)
            code = morton3d_encode(ix, iy, iz)
        else:
            code = np.zeros((e - s,), dtype=np.uint64)
        packed[s:e] = (code << np.uint64(bits_k)) | cls_idx[s:e].astype(np.uint64)

    # Unique over (voxel, label) pairs and counts, in-place sort inside np.unique.
    uniq_pair, pair_counts = np.unique(packed, return_counts=True)
    del packed

    voxel_code = uniq_pair >> np.uint64(bits_k)
    label_idx = (uniq_pair & label_mask).astype(np.uint16, copy=False)

    # Total points per voxel (sum over labels).
    voxel_unique, voxel_start = np.unique(voxel_code, return_index=True)
    total_per_voxel = np.add.reduceat(pair_counts, voxel_start).astype(np.uint32, copy=False)

    # Majority label per voxel: sort the (voxel,label) pairs by (voxel, count) and take last in each voxel group.
    order = np.lexsort((pair_counts, voxel_code))
    voxel_s = voxel_code[order]
    counts_s = pair_counts[order]
    labels_s = label_idx[order]
    is_last = np.r_[voxel_s[1:] != voxel_s[:-1], True]
    voxel_last = voxel_s[is_last]
    max_count = counts_s[is_last].astype(np.uint32, copy=False)
    max_label = labels_s[is_last].astype(np.uint16, copy=False)

    # Align totals with voxel_last (both sorted by voxel code).
    if voxel_last.shape[0] != voxel_unique.shape[0] or not np.array_equal(voxel_last, voxel_unique):
        raise RuntimeError("Internal grouping mismatch while computing majority labels")

    purity_per_voxel = max_count.astype(np.float64) / np.maximum(total_per_voxel.astype(np.float64), 1.0)
    keep = purity_per_voxel >= float(purity)

    leaf_codes = voxel_unique[keep].astype(np.uint64, copy=False)
    leaf_labels = max_label[keep].astype(np.uint16, copy=False)
    leaf_points = total_per_voxel[keep].astype(np.uint32, copy=False)
    occupied_leaf_voxels = int(voxel_unique.size)
    kept_leaf_voxels = int(leaf_codes.size)

    # Bottom-up merge siblings with same label.
    nodes_by_depth: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        depth: (leaf_codes, leaf_labels, leaf_points)
    }

    for d in range(depth, 0, -1):
        codes_d, labels_d, pts_d = nodes_by_depth.get(d, (None, None, None))
        if codes_d is None or codes_d.size == 0:
            continue

        parent = codes_d >> np.uint64(3)
        child_bit = (np.uint16(1) << (codes_d & np.uint64(7)).astype(np.uint16, copy=False)).astype(np.uint16)
        # Group key by (parent, label).
        key = (parent.astype(np.uint64) << np.uint64(bits_k)) | labels_d.astype(np.uint64)
        idx = np.argsort(key)

        key_s = key[idx]
        parent_s = parent[idx]
        labels_s2 = labels_d[idx]
        pts_s2 = pts_d[idx]
        bits_s2 = child_bit[idx]

        group_start = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
        group_n = np.diff(np.r_[group_start, key_s.size]).astype(np.int32, copy=False)
        bits_sum = np.add.reduceat(bits_s2.astype(np.uint16), group_start)
        pts_sum = np.add.reduceat(pts_s2.astype(np.uint64), group_start).astype(np.uint32, copy=False)

        can_merge_group = (group_n == 8) & (bits_sum == np.uint16(0xFF))
        if not np.any(can_merge_group):
            nodes_by_depth[d] = (codes_d, labels_d, pts_d)
            continue

        merged_parent_codes = parent_s[group_start][can_merge_group].astype(np.uint64, copy=False)
        merged_labels = labels_s2[group_start][can_merge_group].astype(np.uint16, copy=False)
        merged_points = pts_sum[can_merge_group].astype(np.uint32, copy=False)

        # Keep members of groups that were NOT merged.
        keep_members = np.repeat(~can_merge_group, group_n)
        kept_codes = codes_d[idx][keep_members]
        kept_labels = labels_d[idx][keep_members]
        kept_points = pts_d[idx][keep_members]
        nodes_by_depth[d] = (kept_codes, kept_labels, kept_points)

        # Append merged parents into depth d-1.
        prev = nodes_by_depth.get(d - 1)
        if prev is None:
            nodes_by_depth[d - 1] = (merged_parent_codes, merged_labels, merged_points)
        else:
            pc, pl, pp = prev
            nodes_by_depth[d - 1] = (
                np.concatenate([pc, merged_parent_codes]),
                np.concatenate([pl, merged_labels]),
                np.concatenate([pp, merged_points]),
            )

    # Gather final blocks across depths.
    depths_list: List[np.ndarray] = []
    codes_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    pts_list: List[np.ndarray] = []
    for d, (c, l, p) in sorted(nodes_by_depth.items()):
        if c.size == 0:
            continue
        depths_list.append(np.full((c.size,), d, dtype=np.uint8))
        codes_list.append(c.astype(np.uint64, copy=False))
        labels_list.append(l.astype(np.uint16, copy=False))
        pts_list.append(p.astype(np.uint32, copy=False))

    depths_out = np.concatenate(depths_list) if depths_list else np.zeros((0,), dtype=np.uint8)
    codes_out = np.concatenate(codes_list) if codes_list else np.zeros((0,), dtype=np.uint64)
    labels_out = np.concatenate(labels_list) if labels_list else np.zeros((0,), dtype=np.uint16)
    pts_out = np.concatenate(pts_list) if pts_list else np.zeros((0,), dtype=np.uint32)

    return BlocksResult(
        cube_min=cube_min,
        cube_edge=edge,
        class_ids=class_ids_sorted,
        threshold=float(purity),
        leaf_depth=int(depth),
        occupied_leaf_voxels=occupied_leaf_voxels,
        kept_leaf_voxels=kept_leaf_voxels,
        depths=depths_out,
        codes=codes_out,
        labels=labels_out,
        n_points=pts_out,
    )


def blocks_to_world(
    res: BlocksResult,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Returns (center_xyz, min_xyz, max_xyz, edge_len, class_raw)
    if res.codes.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=res.class_ids.dtype),
        )

    ix, iy, iz = morton3d_decode(res.codes)
    d = res.depths.astype(np.int32, copy=False)
    div = (2.0 ** d.astype(np.float64))
    size = (res.cube_edge / div).astype(np.float64, copy=False)  # (B,)

    corner = np.empty((res.codes.size, 3), dtype=np.float64)
    corner[:, 0] = res.cube_min[0] + ix.astype(np.float64) * size
    corner[:, 1] = res.cube_min[1] + iy.astype(np.float64) * size
    corner[:, 2] = res.cube_min[2] + iz.astype(np.float64) * size
    center = corner + (size[:, None] * 0.5)
    max_corner = corner + size[:, None]
    class_raw = res.class_ids[res.labels]
    return center, corner, max_corner, size, class_raw


def write_blocks_jsonl(res: BlocksResult, path: Path) -> None:
    center, corner, max_corner, size, class_raw = blocks_to_world(res)
    with path.open("w", encoding="utf-8") as f:
        for i in range(res.codes.size):
            rec = {
                "depth": int(res.depths[i]),
                "morton": int(res.codes[i]),
                "class": int(class_raw[i]),
                "n_points": int(res.n_points[i]),
                "cube_min": [float(corner[i, 0]), float(corner[i, 1]), float(corner[i, 2])],
                "cube_max": [float(max_corner[i, 0]), float(max_corner[i, 1]), float(max_corner[i, 2])],
                "cube_edge": float(size[i]),
                "center": [float(center[i, 0]), float(center[i, 1]), float(center[i, 2])],
                "leaf_purity_threshold": float(res.threshold),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _tab20_colors(class_values: np.ndarray) -> np.ndarray:
    # Deterministic pseudo-colors without extra deps.
    # Map raw class ids -> [0,1] rgb.
    u, inv = np.unique(class_values.astype(np.int64, copy=False), return_inverse=True)
    rng = np.random.default_rng(0)
    colors = (rng.random((u.size, 3)) * 0.85 + 0.15).astype(np.float32)
    return colors[inv]


def write_blocks_centers_ply(res: BlocksResult, path: Path) -> None:
    center, _, _, size, class_raw = blocks_to_world(res)
    rgb = _tab20_colors(class_raw)
    data = np.empty(
        (center.shape[0],),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("class", "i4"),
            ("depth", "u1"),
            ("cube_edge", "f4"),
            ("n_points", "u4"),
        ],
    )
    data["x"] = center[:, 0].astype(np.float32)
    data["y"] = center[:, 1].astype(np.float32)
    data["z"] = center[:, 2].astype(np.float32)
    data["red"] = np.clip(rgb[:, 0] * 255.0, 0, 255).astype(np.uint8)
    data["green"] = np.clip(rgb[:, 1] * 255.0, 0, 255).astype(np.uint8)
    data["blue"] = np.clip(rgb[:, 2] * 255.0, 0, 255).astype(np.uint8)
    data["class"] = class_raw.astype(np.int32, copy=False)
    data["depth"] = res.depths.astype(np.uint8, copy=False)
    data["cube_edge"] = size.astype(np.float32, copy=False)
    data["n_points"] = res.n_points.astype(np.uint32, copy=False)

    PlyData([PlyElement.describe(data, "vertex")], text=True).write(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=str, help="Input labeled PLY (must contain x/y/z and class)")
    ap.add_argument("--purity", type=float, default=0.90, help="Leaf purity threshold, default 0.90")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--depth", type=int, default=None, help="Fixed leaf depth D (0..21)")
    g.add_argument("--leaf_size", type=float, default=0.20, help="Target leaf cube edge length (meters)")
    ap.add_argument("--chunk_points", type=int, default=2_000_000, help="Chunk size for morton encoding")
    ap.add_argument("--mmap", type=str, default="r", help="plyfile mmap mode: r/c/False (default: r)")
    ap.add_argument("--max_points", type=int, default=None, help="Optional random subsample for quick runs")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for --max_points")
    ap.add_argument("--out_jsonl", type=str, default=None, help="Write merged semantic blocks to JSONL")
    ap.add_argument("--out_ply", type=str, default=None, help="Write merged block centers to a colored PLY")
    args = ap.parse_args()

    src = Path(args.ply)
    if not src.exists():
        raise SystemExit(f"Not found: {src}")

    print(f"[INFO] Reading: {src}")
    x, y, z, cls_raw = read_ply_xyz_class(src, mmap=args.mmap)
    if args.max_points is not None and cls_raw.shape[0] > int(args.max_points):
        rng = np.random.default_rng(int(args.seed))
        sel = rng.choice(cls_raw.shape[0], size=int(args.max_points), replace=False)
        x = x[sel].astype(np.float32, copy=False)
        y = y[sel].astype(np.float32, copy=False)
        z = z[sel].astype(np.float32, copy=False)
        cls_raw = cls_raw[sel]
        print(f"[INFO] Subsampled to {cls_raw.shape[0]:,} points (seed={args.seed})")
    cube_min, edge = root_cube_from_xyz(x, y, z)

    if args.depth is None:
        depth = depth_from_leaf_size(edge, float(args.leaf_size), max_depth=21)
    else:
        depth = int(args.depth)
    print(f"[INFO] Root cube edge={edge:.3f} m, cube_min={cube_min.tolist()}, leaf depth={depth}, threshold={args.purity:.2f}")

    res = build_semantic_blocks(
        x,
        y,
        z,
        cls_raw,
        depth=depth,
        purity=float(args.purity),
        chunk_points=int(args.chunk_points),
    )
    print(
        f"[INFO] Leaf voxels occupied={res.occupied_leaf_voxels:,}, kept(purity>={res.threshold:.2f})={res.kept_leaf_voxels:,}"
    )
    print(f"[INFO] Semantic blocks (after merge): {res.codes.size:,}")

    if args.out_jsonl:
        out = Path(args.out_jsonl)
        write_blocks_jsonl(res, out)
        print(f"[DONE] Wrote: {out}")
    if args.out_ply:
        out = Path(args.out_ply)
        write_blocks_centers_ply(res, out)
        print(f"[DONE] Wrote: {out}")


if __name__ == "__main__":
    main()
