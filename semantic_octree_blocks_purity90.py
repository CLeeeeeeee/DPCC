"""
Semantic octree blocks (purity-threshold leaves + bottom-up sibling merge).

For a labeled PLY (x/y/z + class), this script supports two modes:

Semantic mode (default, majority-class purity):
1) Chooses a root cube covering the whole cloud.
2) Builds counts for octree nodes up to a max depth (set by --depth or --leaf_size).
3) Top-down: if a node purity < threshold, it is split; this repeats until purity >= threshold.
4) Leaves are assigned the majority semantic class of that node.
5) Bottom-up merges siblings (8 children) with the same semantic class, but only if the merged parent
   also satisfies the purity threshold.

Instance mode (--target_class, per-instance blocks for ONE target class):
1) Voxelize at max depth, compute target ratio = target_points / total_points per voxel.
2) Core voxels: target_ratio >= --target_ratio; run connected-components with --connectivity (default 26).
3) One-step shell expansion: add neighbor voxels with target_ratio >= --shell_ratio; ambiguous shell voxels
   touching multiple cores are dropped to avoid bridging instances.
4) For each instance, bottom-up merge siblings while the merged parent target_ratio stays >= --target_ratio.

Note: if max depth is reached and a node is still below the purity threshold (rare, but possible with
label noise), the node is output as an impure leaf (purity is written to outputs).

Example (semantic):
  .venv\\Scripts\\python.exe semantic_octree_blocks_purity90.py data\\training_10_classes\\Lille1_1.ply ^
    --purity 0.90 --leaf_size 0.20 --out_jsonl Lille1_1_semblocks.jsonl --out_ply Lille1_1_semblocks_centers.ply

Example (instance):
  .venv\\Scripts\\python.exe semantic_octree_blocks_purity90.py data\\training_10_classes\\Lille1_1.ply ^
    --target_class 202020000 --target_ratio 0.90 --shell_ratio 0.60 --leaf_size 0.20 ^
    --out_jsonl Lille1_1_instblocks.jsonl --out_ply Lille1_1_instblocks_centers.ply

Example (instance + per-instance AABB):
  .venv\\Scripts\\python.exe semantic_octree_blocks_purity90.py data\\training_10_classes\\Lille1_1.ply ^
    --target_class 202020000 --target_ratio 0.90 --shell_ratio 0.60 --leaf_size 0.20 ^
    --out_bbox_jsonl Lille1_1_instbboxes.jsonl --out_bbox_ply Lille1_1_instbboxes_centers.ply
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

#在已有字段名集合names中，从候选名cands中挑第一个存在的字段名
def _pick_field(names: set[str], cands: Iterable[str]) -> Optional[str]:
    for k in cands:
        if k in names:
            return k
    return None

#从PLY里读取x,y,z,class
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

#找到根立方体
def root_cube_from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, float]:
    xyz_min = np.array([x.min(), y.min(), z.min()], dtype=np.float64)
    xyz_max = np.array([x.max(), y.max(), z.max()], dtype=np.float64)
    center = (xyz_min + xyz_max) / 2.0
    span = float(np.max(xyz_max - xyz_min))
    edge = float(span) if float(span) > 0 else 1.0
    cube_min = (center - edge / 2.0).astype(np.float64)
    return cube_min, edge

#Morton编解码（71-105）：先腾空再插值
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

#深度
def depth_from_leaf_size(edge: float, leaf_size: float, max_depth: int = 21) -> int:
    if leaf_size <= 0:
        raise ValueError("leaf_size must be > 0")
    d = int(math.ceil(math.log2(edge / leaf_size))) if edge > leaf_size else 0
    return int(min(max(d, 0), max_depth))

#把类别索引打包进64bit
def _bits_for_k(k: int) -> int:
    return int(math.ceil(math.log2(max(2, k))))

#结果结构体（返回值）
@dataclass(frozen=True)
class BlocksResult:
    cube_min: np.ndarray  # (3,) float64
    cube_edge: float
    class_ids: np.ndarray  # (K,) raw class ids corresponding to label_idx
    threshold: float
    leaf_depth: int
    occupied_leaf_voxels: int
    kept_leaf_voxels: int
    forced_impure_leaves: int
    # Final semantic blocks after merge (varying depth):
    depths: np.ndarray  # (B,) uint8
    codes: np.ndarray  # (B,) uint64 morton code at that depth
    labels: np.ndarray  # (B,) uint16 label_idx in [0..K)
    n_points: np.ndarray  # (B,) uint32 points covered (from leaves)
    purities: np.ndarray  # (B,) float32, max_class_count / total_count at that node

#内部辅助函数，计算出每个node_code的总点数、主类、主类点数
def _node_stats_from_pairs(
    codes: np.ndarray, labels: np.ndarray, counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Input: per-(code,label) counts at one depth.
    # Output: per-code totals, majority label, majority count.
    if codes.size == 0:
        z0 = np.zeros((0,), dtype=np.uint64)
        z1 = np.zeros((0,), dtype=np.uint32)
        z2 = np.zeros((0,), dtype=np.uint16)
        return z0, z1, z2, z1

    # Total points per code (sum over labels).
    code_unique, code_start = np.unique(codes, return_index=True)
    total_per_code = np.add.reduceat(counts.astype(np.uint64, copy=False), code_start).astype(np.uint32, copy=False)

    # Majority label per code: sort (code, count) and take last entry per code.
    order = np.lexsort((counts, codes))
    codes_s = codes[order]
    counts_s = counts[order]
    labels_s = labels[order]
    is_last = np.r_[codes_s[1:] != codes_s[:-1], True]
    code_last = codes_s[is_last]
    max_count = counts_s[is_last].astype(np.uint32, copy=False)
    max_label = labels_s[is_last].astype(np.uint16, copy=False)

    if code_last.shape[0] != code_unique.shape[0] or not np.array_equal(code_last, code_unique):
        raise RuntimeError("Internal grouping mismatch while computing node stats")

    return code_unique.astype(np.uint64, copy=False), total_per_code, max_label, max_count

#聚合，本质是做父节点的各类点数统计
def _aggregate_pairs_to_parent(
    codes: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    *,
    bits_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Aggregate child (code,label)->count into parent (code>>3,label)->sum(count).
    parent = codes >> np.uint64(3)
    key = (parent.astype(np.uint64) << np.uint64(bits_k)) | labels.astype(np.uint64)
    idx = np.argsort(key)
    key_s = key[idx]
    cnt_s = counts[idx].astype(np.uint64, copy=False)
    start = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
    uniq_key = key_s[start]
    sum_cnt = np.add.reduceat(cnt_s, start).astype(np.uint32, copy=False)
    parent_codes = (uniq_key >> np.uint64(bits_k)).astype(np.uint64, copy=False)
    label_mask = np.uint64((1 << bits_k) - 1)
    parent_labels = (uniq_key & label_mask).astype(np.uint16, copy=False)
    return parent_codes, parent_labels, sum_cnt

# =========================
# Instance-Blocks Mode
# =========================

@dataclass(frozen=True)
class InstanceBlocksResult:
    cube_min: np.ndarray  # (3,) float64
    cube_edge: float
    target_class: int
    threshold: float
    shell_threshold: float
    leaf_depth: int
    instance_ids: np.ndarray  # (B,) int32
    depths: np.ndarray  # (B,) uint8
    codes: np.ndarray  # (B,) uint64
    total_points: np.ndarray  # (B,) uint32
    target_points: np.ndarray  # (B,) uint32
    target_ratios: np.ndarray  # (B,) float32


@dataclass(frozen=True)
class InstanceBBoxesResult:
    cube_min: np.ndarray  # (3,) float64
    cube_edge: float
    target_class: int
    threshold: float
    shell_threshold: float
    leaf_depth: int
    instance_ids: np.ndarray  # (I,) int32
    bbox_min: np.ndarray  # (I,3) float64
    bbox_max: np.ndarray  # (I,3) float64
    block_count: np.ndarray  # (I,) uint32
    total_points: np.ndarray  # (I,) uint64
    target_points: np.ndarray  # (I,) uint64
    target_ratios: np.ndarray  # (I,) float32


def _neighbor_offsets(connectivity: int) -> List[Tuple[int, int, int]]:
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be one of {6,18,26}")
    offsets: List[Tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and manhattan != 1:
                    continue
                if connectivity == 18 and manhattan > 2:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


def _compress_instance_voxels(
    *,
    codes_leaf: np.ndarray,
    total_leaf: np.ndarray,
    target_leaf: np.ndarray,
    depth: int,
    ratio_thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Bottom-up merge within ONE instance: merge siblings only if parent target ratio >= ratio_thr.
    codes_cur = codes_leaf.astype(np.uint64, copy=False)
    total_cur = total_leaf.astype(np.uint32, copy=False)
    target_cur = target_leaf.astype(np.uint32, copy=False)

    out_depths: List[np.ndarray] = []
    out_codes: List[np.ndarray] = []
    out_total: List[np.ndarray] = []
    out_target: List[np.ndarray] = []

    for d in range(depth, 0, -1):
        if codes_cur.size == 0:
            break

        parent = codes_cur >> np.uint64(3)
        child_bit = (np.uint16(1) << (codes_cur & np.uint64(7)).astype(np.uint16, copy=False)).astype(np.uint16)

        idx = np.argsort(parent)
        parent_s = parent[idx]
        bits_s = child_bit[idx]
        total_s = total_cur[idx]
        target_s = target_cur[idx]
        codes_s = codes_cur[idx]

        group_start = np.flatnonzero(np.r_[True, parent_s[1:] != parent_s[:-1]])
        group_n = np.diff(np.r_[group_start, parent_s.size]).astype(np.int32, copy=False)
        bits_sum = np.add.reduceat(bits_s.astype(np.uint16), group_start)

        can_merge = (group_n == 8) & (bits_sum == np.uint16(0xFF))
        if not np.any(can_merge):
            out_depths.append(np.full((codes_cur.size,), d, dtype=np.uint8))
            out_codes.append(codes_cur)
            out_total.append(total_cur)
            out_target.append(target_cur)
            codes_cur = np.zeros((0,), dtype=np.uint64)
            total_cur = np.zeros((0,), dtype=np.uint32)
            target_cur = np.zeros((0,), dtype=np.uint32)
            continue

        total_sum = np.add.reduceat(total_s.astype(np.uint64), group_start).astype(np.uint32, copy=False)
        target_sum = np.add.reduceat(target_s.astype(np.uint64), group_start).astype(np.uint32, copy=False)
        ratio_parent = (target_sum.astype(np.float64) / np.maximum(total_sum.astype(np.float64), 1.0)).astype(np.float32)
        merge_ok = can_merge & (ratio_parent >= float(ratio_thr))

        merged_parent_codes = parent_s[group_start][merge_ok].astype(np.uint64, copy=False)
        merged_total = total_sum[merge_ok].astype(np.uint32, copy=False)
        merged_target = target_sum[merge_ok].astype(np.uint32, copy=False)

        keep_members = np.repeat(~merge_ok, group_n)
        kept_codes = codes_s[keep_members]
        kept_total = total_s[keep_members]
        kept_target = target_s[keep_members]

        if kept_codes.size:
            out_depths.append(np.full((kept_codes.size,), d, dtype=np.uint8))
            out_codes.append(kept_codes.astype(np.uint64, copy=False))
            out_total.append(kept_total.astype(np.uint32, copy=False))
            out_target.append(kept_target.astype(np.uint32, copy=False))

        codes_cur = merged_parent_codes
        total_cur = merged_total
        target_cur = merged_target

    if codes_cur.size:
        out_depths.append(np.zeros((codes_cur.size,), dtype=np.uint8))
        out_codes.append(codes_cur)
        out_total.append(total_cur)
        out_target.append(target_cur)

    depths = np.concatenate(out_depths) if out_depths else np.zeros((0,), dtype=np.uint8)
    codes = np.concatenate(out_codes) if out_codes else np.zeros((0,), dtype=np.uint64)
    total = np.concatenate(out_total) if out_total else np.zeros((0,), dtype=np.uint32)
    target = np.concatenate(out_target) if out_target else np.zeros((0,), dtype=np.uint32)
    ratios = (target.astype(np.float64) / np.maximum(total.astype(np.float64), 1.0)).astype(np.float32)
    return depths, codes, total, target, ratios


def build_target_instance_blocks(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    cls_raw: np.ndarray,
    *,
    depth: int,
    target_class: int,
    target_ratio: float = 0.90,
    shell_ratio: float = 0.60,
    connectivity: int = 26,
    min_core_voxels: int = 10,
    min_core_target_points: int = 30,
    chunk_points: int = 2_000_000,
) -> InstanceBlocksResult:
    if not (0 < target_ratio <= 1.0):
        raise ValueError("target_ratio must be in (0,1]")
    if not (0 <= depth <= 21):
        raise ValueError("depth must be in [0, 21] for this morton implementation")
    if not (0 <= shell_ratio <= 1.0):
        raise ValueError("shell_ratio must be in [0,1]")
    if x.shape[0] != cls_raw.shape[0] or y.shape[0] != cls_raw.shape[0] or z.shape[0] != cls_raw.shape[0]:
        raise ValueError("x/y/z and cls_raw must have same length")

    cube_min, edge = root_cube_from_xyz(x, y, z)
    n = cls_raw.shape[0]

    # Only need a binary label per point: target vs other.
    is_target = (cls_raw.astype(np.int64, copy=False) == int(target_class))

    packed = np.empty(n, dtype=np.uint64)
    scale = (2**depth) / edge
    max_coord = np.uint32((2**depth) - 1) if depth > 0 else np.uint32(0)

    for s in range(0, n, chunk_points):
        e = min(n, s + chunk_points)
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
        packed[s:e] = (code << np.uint64(1)) | is_target[s:e].astype(np.uint64)

    uniq_pair, pair_counts = np.unique(packed, return_counts=True)
    del packed
    leaf_codes = (uniq_pair >> np.uint64(1)).astype(np.uint64, copy=False)
    leaf_bit = (uniq_pair & np.uint64(1)).astype(np.uint8, copy=False)
    leaf_counts = pair_counts.astype(np.uint32, copy=False)

    # Total points per occupied voxel.
    vox_codes, vox_start = np.unique(leaf_codes, return_index=True)
    vox_total = np.add.reduceat(leaf_counts.astype(np.uint64, copy=False), vox_start).astype(np.uint32, copy=False)

    # Target points per occupied voxel (sparse).
    tmask = leaf_bit == np.uint8(1)
    t_codes = leaf_codes[tmask]
    t_counts = leaf_counts[tmask]
    vox_target = np.zeros((vox_codes.size,), dtype=np.uint32)
    if t_codes.size:
        t_unique, t_start = np.unique(t_codes, return_index=True)
        t_sum = np.add.reduceat(t_counts.astype(np.uint64, copy=False), t_start).astype(np.uint32, copy=False)
        tidx = np.searchsorted(vox_codes, t_unique)
        vox_target[tidx] = t_sum

    vox_ratio = (vox_target.astype(np.float64) / np.maximum(vox_total.astype(np.float64), 1.0)).astype(np.float32)

    # Decode for adjacency (depth is fixed here).
    grid = int(2**depth) if depth > 0 else 1
    stride_y = grid
    stride_x = grid * grid
    ix, iy, iz = morton3d_decode(vox_codes)
    lin = (ix.astype(np.int64) * stride_x + iy.astype(np.int64) * stride_y + iz.astype(np.int64)).astype(np.int64)

    # Core voxels are seeds (high ratio).
    core_mask = (vox_ratio >= float(target_ratio)) & (vox_target > 0)
    core_idx = np.nonzero(core_mask)[0].astype(np.int64, copy=False)
    core_lin = lin[core_idx]
    core_lin_to_vox: Dict[int, int] = {int(l): int(i) for l, i in zip(core_lin, core_idx)}
    core_lin_set = set(core_lin_to_vox.keys())

    offsets = _neighbor_offsets(int(connectivity))
    visited: set[int] = set()
    instances_core: List[List[int]] = []

    for l0 in core_lin_to_vox.keys():
        if l0 in visited:
            continue
        stack = [l0]
        visited.add(l0)
        comp_vox: List[int] = []
        while stack:
            lcur = stack.pop()
            vcur = core_lin_to_vox[lcur]
            comp_vox.append(vcur)
            cx, cy, cz = int(ix[vcur]), int(iy[vcur]), int(iz[vcur])
            for dx, dy, dz in offsets:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if nx < 0 or ny < 0 or nz < 0 or nx >= grid or ny >= grid or nz >= grid:
                    continue
                nlin = lcur + dx * stride_x + dy * stride_y + dz
                if nlin in core_lin_set and nlin not in visited:
                    visited.add(nlin)
                    stack.append(nlin)

        if len(comp_vox) < int(min_core_voxels):
            continue
        if int(vox_target[np.array(comp_vox, dtype=np.int64)].sum()) < int(min_core_target_points):
            continue
        instances_core.append(comp_vox)

    # One-step shell expansion (26-neigh): add neighboring voxels with lower ratio, but avoid instance bridging:
    # if a shell voxel touches multiple cores, it is dropped.
    shell_mask = (vox_ratio >= float(shell_ratio)) & (vox_target > 0)
    shell_candidate_set = set(map(int, lin[shell_mask]))
    shell_owner: Dict[int, int] = {}
    conflicts: set[int] = set()

    for inst_id, comp_vox in enumerate(instances_core):
        for vcur in comp_vox:
            lcur = int(lin[vcur])
            cx, cy, cz = int(ix[vcur]), int(iy[vcur]), int(iz[vcur])
            for dx, dy, dz in offsets:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if nx < 0 or ny < 0 or nz < 0 or nx >= grid or ny >= grid or nz >= grid:
                    continue
                nlin = lcur + dx * stride_x + dy * stride_y + dz
                if nlin in core_lin_set:
                    continue
                if nlin not in shell_candidate_set:
                    continue
                prev = shell_owner.get(nlin)
                if prev is None:
                    shell_owner[nlin] = inst_id
                elif prev != inst_id:
                    conflicts.add(nlin)

    for nlin in conflicts:
        shell_owner.pop(nlin, None)

    out_inst: List[np.ndarray] = []
    out_depth: List[np.ndarray] = []
    out_code: List[np.ndarray] = []
    out_total: List[np.ndarray] = []
    out_target: List[np.ndarray] = []
    out_ratio: List[np.ndarray] = []

    shell_lin = np.fromiter(shell_owner.keys(), dtype=np.int64, count=len(shell_owner))
    shell_inst = np.fromiter(shell_owner.values(), dtype=np.int32, count=len(shell_owner))
    if shell_lin.size:
        order_lin = np.argsort(lin)
        lin_sorted = lin[order_lin]
        shell_pos = np.searchsorted(lin_sorted, shell_lin)
        ok = (shell_pos < lin_sorted.size) & (lin_sorted[shell_pos] == shell_lin)
        shell_vox = order_lin[shell_pos[ok]].astype(np.int64, copy=False)
        shell_inst = shell_inst[ok]
    else:
        shell_vox = np.zeros((0,), dtype=np.int64)

    for inst_id, comp_vox in enumerate(instances_core):
        extra_vox = shell_vox[shell_inst == inst_id] if shell_vox.size else np.zeros((0,), dtype=np.int64)
        vox_ids = np.concatenate([np.asarray(comp_vox, dtype=np.int64), extra_vox.astype(np.int64, copy=False)])
        depths_i, codes_i, total_i, target_i, ratio_i = _compress_instance_voxels(
            codes_leaf=vox_codes[vox_ids],
            total_leaf=vox_total[vox_ids],
            target_leaf=vox_target[vox_ids],
            depth=depth,
            ratio_thr=float(target_ratio),
        )
        out_inst.append(np.full((codes_i.size,), inst_id, dtype=np.int32))
        out_depth.append(depths_i)
        out_code.append(codes_i)
        out_total.append(total_i)
        out_target.append(target_i)
        out_ratio.append(ratio_i)

    instance_ids = np.concatenate(out_inst) if out_inst else np.zeros((0,), dtype=np.int32)
    depths_out = np.concatenate(out_depth) if out_depth else np.zeros((0,), dtype=np.uint8)
    codes_out = np.concatenate(out_code) if out_code else np.zeros((0,), dtype=np.uint64)
    total_out = np.concatenate(out_total) if out_total else np.zeros((0,), dtype=np.uint32)
    target_out = np.concatenate(out_target) if out_target else np.zeros((0,), dtype=np.uint32)
    ratio_out = np.concatenate(out_ratio) if out_ratio else np.zeros((0,), dtype=np.float32)

    return InstanceBlocksResult(
        cube_min=cube_min,
        cube_edge=edge,
        target_class=int(target_class),
        threshold=float(target_ratio),
        shell_threshold=float(shell_ratio),
        leaf_depth=int(depth),
        instance_ids=instance_ids,
        depths=depths_out,
        codes=codes_out,
        total_points=total_out,
        target_points=target_out,
        target_ratios=ratio_out,
    )

#输入点云与标签，输出自适应分块+合并后的语义块
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

    # Leaf level: unique over (code, label) pairs and counts.最大深度叶子层，pair打包值，counts计数
    uniq_pair, pair_counts = np.unique(packed, return_counts=True)
    del packed

    leaf_codes = (uniq_pair >> np.uint64(bits_k)).astype(np.uint64, copy=False)
    leaf_labels = (uniq_pair & label_mask).astype(np.uint16, copy=False)
    leaf_counts = pair_counts.astype(np.uint32, copy=False)

    # Build per-depth node stats bottom-up from max depth -> root, without dense (K) arrays.
    level_codes: List[np.ndarray] = [np.zeros((0,), dtype=np.uint64) for _ in range(depth + 1)]
    level_total: List[np.ndarray] = [np.zeros((0,), dtype=np.uint32) for _ in range(depth + 1)]
    level_maj_label: List[np.ndarray] = [np.zeros((0,), dtype=np.uint16) for _ in range(depth + 1)]
    level_maj_count: List[np.ndarray] = [np.zeros((0,), dtype=np.uint32) for _ in range(depth + 1)]

    codes_d = leaf_codes
    labels_d = leaf_labels
    counts_d = leaf_counts
    for d in range(depth, -1, -1):
        node_c, node_tot, node_lab, node_max = _node_stats_from_pairs(codes_d, labels_d, counts_d)
        level_codes[d] = node_c
        level_total[d] = node_tot
        level_maj_label[d] = node_lab
        level_maj_count[d] = node_max
        if d == 0:
            break
        codes_d, labels_d, counts_d = _aggregate_pairs_to_parent(codes_d, labels_d, counts_d, bits_k=bits_k)

    # Top-down adaptive splitting: split nodes until purity>=threshold, else stop at max depth.
    thr = float(purity)
    nodes_by_depth: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    active_codes = np.array([0], dtype=np.uint64)
    forced_impure = 0
    occupied_leaf_voxels = int(level_codes[depth].size)
    kept_leaf_voxels = 0

    for d in range(0, depth + 1):
        if active_codes.size == 0:
            break
        codes_this = level_codes[d]
        idx = np.searchsorted(codes_this, active_codes)
        if np.any(idx >= codes_this.size) or not np.array_equal(codes_this[idx], active_codes):
            raise RuntimeError("Active node codes not found in level stats (unexpected)")

        tot = level_total[d][idx].astype(np.uint32, copy=False)
        maxc = level_maj_count[d][idx].astype(np.uint32, copy=False)
        lab = level_maj_label[d][idx].astype(np.uint16, copy=False)
        pur = (maxc.astype(np.float64) / np.maximum(tot.astype(np.float64), 1.0)).astype(np.float32)

        accept = (pur >= thr) | (d == depth)
        if d == depth:
            forced_impure += int(np.sum(pur < thr))

        if np.any(accept):
            ac = active_codes[accept]
            al = lab[accept]
            ap = tot[accept]
            au = pur[accept]
            prev = nodes_by_depth.get(d)
            if prev is None:
                nodes_by_depth[d] = (ac, al, ap, au)
            else:
                pc, pl, pp, pu = prev
                nodes_by_depth[d] = (
                    np.concatenate([pc, ac]),
                    np.concatenate([pl, al]),
                    np.concatenate([pp, ap]),
                    np.concatenate([pu, au]),
                )

        if d == depth:
            break

        rej = active_codes[~accept]
        if rej.size == 0:
            active_codes = np.zeros((0,), dtype=np.uint64)
            continue

        child_codes = (rej[:, None] << np.uint64(3)) | np.arange(8, dtype=np.uint64)[None, :]
        child_codes = child_codes.reshape(-1)
        # Keep only children that exist (have points).
        codes_next = level_codes[d + 1]
        idx2 = np.searchsorted(codes_next, child_codes)
        ok = idx2 < codes_next.size
        if np.any(ok):
            ok_idx = idx2[ok]
            ok[ok] = codes_next[ok_idx] == child_codes[ok]
        active_codes = child_codes[ok]

    # Leaf count after adaptive splitting (not necessarily at max depth).
    kept_leaf_voxels = int(sum(v[0].size for v in nodes_by_depth.values()))

    # Bottom-up merge siblings with same label, but only if merged parent also satisfies purity threshold.
    for d in range(depth, 0, -1):
        cur = nodes_by_depth.get(d)
        if cur is None:
            continue
        codes_cur, labels_cur, pts_cur, pur_cur = cur
        if codes_cur.size == 0:
            continue

        parent = codes_cur >> np.uint64(3)
        child_bit = (np.uint16(1) << (codes_cur & np.uint64(7)).astype(np.uint16, copy=False)).astype(np.uint16)
        key = (parent.astype(np.uint64) << np.uint64(bits_k)) | labels_cur.astype(np.uint64)
        idxm = np.argsort(key)
        key_s = key[idxm]
        parent_s = parent[idxm]
        labels_s2 = labels_cur[idxm]
        pts_s2 = pts_cur[idxm]
        bits_s2 = child_bit[idxm]

        group_start = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
        group_n = np.diff(np.r_[group_start, key_s.size]).astype(np.int32, copy=False)
        bits_sum = np.add.reduceat(bits_s2.astype(np.uint16), group_start)
        pts_sum = np.add.reduceat(pts_s2.astype(np.uint64), group_start).astype(np.uint32, copy=False)

        can_merge_group = (group_n == 8) & (bits_sum == np.uint16(0xFF))
        if not np.any(can_merge_group):
            nodes_by_depth[d] = (codes_cur, labels_cur, pts_cur, pur_cur)
            continue

        cand_parent_codes = parent_s[group_start][can_merge_group].astype(np.uint64, copy=False)
        cand_labels = labels_s2[group_start][can_merge_group].astype(np.uint16, copy=False)

        # Check parent purity and majority label at depth d-1.
        pcodes = level_codes[d - 1]
        pidx = np.searchsorted(pcodes, cand_parent_codes)
        if np.any(pidx >= pcodes.size) or not np.array_equal(pcodes[pidx], cand_parent_codes):
            raise RuntimeError("Parent codes not found in stats (unexpected)")
        ptot = level_total[d - 1][pidx].astype(np.uint32, copy=False)
        pmax = level_maj_count[d - 1][pidx].astype(np.uint32, copy=False)
        plab = level_maj_label[d - 1][pidx].astype(np.uint16, copy=False)
        ppur = (pmax.astype(np.float64) / np.maximum(ptot.astype(np.float64), 1.0)).astype(np.float32)

        merge_ok = can_merge_group.copy()
        # Tighten: parent must meet purity threshold and match the merged label.
        merge_ok[can_merge_group] = (ppur >= thr) & (plab == cand_labels)

        if not np.any(merge_ok):
            nodes_by_depth[d] = (codes_cur, labels_cur, pts_cur, pur_cur)
            continue

        merged_parent_codes = parent_s[group_start][merge_ok].astype(np.uint64, copy=False)
        merged_labels = labels_s2[group_start][merge_ok].astype(np.uint16, copy=False)
        merged_points = pts_sum[merge_ok].astype(np.uint32, copy=False)

        # Purity for merged nodes from parent stats.
        mpidx = np.searchsorted(pcodes, merged_parent_codes)
        mpur = (
            level_maj_count[d - 1][mpidx].astype(np.float64) / np.maximum(level_total[d - 1][mpidx].astype(np.float64), 1.0)
        ).astype(np.float32)

        # Keep members of groups that were NOT merged.
        keep_members = np.repeat(~merge_ok, group_n)
        kept_codes = codes_cur[idxm][keep_members]
        kept_labels = labels_cur[idxm][keep_members]
        kept_points = pts_cur[idxm][keep_members]
        kept_pur = pur_cur[idxm][keep_members]
        nodes_by_depth[d] = (kept_codes, kept_labels, kept_points, kept_pur)

        prev = nodes_by_depth.get(d - 1)
        if prev is None:
            nodes_by_depth[d - 1] = (merged_parent_codes, merged_labels, merged_points, mpur)
        else:
            pc, pl, pp, pu = prev
            nodes_by_depth[d - 1] = (
                np.concatenate([pc, merged_parent_codes]),
                np.concatenate([pl, merged_labels]),
                np.concatenate([pp, merged_points]),
                np.concatenate([pu, mpur]),
            )

    # Gather final blocks across depths.
    depths_list: List[np.ndarray] = []
    codes_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    pts_list: List[np.ndarray] = []
    pur_list: List[np.ndarray] = []
    for d, (c, l, p, u) in sorted(nodes_by_depth.items()):
        if c.size == 0:
            continue
        depths_list.append(np.full((c.size,), d, dtype=np.uint8))
        codes_list.append(c.astype(np.uint64, copy=False))
        labels_list.append(l.astype(np.uint16, copy=False))
        pts_list.append(p.astype(np.uint32, copy=False))
        pur_list.append(u.astype(np.float32, copy=False))

    depths_out = np.concatenate(depths_list) if depths_list else np.zeros((0,), dtype=np.uint8)
    codes_out = np.concatenate(codes_list) if codes_list else np.zeros((0,), dtype=np.uint64)
    labels_out = np.concatenate(labels_list) if labels_list else np.zeros((0,), dtype=np.uint16)
    pts_out = np.concatenate(pts_list) if pts_list else np.zeros((0,), dtype=np.uint32)
    pur_out = np.concatenate(pur_list) if pur_list else np.zeros((0,), dtype=np.float32)

    return BlocksResult(
        cube_min=cube_min,
        cube_edge=edge,
        class_ids=class_ids_sorted,
        threshold=float(purity),
        leaf_depth=int(depth),
        occupied_leaf_voxels=occupied_leaf_voxels,
        kept_leaf_voxels=kept_leaf_voxels,
        forced_impure_leaves=int(forced_impure),
        depths=depths_out,
        codes=codes_out,
        labels=labels_out,
        n_points=pts_out,
        purities=pur_out,
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


def instance_blocks_to_world(
    res: InstanceBlocksResult,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Returns (center_xyz, min_xyz, max_xyz, edge_len)
    if res.codes.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    ix, iy, iz = morton3d_decode(res.codes)
    d = res.depths.astype(np.int32, copy=False)
    div = (2.0 ** d.astype(np.float64))
    size = (res.cube_edge / div).astype(np.float64, copy=False)

    corner = np.empty((res.codes.size, 3), dtype=np.float64)
    corner[:, 0] = res.cube_min[0] + ix.astype(np.float64) * size
    corner[:, 1] = res.cube_min[1] + iy.astype(np.float64) * size
    corner[:, 2] = res.cube_min[2] + iz.astype(np.float64) * size
    center = corner + (size[:, None] * 0.5)
    max_corner = corner + size[:, None]
    return center, corner, max_corner, size


def instance_blocks_to_bboxes(res: InstanceBlocksResult) -> InstanceBBoxesResult:
    # Union all instance blocks into one AABB per instance (no cube constraint in the final output).
    _, corner, max_corner, _ = instance_blocks_to_world(res)
    if corner.size == 0:
        return InstanceBBoxesResult(
            cube_min=res.cube_min,
            cube_edge=float(res.cube_edge),
            target_class=int(res.target_class),
            threshold=float(res.threshold),
            shell_threshold=float(res.shell_threshold),
            leaf_depth=int(res.leaf_depth),
            instance_ids=np.zeros((0,), dtype=np.int32),
            bbox_min=np.zeros((0, 3), dtype=np.float64),
            bbox_max=np.zeros((0, 3), dtype=np.float64),
            block_count=np.zeros((0,), dtype=np.uint32),
            total_points=np.zeros((0,), dtype=np.uint64),
            target_points=np.zeros((0,), dtype=np.uint64),
            target_ratios=np.zeros((0,), dtype=np.float32),
        )

    inst = res.instance_ids.astype(np.int32, copy=False)
    order = np.argsort(inst, kind="mergesort")
    inst_s = inst[order]
    corner_s = corner[order]
    max_s = max_corner[order]
    total_s = res.total_points[order].astype(np.uint64, copy=False)
    target_s = res.target_points[order].astype(np.uint64, copy=False)

    start = np.flatnonzero(np.r_[True, inst_s[1:] != inst_s[:-1]])
    uniq_inst = inst_s[start].astype(np.int32, copy=False)
    block_count = np.diff(np.r_[start, inst_s.size]).astype(np.uint32, copy=False)

    bbox_min = np.minimum.reduceat(corner_s, start, axis=0).astype(np.float64, copy=False)
    bbox_max = np.maximum.reduceat(max_s, start, axis=0).astype(np.float64, copy=False)
    total_sum = np.add.reduceat(total_s, start).astype(np.uint64, copy=False)
    target_sum = np.add.reduceat(target_s, start).astype(np.uint64, copy=False)
    ratio = (target_sum.astype(np.float64) / np.maximum(total_sum.astype(np.float64), 1.0)).astype(np.float32)

    return InstanceBBoxesResult(
        cube_min=res.cube_min,
        cube_edge=float(res.cube_edge),
        target_class=int(res.target_class),
        threshold=float(res.threshold),
        shell_threshold=float(res.shell_threshold),
        leaf_depth=int(res.leaf_depth),
        instance_ids=uniq_inst,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        block_count=block_count,
        total_points=total_sum,
        target_points=target_sum,
        target_ratios=ratio,
    )


def write_blocks_jsonl(res: BlocksResult, path: Path) -> None:
    center, corner, max_corner, size, class_raw = blocks_to_world(res)
    with path.open("w", encoding="utf-8") as f:
        for i in range(res.codes.size):
            rec = {
                "depth": int(res.depths[i]),
                "morton": int(res.codes[i]),
                "class": int(class_raw[i]),
                "n_points": int(res.n_points[i]),
                "purity": float(res.purities[i]),
                "is_pure": bool(res.purities[i] >= res.threshold),
                "cube_min": [float(corner[i, 0]), float(corner[i, 1]), float(corner[i, 2])],
                "cube_max": [float(max_corner[i, 0]), float(max_corner[i, 1]), float(max_corner[i, 2])],
                "cube_edge": float(size[i]),
                "center": [float(center[i, 0]), float(center[i, 1]), float(center[i, 2])],
                "leaf_purity_threshold": float(res.threshold),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_instance_blocks_jsonl(res: InstanceBlocksResult, path: Path) -> None:
    center, corner, max_corner, size = instance_blocks_to_world(res)
    with path.open("w", encoding="utf-8") as f:
        for i in range(res.codes.size):
            rec = {
                "instance_id": int(res.instance_ids[i]),
                "target_class": int(res.target_class),
                "depth": int(res.depths[i]),
                "morton": int(res.codes[i]),
                "total_points": int(res.total_points[i]),
                "target_points": int(res.target_points[i]),
                "target_ratio": float(res.target_ratios[i]),
                "ratio_threshold": float(res.threshold),
                "shell_ratio_threshold": float(res.shell_threshold),
                "cube_min": [float(corner[i, 0]), float(corner[i, 1]), float(corner[i, 2])],
                "cube_max": [float(max_corner[i, 0]), float(max_corner[i, 1]), float(max_corner[i, 2])],
                "cube_edge": float(size[i]),
                "center": [float(center[i, 0]), float(center[i, 1]), float(center[i, 2])],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_instance_bboxes_jsonl(res: InstanceBBoxesResult, path: Path) -> None:
    center = (res.bbox_min + res.bbox_max) * 0.5
    size = res.bbox_max - res.bbox_min
    with path.open("w", encoding="utf-8") as f:
        for i in range(res.instance_ids.size):
            rec = {
                "instance_id": int(res.instance_ids[i]),
                "target_class": int(res.target_class),
                "block_count": int(res.block_count[i]),
                "total_points": int(res.total_points[i]),
                "target_points": int(res.target_points[i]),
                "target_ratio": float(res.target_ratios[i]),
                "ratio_threshold": float(res.threshold),
                "shell_ratio_threshold": float(res.shell_threshold),
                "bbox_min": [float(res.bbox_min[i, 0]), float(res.bbox_min[i, 1]), float(res.bbox_min[i, 2])],
                "bbox_max": [float(res.bbox_max[i, 0]), float(res.bbox_max[i, 1]), float(res.bbox_max[i, 2])],
                "bbox_size": [float(size[i, 0]), float(size[i, 1]), float(size[i, 2])],
                "center": [float(center[i, 0]), float(center[i, 1]), float(center[i, 2])],
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
            ("purity", "f4"),
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
    data["purity"] = res.purities.astype(np.float32, copy=False)

    PlyData([PlyElement.describe(data, "vertex")], text=True).write(str(path))


def write_instance_blocks_centers_ply(res: InstanceBlocksResult, path: Path) -> None:
    center, _, _, size = instance_blocks_to_world(res)
    if center.shape[0] == 0:
        data = np.empty((0,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        PlyData([PlyElement.describe(data, "vertex")], text=True).write(str(path))
        return

    inst = res.instance_ids.astype(np.int64, copy=False)
    u, inv = np.unique(inst, return_inverse=True)
    rng = np.random.default_rng(0)
    colors = (rng.random((u.size, 3)) * 0.85 + 0.15).astype(np.float32)
    rgb = colors[inv]

    data = np.empty(
        (center.shape[0],),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("instance_id", "i4"),
            ("target_class", "i4"),
            ("depth", "u1"),
            ("cube_edge", "f4"),
            ("total_points", "u4"),
            ("target_points", "u4"),
            ("target_ratio", "f4"),
        ],
    )
    data["x"] = center[:, 0].astype(np.float32)
    data["y"] = center[:, 1].astype(np.float32)
    data["z"] = center[:, 2].astype(np.float32)
    data["red"] = np.clip(rgb[:, 0] * 255.0, 0, 255).astype(np.uint8)
    data["green"] = np.clip(rgb[:, 1] * 255.0, 0, 255).astype(np.uint8)
    data["blue"] = np.clip(rgb[:, 2] * 255.0, 0, 255).astype(np.uint8)
    data["instance_id"] = res.instance_ids.astype(np.int32, copy=False)
    data["target_class"] = np.full((center.shape[0],), int(res.target_class), dtype=np.int32)
    data["depth"] = res.depths.astype(np.uint8, copy=False)
    data["cube_edge"] = size.astype(np.float32, copy=False)
    data["total_points"] = res.total_points.astype(np.uint32, copy=False)
    data["target_points"] = res.target_points.astype(np.uint32, copy=False)
    data["target_ratio"] = res.target_ratios.astype(np.float32, copy=False)

    PlyData([PlyElement.describe(data, "vertex")], text=True).write(str(path))


def write_instance_bboxes_centers_ply(res: InstanceBBoxesResult, path: Path) -> None:
    if res.instance_ids.size == 0:
        data = np.empty(
            (0,),
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("instance_id", "i4"),
                ("target_class", "i4"),
                ("bbox_dx", "f4"),
                ("bbox_dy", "f4"),
                ("bbox_dz", "f4"),
                ("block_count", "u4"),
                # plyfile doesn't support uint64 ("u8"), so we store counts as uint32 (clipped if needed).
                ("total_points", "u4"),
                ("target_points", "u4"),
                ("target_ratio", "f4"),
            ],
        )
        PlyData([PlyElement.describe(data, "vertex")], text=True).write(str(path))
        return

    center = (res.bbox_min + res.bbox_max) * 0.5
    size = res.bbox_max - res.bbox_min
    rgb = _tab20_colors(res.instance_ids.astype(np.int64, copy=False))

    data = np.empty(
        (res.instance_ids.size,),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("instance_id", "i4"),
            ("target_class", "i4"),
            ("bbox_dx", "f4"),
            ("bbox_dy", "f4"),
            ("bbox_dz", "f4"),
            ("block_count", "u4"),
            # plyfile doesn't support uint64 ("u8"), so we store counts as uint32 (clipped if needed).
            ("total_points", "u4"),
            ("target_points", "u4"),
            ("target_ratio", "f4"),
        ],
    )
    data["x"] = center[:, 0].astype(np.float32)
    data["y"] = center[:, 1].astype(np.float32)
    data["z"] = center[:, 2].astype(np.float32)
    data["red"] = np.clip(rgb[:, 0] * 255.0, 0, 255).astype(np.uint8)
    data["green"] = np.clip(rgb[:, 1] * 255.0, 0, 255).astype(np.uint8)
    data["blue"] = np.clip(rgb[:, 2] * 255.0, 0, 255).astype(np.uint8)
    data["instance_id"] = res.instance_ids.astype(np.int32, copy=False)
    data["target_class"] = np.full((res.instance_ids.size,), int(res.target_class), dtype=np.int32)
    data["bbox_dx"] = size[:, 0].astype(np.float32, copy=False)
    data["bbox_dy"] = size[:, 1].astype(np.float32, copy=False)
    data["bbox_dz"] = size[:, 2].astype(np.float32, copy=False)
    data["block_count"] = res.block_count.astype(np.uint32, copy=False)
    u32max = np.uint64(np.iinfo(np.uint32).max)
    data["total_points"] = np.minimum(res.total_points.astype(np.uint64, copy=False), u32max).astype(np.uint32)
    data["target_points"] = np.minimum(res.target_points.astype(np.uint64, copy=False), u32max).astype(np.uint32)
    data["target_ratio"] = res.target_ratios.astype(np.float32, copy=False)

    PlyData([PlyElement.describe(data, "vertex")], text=True).write(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply", type=str, help="Input labeled PLY (must contain x/y/z and class)")
    ap.add_argument("--purity", type=float, default=0.90, help="Semantic mode: leaf purity threshold, default 0.90")
    ap.add_argument("--target_class", type=int, default=None, help="Instance mode: raw class id to extract instances for")
    ap.add_argument("--target_ratio", type=float, default=0.90, help="Instance mode: target ratio threshold")
    ap.add_argument("--shell_ratio", type=float, default=0.60, help="Instance mode: 1-step shell expansion ratio threshold")
    ap.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26], help="Instance mode: connectivity for core CCs")
    ap.add_argument("--min_core_voxels", type=int, default=10, help="Instance mode: min core voxels per instance")
    ap.add_argument("--min_core_target_points", type=int, default=30, help="Instance mode: min target points per instance core")
    ap.add_argument("--list_classes", action="store_true", help="Print unique class ids in the PLY and exit")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--depth", type=int, default=None, help="Max octree depth D (0..21)")
    g.add_argument("--leaf_size", type=float, default=0.20, help="Minimum leaf cube edge length (meters)")
    ap.add_argument("--chunk_points", type=int, default=2_000_000, help="Chunk size for morton encoding")
    ap.add_argument("--mmap", type=str, default="r", help="plyfile mmap mode: r/c/False (default: r)")
    ap.add_argument("--max_points", type=int, default=None, help="Optional random subsample for quick runs")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for --max_points")
    ap.add_argument("--out_jsonl", type=str, default=None, help="Write merged semantic blocks to JSONL")
    ap.add_argument("--out_ply", type=str, default=None, help="Write merged block centers to a colored PLY")
    ap.add_argument("--out_bbox_jsonl", type=str, default=None, help="Instance mode: write one AABB per instance to JSONL")
    ap.add_argument("--out_bbox_ply", type=str, default=None, help="Instance mode: write per-instance bbox centers to PLY")
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
    if args.list_classes:
        uniq = np.unique(cls_raw)
        print(f"[INFO] Unique classes ({uniq.size}): {uniq.tolist()}")
        return
    cube_min, edge = root_cube_from_xyz(x, y, z)

    if args.depth is None:
        depth = depth_from_leaf_size(edge, float(args.leaf_size), max_depth=21)
    else:
        depth = int(args.depth)
    print(f"[INFO] Root cube edge={edge:.3f} m, cube_min={cube_min.tolist()}, max depth={depth}")

    if args.target_class is None:
        print(f"[INFO] Mode=semantic threshold={args.purity:.2f}")
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
            f"[INFO] Occupied voxels at max depth={res.occupied_leaf_voxels:,}, adaptive leaves={res.kept_leaf_voxels:,}, forced-impure-at-max-depth={res.forced_impure_leaves:,}"
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
    else:
        print(
            f"[INFO] Mode=instance target_class={int(args.target_class)} target_ratio={args.target_ratio:.2f} "
            f"shell_ratio={args.shell_ratio:.2f} connectivity={args.connectivity}"
        )
        resi = build_target_instance_blocks(
            x,
            y,
            z,
            cls_raw,
            depth=depth,
            target_class=int(args.target_class),
            target_ratio=float(args.target_ratio),
            shell_ratio=float(args.shell_ratio),
            connectivity=int(args.connectivity),
            min_core_voxels=int(args.min_core_voxels),
            min_core_target_points=int(args.min_core_target_points),
            chunk_points=int(args.chunk_points),
        )
        n_inst = int(resi.instance_ids.max() + 1) if resi.instance_ids.size else 0
        print(f"[INFO] Instances={n_inst} blocks={resi.codes.size:,}")
        if args.out_jsonl:
            out = Path(args.out_jsonl)
            write_instance_blocks_jsonl(resi, out)
            print(f"[DONE] Wrote: {out}")
        if args.out_ply:
            out = Path(args.out_ply)
            write_instance_blocks_centers_ply(resi, out)
            print(f"[DONE] Wrote: {out}")
        if args.out_bbox_jsonl or args.out_bbox_ply:
            resb = instance_blocks_to_bboxes(resi)
            if args.out_bbox_jsonl:
                out = Path(args.out_bbox_jsonl)
                write_instance_bboxes_jsonl(resb, out)
                print(f"[DONE] Wrote: {out}")
            if args.out_bbox_ply:
                out = Path(args.out_bbox_ply)
                write_instance_bboxes_centers_ply(resb, out)
                print(f"[DONE] Wrote: {out}")


if __name__ == "__main__":
    main()
