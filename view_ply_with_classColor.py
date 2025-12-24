import sys, numpy as np, open3d as o3d
from pathlib import Path
from plyfile import PlyData

# ---- 输入路径 ----
PLY_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else r"data\training_10_classes\Lille1_1.ply")
assert PLY_PATH.exists(), f"找不到文件：{PLY_PATH}"

# ---- 用 plyfile 读取字段 ----
ply = PlyData.read(str(PLY_PATH))
v = ply["vertex"].data
names = set(v.dtype.names)
print("[INFO] vertex 字段：", v.dtype.names)

def pick_name(cands):
    for k in cands:
        if k in names:
            return k
    return None

# 兼容大小写/不同命名：坐标
kx = pick_name(("x","X","x_coord"))
ky = pick_name(("y","Y","y_coord"))
kz = pick_name(("z","Z","z_coord"))
assert kx and ky and kz, "此 PLY 不含可识别的坐标字段（x/y/z 或 X/Y/Z）"

xyz = np.vstack([v[kx], v[ky], v[kz]]).T.astype(np.float32)

# ------------------------------------------------------------------
# 1) 尝试读取语义 class 字段
# ------------------------------------------------------------------
kcls = pick_name(("class", "Class", "label", "semantic", "sem_class"))
class_colors = None

if kcls:
    class_raw = np.asarray(v[kcls])
    uniq = np.unique(class_raw)
    print(f"[INFO] 发现语义字段 {kcls}，共有 {len(uniq)} 个不同类别")

    # 建一个 class_id -> 索引 的映射，方便送进 colormap
    id2idx = {cid: i for i, cid in enumerate(uniq)}
    idx = np.vectorize(id2idx.get, otypes=[np.int32])(class_raw)

    # 用 matplotlib 的分类调色板上色（tab20 比较适合语义）
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab20", len(uniq))
        class_colors = cmap(idx)[:, :3]  # 去掉 alpha 通道
    except ImportError:
        # 没有 matplotlib 就用随机颜色
        rng = np.random.default_rng(0)
        lut = rng.random((len(uniq), 3))
        class_colors = lut[idx]

# ------------------------------------------------------------------
# 2) 尝试 RGB（若数据集没有 RGB，会走强度或统一色）
# ------------------------------------------------------------------
rgb = None
kr, kg, kb = [pick_name(c) for c in (("red","r","Red"),
                                      ("green","g","Green"),
                                      ("blue","b","Blue"))]
if kr and kg and kb:
    rgb = np.vstack([v[kr], v[kg], v[kb]]).T.astype(np.float32) / 255.0

# ------------------------------------------------------------------
# 3) 尝试强度字段（reflectance/intensity/remission）
# ------------------------------------------------------------------
I = None
ki = pick_name(("intensity","Intensity","reflectance","Reflectance",
                "remission","Remission"))
if ki:
    Iraw = np.asarray(v[ki], dtype=np.float32)
    lo, hi = np.percentile(Iraw, [1, 99])
    I = np.clip((Iraw - lo) / max(hi - lo, 1e-6), 0, 1)

print(f"[INFO] 读取成功：点数 {xyz.shape[0]:,}  hasClass={kcls is not None}  hasRGB={rgb is not None}  hasI={I is not None}")

# ---- 构建 Open3D 点云 ----
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

title = "Paris-Lille-3D"
if class_colors is not None:
    pcd.colors = o3d.utility.Vector3dVector(class_colors.astype(np.float64))
    title += f" (Semantic classes: {len(np.unique(class_raw))})"
elif rgb is not None:
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    title += " (RGB)"
elif I is not None:
    gray = np.stack([I, I, I], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(gray)
    title += " (Intensity grayscale)"
else:
    pcd.paint_uniform_color([0.7, 0.7, 0.9])
    title += " (No color/intensity)"

# ---- 大数据防护：体素下采样 ----
MAX_POINTS = 15_000_000
VOXEL_SIZE = 0.10
if xyz.shape[0] > MAX_POINTS:
    print(f"[INFO] 点数>{MAX_POINTS:,}，体素下采样 {VOXEL_SIZE} m …")
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    print(f"[INFO] 下采样后点数：{np.asarray(pcd.points).shape[0]:,}")

# ---- 显示 ----
o3d.visualization.draw_geometries(
    [pcd],
    window_name=title,
    width=1280,
    height=800
)