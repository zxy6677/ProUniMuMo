import numpy as np
import matplotlib.pyplot as plt

PATH = "/home/ubuntu/UniMuMo/outputs/motion_gen.npy"

arr = np.load(PATH, allow_pickle=True)

print("===== BASIC INFO =====")
print("type:", type(arr))
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("ndim:", arr.ndim)

if not (isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 3):
    raise ValueError("Expected ndarray of shape (T, J, 3)")

T, J, C = arr.shape
print("frames T:", T)
print("joints J:", J)
print("coord dim C:", C)

flat = arr.reshape(-1, 3)
print("\n===== GLOBAL STATS =====")
print("global min:", float(arr.min()))
print("global max:", float(arr.max()))
print("per-axis min [x,y,z]:", flat.min(axis=0))
print("per-axis max [x,y,z]:", flat.max(axis=0))
print("per-axis mean[x,y,z]:", flat.mean(axis=0))
print("per-axis std [x,y,z]:", flat.std(axis=0))

print("\n===== ROOT CANDIDATE CHECK =====")
# 先把每个 joint 的平均速度 / 位移量打出来，帮助判断哪个像 root
vel = np.diff(arr, axis=0)                     # (T-1, J, 3)
speed = np.linalg.norm(vel, axis=-1)          # (T-1, J)
mean_speed = speed.mean(axis=0)               # (J,)

for j in range(J):
    joint_xyz = arr[:, j, :]
    drift = joint_xyz[-1] - joint_xyz[0]
    print(f"joint {j:02d}: "
          f"first={joint_xyz[0]}, "
          f"last={joint_xyz[-1]}, "
          f"drift={drift}, "
          f"mean_speed={mean_speed[j]:.6f}")

print("\n===== FIRST FRAME JOINTS =====")
for j in range(J):
    print(f"joint {j:02d}: {arr[0, j]}")

print("\n===== LAST FRAME JOINTS =====")
for j in range(J):
    print(f"joint {j:02d}: {arr[-1, j]}")

print("\n===== BODY SIZE HEURISTICS =====")
# 看第一帧里哪些点最靠近 joint 0，帮助猜骨架
ref = arr[0, 0]  # 先拿 joint 0 做参考
dists = np.linalg.norm(arr[0] - ref[None, :], axis=1)
order = np.argsort(dists)
print("distance to joint 0 in frame 0:")
for idx in order:
    print(f"joint {idx:02d}: dist={dists[idx]:.6f}")

print("\n===== PER-JOINT HEIGHT RANGE =====")
# 假设第 1 维可能是 up 轴，先打印每个点第1维范围
for j in range(J):
    y = arr[:, j, 1]
    print(f"joint {j:02d}: axis1 min={y.min():.6f}, max={y.max():.6f}, mean={y.mean():.6f}")

# ===== 保存第一帧和最后一帧的关节编号图 =====
def save_joint_plot(points, out_path, title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=40)

    for i, p in enumerate(points):
        ax.text(p[0], p[1], p[2], str(i), fontsize=10)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)

    # 保持比例尽量接近
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    scale = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - scale, center[0] + scale)
    ax.set_ylim(center[1] - scale, center[1] + scale)
    ax.set_zlim(center[2] - scale, center[2] + scale)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

save_joint_plot(arr[0], "frame0_joints.png", "Frame 0 Joints")
save_joint_plot(arr[-1], "frame_last_joints.png", "Last Frame Joints")

print("\nSaved plots:")
print("  frame0_joints.png")
print("  frame_last_joints.png")