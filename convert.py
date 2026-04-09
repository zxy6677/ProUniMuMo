import numpy as np
from scipy.spatial.transform import Rotation as R

SRC = "/home/ubuntu/UniMuMo/outputs/motion_gen_smplx.npz"
DST = "/home/ubuntu/UniMuMo/outputs/motion_gen_smplx_for_gmr.npz"
FPS = 30.0

# ===== 坐标修正参数 =====
# 先从这组开始试：
#   X_DEG = 90
# 如果还是“头朝下竖着”，改成 -90
# 如果站起来了但朝向反了，再把 Z_YAW_DEG 改成 180
X_DEG = 90
Z_YAW_DEG = 0

# 是否把同一个基变换应用到 trans
# 对你当前这条“XRMoCap 直接输出 SMPL-X”的链路，先设 True 试
# 如果后面出现“越走越上浮/轨迹怪”，再改成 False，只做 root_orient 修正 + grounding
APPLY_BASIS_TO_TRANS = True

# 是否把最低点抬到 z=0
GROUND_TO_ZERO = True


data = np.load(SRC, allow_pickle=True)
print("src keys:", data.files)

# ===== fullpose =====
fullpose = np.asarray(data["fullpose"], dtype=np.float32)

# 兼容两种常见形状：
# 1) (T, 55, 3)
# 2) (T, 165)
if fullpose.ndim == 3:
    T, J, C = fullpose.shape
    assert C == 3, f"unexpected fullpose shape: {fullpose.shape}"
elif fullpose.ndim == 2:
    T, D = fullpose.shape
    assert D % 3 == 0, f"unexpected fullpose shape: {fullpose.shape}"
    J = D // 3
    fullpose = fullpose.reshape(T, J, 3)
else:
    raise ValueError(f"unexpected fullpose ndim: {fullpose.ndim}, shape={fullpose.shape}")

print("fullpose reshaped:", fullpose.shape)

# ===== SMPL-X fullpose 约定 =====
# [0]          global_orient
# [1:22]       body_pose (21 joints)
# [22]         jaw
# [23]         leye
# [24]         reye
# [25:40]      left_hand (15 joints)
# [40:55]      right_hand (15 joints)

if fullpose.shape[1] < 22:
    raise ValueError(f"fullpose joint count too small: {fullpose.shape}")

root_orient = fullpose[:, 0, :]                   # (T, 3), axis-angle
pose_body = fullpose[:, 1:22, :].reshape(T, -1)  # (T, 63)

# ===== transl -> trans =====
if "transl" not in data.files:
    raise KeyError("transl is not in the archive")
trans = np.asarray(data["transl"], dtype=np.float32).reshape(T, 3)

# ===== 坐标轴修正 =====
# y-up -> z-up 的固定基变换
basis = R.from_euler("z", Z_YAW_DEG, degrees=True) * R.from_euler("x", X_DEG, degrees=True)

# 1) 修 root_orient
root_rot = R.from_rotvec(root_orient)      # old world rotation
root_rot_new = basis * root_rot            # new world rotation
root_orient = root_rot_new.as_rotvec().astype(np.float32)

# 2) 修 trans（可开关）
if APPLY_BASIS_TO_TRANS:
    trans = basis.apply(trans).astype(np.float32)

# 3) grounding
if GROUND_TO_ZERO:
    trans[:, 2] -= trans[:, 2].min()

# ===== betas =====
# GMR 这里不要逐帧 betas，而是单个 shape 向量
# 并 pad 到 16 维
if "betas" in data.files:
    betas_raw = np.asarray(data["betas"], dtype=np.float32)

    if betas_raw.ndim == 2:
        betas = betas_raw[0]
    elif betas_raw.ndim == 1:
        betas = betas_raw
    else:
        raise ValueError(f"unexpected betas shape: {betas_raw.shape}")
else:
    betas = np.zeros((10,), dtype=np.float32)

if betas.shape[0] < 16:
    betas = np.pad(betas, (0, 16 - betas.shape[0]))
elif betas.shape[0] > 16:
    betas = betas[:16]

betas = betas.astype(np.float32)

# ===== gender =====
if "gender" in data.files:
    gender_raw = data["gender"]
    if isinstance(gender_raw, np.ndarray):
        if gender_raw.shape == ():
            gender = str(gender_raw.item())
        else:
            gender = str(gender_raw.reshape(-1)[0])
    else:
        gender = str(gender_raw)
else:
    gender = "neutral"

gender = gender.lower()
if gender not in ["male", "female", "neutral"]:
    gender = "neutral"

# ===== save =====
np.savez(
    DST,
    pose_body=pose_body.astype(np.float32),
    root_orient=root_orient.astype(np.float32),
    trans=trans.astype(np.float32),
    betas=betas.astype(np.float32),
    gender=np.array(gender),
    mocap_frame_rate=np.array([FPS], dtype=np.float32),
)

print("saved to:", DST)
print("pose_body:", pose_body.shape)
print("root_orient:", root_orient.shape)
print("trans:", trans.shape)
print("betas:", betas.shape)
print("gender:", gender)
print("mocap_frame_rate:", FPS)
print("trans z min/max:", trans[:, 2].min(), trans[:, 2].max())