import pickle
import numpy as np
import mujoco

MJCF_PATH = "/home/ubuntu/GMR/assets/unitree_g1/g1_mocap_29dof.xml"
SRC = "/home/ubuntu/UniMuMo/outputs/g1_motion.pkl"
DST = "/home/ubuntu/UniMuMo/outputs/g1_motion_grounded.pkl"

# 给脚底留一点点安全间隙，避免看起来还在擦地
CLEARANCE = 0.0
EXTRA_SINK = 0.005

# 优先用这些 body 做“接地参考”
FOOT_BODY_NAMES = [
    "left_toe_link",
    "right_toe_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

def xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)

with open(SRC, "rb") as f:
    motion = pickle.load(f)

root_pos = np.asarray(motion["root_pos"], dtype=np.float64).copy()
root_rot = np.asarray(motion["root_rot"], dtype=np.float64).copy()
dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64).copy()

model = mujoco.MjModel.from_xml_path(MJCF_PATH)
data = mujoco.MjData(model)

body_ids = []
for name in FOOT_BODY_NAMES:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid == -1:
        print(f"[WARN] body not found: {name}")
    else:
        body_ids.append((name, bid))

if not body_ids:
    raise RuntimeError("No valid foot bodies found in XML.")

T = root_pos.shape[0]
frame_min_z = []

for t in range(T):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    # floating base
    data.qpos[0:3] = root_pos[t]
    data.qpos[3:7] = xyzw_to_wxyz(root_rot[t])

    # joints
    nq_rest = min(len(data.qpos) - 7, dof_pos.shape[1])
    data.qpos[7:7+nq_rest] = dof_pos[t, :nq_rest]

    mujoco.mj_forward(model, data)

    z_vals = []
    for name, bid in body_ids:
        z_vals.append(data.xpos[bid, 2])

    frame_min_z.append(min(z_vals))

frame_min_z = np.asarray(frame_min_z, dtype=np.float64)

global_min_z = frame_min_z.min()
offset = -global_min_z + CLEARANCE - EXTRA_SINK

print("frame_min_z first/last:", frame_min_z[0], frame_min_z[-1])
print("frame_min_z min/max   :", frame_min_z.min(), frame_min_z.max())
print("apply z offset        :", offset)

motion["root_pos"] = root_pos.copy()
motion["root_pos"][:, 2] += offset

with open(DST, "wb") as f:
    pickle.dump(motion, f)

print("saved to:", DST)