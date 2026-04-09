import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# These imports exist in the UniMuMo repo.
from unimumo.motion.common.skeleton import Skeleton
from unimumo.motion.common.quaternion import (
    qbetween_np,
    qfix,
    qinv,
    qinv_np,
    qmul_np,
    qrot,
    qrot_np,
    quaternion_to_cont6d_np,
)
from unimumo.motion.motion_process import recover_from_ric


# -----------------------------
# HumanML / UniMuMo 22-joint setup
# -----------------------------
T2M_RAW_OFFSETS = np.array([
    [0, 0, 0],
    [1, 0, 0], [-1, 0, 0], [0, 1, 0],
    [0, -1, 0], [0, -1, 0], [0, 1, 0],
    [0, -1, 0], [0, -1, 0], [0, 1, 0],
    [0, 0, 1], [0, 0, 1], [0, 1, 0],
    [1, 0, 0], [-1, 0, 0], [0, 0, 1],
    [0, -1, 0], [0, -1, 0], [0, -1, 0],
    [0, -1, 0], [0, -1, 0], [0, -1, 0],
], dtype=np.float32)

T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

# HumanML processing constants, matching the public HumanML / UniMoCap pipeline.
LEG_IDX_1, LEG_IDX_2 = 5, 8
FID_R, FID_L = [8, 11], [7, 10]
FACE_JOINT_INDX = [2, 1, 17, 16]  # right hip, left hip, right shoulder, left shoulder


# -----------------------------
# FineDance / SMPL-X setup
# -----------------------------
# First 22 joints are the body skeleton; extra 3 face joints and 30 hand joints follow.
# Order is taken from FineDance official vis.py.
SMPLX_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
    'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw',
    'left_eye_smplhf', 'right_eye_smplhf',
    'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3',
    'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3',
    'left_thumb1', 'left_thumb2', 'left_thumb3',
    'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3',
    'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3',
    'right_thumb1', 'right_thumb2', 'right_thumb3'
]

SMPLX_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53
]

BODY22_NAMES = SMPLX_JOINT_NAMES[:22]


# -----------------------------
# Math helpers
# -----------------------------
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to axis-angle with a torch-only implementation."""
    # matrix: [..., 3, 3]
    m = matrix
    cos_theta = ((m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]) - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)

    rx = m[..., 2, 1] - m[..., 1, 2]
    ry = m[..., 0, 2] - m[..., 2, 0]
    rz = m[..., 1, 0] - m[..., 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1)

    sin_theta = torch.sin(theta)
    small = sin_theta.abs() < 1e-6
    scale = torch.empty_like(theta)
    scale[~small] = 0.5 / sin_theta[~small]
    scale[small] = 0.5

    axis = axis * scale.unsqueeze(-1)
    axis = F.normalize(axis, dim=-1)
    axis_angle = axis * theta.unsqueeze(-1)
    axis_angle[small] = 0.0
    return axis_angle


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula. axis_angle: [..., 3] -> [..., 3, 3]"""
    orig_shape = axis_angle.shape[:-1]
    x = axis_angle.reshape(-1, 3)
    angle = torch.norm(x, dim=1, keepdim=True).clamp_min(1e-8)
    axis = x / angle
    ax, ay, az = axis[:, 0], axis[:, 1], axis[:, 2]
    c = torch.cos(angle[:, 0])
    s = torch.sin(angle[:, 0])
    one_c = 1.0 - c

    R = torch.zeros((x.shape[0], 3, 3), dtype=x.dtype, device=x.device)
    R[:, 0, 0] = c + ax * ax * one_c
    R[:, 0, 1] = ax * ay * one_c - az * s
    R[:, 0, 2] = ax * az * one_c + ay * s
    R[:, 1, 0] = ay * ax * one_c + az * s
    R[:, 1, 1] = c + ay * ay * one_c
    R[:, 1, 2] = ay * az * one_c - ax * s
    R[:, 2, 0] = az * ax * one_c - ay * s
    R[:, 2, 1] = az * ay * one_c + ax * s
    R[:, 2, 2] = c + az * az * one_c
    return R.reshape(*orig_shape, 3, 3)


# -----------------------------
# FineDance -> 55 joints FK
# -----------------------------
def add_face_zeros_to_165(axis_angle_156: torch.Tensor) -> torch.Tensor:
    """FineDance official vis.py inserts 9 zeros after the first 66 dims."""
    assert axis_angle_156.ndim == 2 and axis_angle_156.shape[1] == 156
    zeros = torch.zeros((axis_angle_156.shape[0], 9), dtype=axis_angle_156.dtype, device=axis_angle_156.device)
    return torch.cat([axis_angle_156[:, :66], zeros, axis_angle_156[:, 66:]], dim=1)


def batch_rigid_transform(rot_mats: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """Minimal version of FineDance vis.py batch_rigid_transform."""
    # rot_mats: [T, J, 3, 3], joints: [T, J, 3], parents: [J]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    T, J = joints.shape[:2]
    transforms_mat = transform_mat(rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).reshape(T, J, 4, 4)
    transform_chain = [transforms_mat[:, 0]]
    for j in range(1, J):
        curr = torch.matmul(transform_chain[parents[j]], transforms_mat[:, j])
        transform_chain.append(curr)
    transforms = torch.stack(transform_chain, dim=1)
    return transforms[:, :, :3, 3]


def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def finedance_315_to_body22_joints(
    motion_315: np.ndarray,
    smplx_rest_joints: np.ndarray,
    source_fps: int,
    target_fps: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Convert FineDance raw motion [T, 315] = [trans(3), 52*6 rot6d] to body-22 joint positions [T', 22, 3].
    """
    assert motion_315.ndim == 2 and motion_315.shape[1] == 315, motion_315.shape

    trans = torch.from_numpy(motion_315[:, :3]).float().to(device)
    rot6d = torch.from_numpy(motion_315[:, 3:]).float().to(device).view(-1, 52, 6)

    rot_mats = rotation_6d_to_matrix(rot6d)
    axis_angle_156 = matrix_to_axis_angle(rot_mats).reshape(rot6d.shape[0], 156)
    axis_angle_165 = add_face_zeros_to_165(axis_angle_156).view(rot6d.shape[0], 55, 3)

    if source_fps != target_fps:
        # Resample trans and axis-angle linearly before FK for simplicity / stability.
        # This is okay for MVP conversion; if you want perfect SO(3) interpolation later, replace with SLERP.
        new_len = int(round(axis_angle_165.shape[0] / source_fps * target_fps))
        axis_angle_165 = F.interpolate(
            axis_angle_165.permute(1, 2, 0).reshape(1, 55 * 3, -1),
            size=new_len,
            mode="linear",
            align_corners=False,
        ).reshape(55, 3, new_len).permute(2, 0, 1).contiguous()
        trans = F.interpolate(
            trans.T.unsqueeze(0),
            size=new_len,
            mode="linear",
            align_corners=False,
        ).squeeze(0).T.contiguous()

    J = torch.from_numpy(smplx_rest_joints).float().to(device)
    if J.ndim == 2:
        J = J.unsqueeze(0)
    J = J.expand(axis_angle_165.shape[0], -1, -1).contiguous()  # [T, 55, 3]

    parents = torch.tensor(SMPLX_PARENTS, dtype=torch.long, device=device)
    rot_mats_55 = axis_angle_to_matrix(axis_angle_165)
    joints55 = batch_rigid_transform(rot_mats_55, J, parents) + trans.unsqueeze(1)

    joints22 = joints55[:, :22, :].detach().cpu().numpy().astype(np.float32)
    return joints22


# -----------------------------
# HumanML / UniMuMo 263-d encoder
# -----------------------------
def uniform_skeleton(positions: np.ndarray, target_offset: torch.Tensor) -> np.ndarray:
    src_skel = Skeleton(torch.from_numpy(T2M_RAW_OFFSETS), T2M_KINEMATIC_CHAIN, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    src_leg_len = np.abs(src_offset[LEG_IDX_1]).max() + np.abs(src_offset[LEG_IDX_2]).max()
    tgt_leg_len = np.abs(tgt_offset[LEG_IDX_1]).max() + np.abs(tgt_offset[LEG_IDX_2]).max()
    scale_rt = tgt_leg_len / max(src_leg_len, 1e-8)

    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    quat_params = src_skel.inverse_kinematics_np(positions, FACE_JOINT_INDX)
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def process_body22_to_vec263(positions: np.ndarray, feet_thre: float, tgt_offsets: torch.Tensor) -> np.ndarray:
    positions = uniform_skeleton(positions, tgt_offsets)

    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1], dtype=np.float32)
    positions = positions - root_pose_init_xz

    r_hip, l_hip, sdr_r, sdr_l = FACE_JOINT_INDX
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]], dtype=np.float32), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]], dtype=np.float32)
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,), dtype=np.float32) * root_quat_init
    positions = qrot_np(root_quat_init, positions)
    global_positions = positions.copy()

    def foot_detect(pos: np.ndarray, thres: float) -> Tuple[np.ndarray, np.ndarray]:
        velfactor = np.array([thres, thres], dtype=np.float32)
        feet_l_x = (pos[1:, FID_L, 0] - pos[:-1, FID_L, 0]) ** 2
        feet_l_y = (pos[1:, FID_L, 1] - pos[:-1, FID_L, 1]) ** 2
        feet_l_z = (pos[1:, FID_L, 2] - pos[:-1, FID_L, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (pos[1:, FID_R, 0] - pos[:-1, FID_R, 0]) ** 2
        feet_r_y = (pos[1:, FID_R, 1] - pos[:-1, FID_R, 1]) ** 2
        feet_r_z = (pos[1:, FID_R, 2] - pos[:-1, FID_R, 2]) ** 2
        feet_r = ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    def get_rifke(pos: np.ndarray, r_rot: np.ndarray) -> np.ndarray:
        pos = pos.copy()
        pos[..., 0] -= pos[:, 0:1, 0]
        pos[..., 2] -= pos[:, 0:1, 2]
        pos = qrot_np(np.repeat(r_rot[:, None], pos.shape[1], axis=1), pos)
        return pos

    def get_cont6d_params(pos: np.ndarray):
        skel = Skeleton(torch.from_numpy(T2M_RAW_OFFSETS), T2M_KINEMATIC_CHAIN, 'cpu')
        quat_params = skel.inverse_kinematics_np(pos, FACE_JOINT_INDX, smooth_forward=True)
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()
        velocity = (pos[1:, 0] - pos[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity_quat, velocity, r_rot = get_cont6d_params(positions)
    positions_local = get_rifke(positions, r_rot)

    root_y = positions_local[:, 0, 1:2]
    r_velocity = np.arcsin(r_velocity_quat[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
    ric_data = positions_local[:, 1:].reshape(len(positions_local), -1)

    local_vel = qrot_np(
        np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)
    return data.astype(np.float32)


# -----------------------------
# Driver
# -----------------------------
def gather_motion_files(motion_dir: Path) -> List[Path]:
    return sorted([p for p in motion_dir.glob("*.npy") if p.is_file()])


def maybe_make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_tgt_offsets(example_joints22: np.ndarray) -> torch.Tensor:
    tgt_skel = Skeleton(torch.from_numpy(T2M_RAW_OFFSETS), T2M_KINEMATIC_CHAIN, 'cpu')
    return tgt_skel.get_offsets_joints(torch.from_numpy(example_joints22[0]).float())


def compute_mean_std(vec_files: List[Path], out_dir: Path):
    all_frames = []
    lengths = {}
    for path in tqdm(vec_files, desc="Compute mean/std"):
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != 263:
            raise ValueError(f"Expected [T,263], got {arr.shape} for {path}")
        all_frames.append(arr)
        lengths[path.stem] = int(arr.shape[0])
    cat = np.concatenate(all_frames, axis=0)
    mean = cat.mean(axis=0).astype(np.float32)
    std = cat.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    np.save(out_dir / "Mean.npy", mean)
    np.save(out_dir / "Std.npy", std)
    with open(out_dir / "lengths.json", "w", encoding="utf-8") as f:
        json.dump(lengths, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_dir", type=str, required=True, help="FineDance raw motion dir, e.g. FINEDANCE/motion")
    parser.add_argument("--out_dir", type=str, required=True, help="Output root dir")
    parser.add_argument("--smplx_rest_joints", type=str, required=True, help="Path to FineDance smplx_neu_J_1.npy")
    parser.add_argument("--source_fps", type=int, default=30, help="Raw FineDance motion fps. Keep explicit to avoid hard-coding mistakes.")
    parser.add_argument("--target_fps", type=int, default=60, choices=[20, 30, 60], help="Target fps for converted 22-joint positions before 263 encoding.")
    parser.add_argument("--feet_thre", type=float, default=0.002)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_joints22", action="store_true", help="Also save intermediate [T,22,3] joints for debugging")
    parser.add_argument("--save_reports", action="store_true", help="Save per-file reconstruction error reports")
    parser.add_argument("--compute_mean_std", action="store_true", help="Compute Mean.npy and Std.npy over converted vec263 files")
    args = parser.parse_args()

    motion_dir = Path(args.motion_dir)
    out_dir = Path(args.out_dir)
    vec_dir = out_dir / f"motion_vec263_fps{args.target_fps}"
    joints22_dir = out_dir / f"motion_joints22_fps{args.target_fps}"
    report_dir = out_dir / "conversion_reports"

    maybe_make_dir(vec_dir)
    if args.save_joints22:
        maybe_make_dir(joints22_dir)
    if args.save_reports:
        maybe_make_dir(report_dir)

    smplx_rest_joints = np.load(args.smplx_rest_joints)
    if smplx_rest_joints.shape != (55, 3):
        raise ValueError(f"Expected smplx_neu_J_1.npy shape [55,3], got {smplx_rest_joints.shape}")

    files = gather_motion_files(motion_dir)
    if not files:
        raise FileNotFoundError(f"No .npy files found under {motion_dir}")

    # Build target offsets from the first successfully converted sample.
    tgt_offsets = None
    converted = []

    for path in tqdm(files, desc="Convert FineDance -> UniMuMo vec263"):
        raw = np.load(path)
        if raw.ndim != 2 or raw.shape[1] != 315:
            print(f"[skip] {path.name}: expected [T,315], got {raw.shape}")
            continue

        joints22 = finedance_315_to_body22_joints(
            raw,
            smplx_rest_joints=smplx_rest_joints,
            source_fps=args.source_fps,
            target_fps=args.target_fps,
            device=args.device,
        )

        if tgt_offsets is None:
            tgt_offsets = load_tgt_offsets(joints22)

        vec263 = process_body22_to_vec263(joints22, feet_thre=args.feet_thre, tgt_offsets=tgt_offsets)
        np.save(vec_dir / path.name, vec263)
        converted.append(vec_dir / path.name)

        if args.save_joints22:
            np.save(joints22_dir / path.name, joints22.astype(np.float32))

        if args.save_reports:
            # HumanML representation is T-1 long. We validate decode consistency on overlapping frames.
            recon = recover_from_ric(torch.from_numpy(vec263).unsqueeze(0), joints_num=22).squeeze(0).cpu().numpy()
            gt = joints22[:-1]
            mpjpe = float(np.mean(np.linalg.norm(recon - gt, axis=-1)))
            report = {
                "file": path.name,
                "raw_shape": list(raw.shape),
                "joints22_shape": list(joints22.shape),
                "vec263_shape": list(vec263.shape),
                "recon_mpjpe_mean": mpjpe,
                "note": "vec263 is T-1 long by construction because root velocity / local velocity use frame differences.",
            }
            with open(report_dir / f"{path.stem}.json", "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

    if not converted:
        raise RuntimeError("No files were successfully converted.")

    if args.compute_mean_std:
        compute_mean_std(converted, out_dir)

    summary = {
        "num_converted": len(converted),
        "output_vec_dir": str(vec_dir),
        "output_joints22_dir": str(joints22_dir) if args.save_joints22 else None,
        "output_report_dir": str(report_dir) if args.save_reports else None,
        "mean_std_written": bool(args.compute_mean_std),
        "body22_joint_order": BODY22_NAMES,
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
    }
    with open(out_dir / "conversion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
