import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from unimumo.motion.motion_process import recover_from_ric


T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]


def load_motion_as_joints(path: str) -> np.ndarray:
    arr = np.load(path)

    if arr.ndim == 3 and arr.shape[1:] == (22, 3):
        joints = arr.astype(np.float32)
        return joints

    if arr.ndim == 2 and arr.shape[1] == 263:
        joints = recover_from_ric(
            torch.from_numpy(arr).float().unsqueeze(0),
            joints_num=22,
        ).squeeze(0).cpu().numpy().astype(np.float32)
        return joints

    raise ValueError(
        f"Unsupported input shape {arr.shape}. "
        f"Expected [T,22,3] or [T,263]."
    )


def maybe_swap_axes(joints: np.ndarray, axis_mode: str) -> np.ndarray:
    """
    axis_mode:
      - y_up: keep as-is
      - z_up: swap y/z for visualization
    """
    joints = joints.copy()
    if axis_mode == "y_up":
        return joints
    if axis_mode == "z_up":
        joints = joints[..., [0, 2, 1]]
        return joints
    raise ValueError(axis_mode)


def auto_floor(joints: np.ndarray, up_axis: int) -> np.ndarray:
    joints = joints.copy()
    joints[..., up_axis] -= joints[..., up_axis].min()
    return joints


def compute_bounds(joints: np.ndarray):
    pts = joints.reshape(-1, 3)
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    center = 0.5 * (pmin + pmax)
    radius = max(float((pmax - pmin).max()) * 0.6, 1.0)
    return center, radius, pmin, pmax


def draw_ground(ax, center, radius, up_axis: int):
    # Draw a square ground plane at coordinate 0 along the up axis.
    grid = np.linspace(-radius, radius, 2)
    uu, vv = np.meshgrid(grid, grid)

    if up_axis == 1:  # y-up
        xx = center[0] + uu
        yy = np.zeros_like(xx)
        zz = center[2] + vv
    elif up_axis == 2:  # z-up
        xx = center[0] + uu
        yy = center[1] + vv
        zz = np.zeros_like(xx)
    else:
        raise ValueError(up_axis)

    ax.plot_surface(xx, yy, zz, color="#dddddd", alpha=0.25, shade=False)


def render_sequence(
    joints: np.ndarray,
    output_path: str,
    fps: int = 60,
    width: int = 960,
    height: int = 960,
    axis_mode: str = "y_up",
    draw_traj: bool = True,
):
    joints = maybe_swap_axes(joints, axis_mode=axis_mode)
    up_axis = 1 if axis_mode == "y_up" else 2
    joints = auto_floor(joints, up_axis=up_axis)

    center, radius, _, _ = compute_bounds(joints)
    root_traj = joints[:, 0].copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figsize = (width / 100.0, height / 100.0)
    fig = plt.figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    frames = []

    for t in tqdm(range(len(joints)), desc="Rendering joints"):
        ax.cla()

        # Ground
        draw_ground(ax, center=center, radius=radius, up_axis=up_axis)

        # Skeleton
        pose = joints[t]
        colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]

        for chain, color in zip(T2M_KINEMATIC_CHAIN, colors):
            pts = pose[np.array(chain)]
            ax.plot(
                pts[:, 0], pts[:, 1], pts[:, 2],
                linewidth=3.0,
                color=color,
            )

        # Joints
        ax.scatter(
            pose[:, 0], pose[:, 1], pose[:, 2],
            s=20,
            c="black",
            depthshade=False,
        )

        # Root trajectory
        if draw_traj and t > 1:
            traj = root_traj[: t + 1]
            ax.plot(
                traj[:, 0], traj[:, 1], traj[:, 2],
                color="red",
                linewidth=1.5,
                alpha=0.7,
            )

        # Fixed bounds
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))

        # View
        if axis_mode == "y_up":
            ax.view_init(elev=15, azim=-75)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        else:
            ax.view_init(elev=20, azim=-75)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

        ax.set_title(f"frame {t}")
        ax.grid(False)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frame = buf[..., :3].copy()
        frames.append(frame)

    plt.close(fig)

    if output_path.suffix.lower() == ".gif":
        imageio.mimsave(str(output_path), frames, fps=fps)
    else:
        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec="libx264",
            format="FFMPEG",
        )
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()

    print(f"[OK] wrote: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="[T,22,3] or [T,263] npy")
    parser.add_argument("--output", type=str, required=True, help=".mp4 or .gif")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--axis_mode", type=str, default="y_up", choices=["y_up", "z_up"])
    parser.add_argument("--no_traj", action="store_true")
    args = parser.parse_args()

    joints = load_motion_as_joints(args.input)
    render_sequence(
        joints=joints,
        output_path=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        axis_mode=args.axis_mode,
        draw_traj=not args.no_traj,
    )


if __name__ == "__main__":
    main()