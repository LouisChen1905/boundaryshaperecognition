"""Replay boundary frames in global XY and camera XZ coordinates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

from boundary_shape import orient_shape, recognize


LOG_PATTERN = re.compile(
    r"^\[(?P<clock>[^]]+)\].*?boundary_xz:\s*\[(?P<points>.*?)\]"
    r"\s*,\s*location info:\s*(?P<pose>[^\r\n]+)$"
)
POINT_PATTERN = re.compile(r"\(\s*([^,()]+)\s*,\s*([^,()]+)\s*\)")
CAMERA_TRANSLATION = np.array([0.4, 0.0, 0.0])
CAMERA_RPY = (-1.57, 0.0, 0)  # roll, pitch, yaw (radians)


@dataclass
class Frame:
    clock: str
    timestamp: float
    robot_xyz: np.ndarray
    robot_yaw: float
    camera_xz: np.ndarray
    world_points: np.ndarray


def rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Standard right-handed, extrinsic XYZ Euler angles: Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def parse_frame(line: str, line_number: int) -> Frame:
    match = LOG_PATTERN.match(line.strip())
    if not match:
        raise ValueError(f"Line {line_number}: unrecognized frame format")

    pose = [float(value.strip()) for value in match.group("pose").split(",")]
    if len(pose) != 7:
        raise ValueError(f"Line {line_number}: expected 7 location values, got {len(pose)}")
    pairs = POINT_PATTERN.findall(match.group("points"))
    if not pairs and match.group("points").strip():
        raise ValueError(f"Line {line_number}: could not parse boundary points")

    camera_points = np.zeros((len(pairs), 3))
    if pairs:
        camera_points[:, (0, 2)] = np.asarray(pairs, dtype=float)
    robot_xyz = np.asarray(pose[:3])
    world_points = (camera_points @ rotation(*CAMERA_RPY).T + CAMERA_TRANSLATION)
    world_points = world_points @ rotation(*pose[3:6]).T + robot_xyz
    return Frame(match.group("clock"), pose[6], robot_xyz, pose[5],
                 camera_points[:, (0, 2)], world_points)


def load_frames(path: Path) -> list[Frame]:
    with path.open("r", encoding="utf-8") as source:
        frames = [parse_frame(line, number) for number, line in enumerate(source, 1)]
    if not frames:
        raise ValueError("The log is empty")
    return frames


def replay(frames: list[Frame], speed: float) -> None:
    fig, (ax, camera_ax) = plt.subplots(1, 2, figsize=(15, 7))
    fig.subplots_adjust(bottom=0.22, wspace=0.28)
    all_xy = np.vstack([f.world_points[:, :2] for f in frames if len(f.world_points)] +
                       [np.vstack([f.robot_xyz[:2] for f in frames])])
    low, high = all_xy.min(axis=0), all_xy.max(axis=0)
    center = (low + high) / 2
    half_range = max((high - low).max() / 2, 0.5) + 0.3
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Global X (m)")
    ax.set_ylabel("Global Y (m)")
    ax.grid(True, alpha=0.3)
    points_artist = ax.scatter([], [], s=12, c="tab:green", label="Boundary points")
    robot_artist = ax.scatter([], [], s=65, marker="x", c="tab:red", label="Robot")
    heading_artist, = ax.plot([], [], c="tab:red", lw=2)
    trend_artist, = ax.plot([], [], c="tab:blue", lw=2, label="Estimated shape")
    ax.legend(loc="upper right")

    camera_xz = np.vstack([f.camera_xz for f in frames if len(f.camera_xz)])
    camera_low, camera_high = camera_xz.min(axis=0), camera_xz.max(axis=0)
    camera_padding = np.maximum((camera_high - camera_low) * 0.05, 0.1)
    camera_ax.set_xlim(camera_low[0] - camera_padding[0], camera_high[0] + camera_padding[0])
    camera_ax.set_ylim(camera_low[1] - camera_padding[1], camera_high[1] + camera_padding[1])
    camera_ax.set_aspect("equal", adjustable="box")
    camera_ax.set_xlabel("Camera X (m)")
    camera_ax.set_ylabel("Camera Z (m)")
    camera_ax.grid(True, alpha=0.3)
    camera_artist = camera_ax.scatter([], [], s=12, c="tab:green", label="Boundary points")
    camera_trend_artist, = camera_ax.plot([], [], c="tab:blue", lw=2,
                                          label="Estimated shape")
    camera_ax.legend(loc="upper right")

    slider_ax = fig.add_axes((0.13, 0.12, 0.74, 0.03))
    slider = Slider(slider_ax, "Frame", 1, len(frames), valinit=1, valstep=1)
    previous = Button(fig.add_axes((0.18, 0.04, 0.15, 0.05)), "Previous")
    play_button = Button(fig.add_axes((0.42, 0.04, 0.15, 0.05)), "Play")
    next_button = Button(fig.add_axes((0.66, 0.04, 0.15, 0.05)), "Next")
    state = {"playing": False}

    def show(index: int) -> None:
        frame = frames[index]
        points_artist.set_offsets(frame.world_points[:, :2])
        camera_artist.set_offsets(frame.camera_xz)
        camera_shape = orient_shape(recognize(frame.camera_xz), (0.0, 0.0))
        if camera_shape is None:
            camera_trend_artist.set_data([], [])
            camera_detail = "insufficient shape data"
        else:
            camera_trend_artist.set_data(camera_shape.centerline[:, 0],
                                         camera_shape.centerline[:, 1])
            camera_detail = (f"{camera_shape.label} | length {camera_shape.length_m:.2f} m | "
                             f"turn {camera_shape.turn_deg:+.0f} deg | "
                             f"direction {camera_shape.directions_deg[0]:.0f} deg")
        camera_ax.set_title(f"Camera XZ | {len(frame.camera_xz)} points\n{camera_detail}")
        robot_artist.set_offsets(frame.robot_xyz[:2].reshape(1, 2))
        origin = frame.robot_xyz[:2]
        # The pose yaw refers to the robot's lateral axis; its forward axis is 90° counterclockwise.
        heading = frame.robot_yaw + np.pi / 2
        tip = origin + 0.25 * np.array([np.cos(heading), np.sin(heading)])
        heading_artist.set_data([origin[0], tip[0]], [origin[1], tip[1]])
        shape = orient_shape(recognize(frame.world_points), origin)
        if shape is None:
            trend_artist.set_data([], [])
            detail = "insufficient shape data"
        else:
            trend_artist.set_data(shape.centerline[:, 0], shape.centerline[:, 1])
            detail = (f"{shape.label} | length {shape.length_m:.2f} m | "
                      f"turn {shape.turn_deg:+.0f} deg | "
                      f"direction {shape.directions_deg[0]:.0f} deg")
        ax.set_title(f"Frame {index + 1}/{len(frames)} | {frame.clock} | "
                     f"{len(frame.world_points)} points\n{detail}")
        fig.canvas.draw_idle()

    slider.on_changed(lambda value: show(int(value) - 1))
    previous.on_clicked(lambda event: slider.set_val(max(1, int(slider.val) - 1)))
    next_button.on_clicked(lambda event: slider.set_val(min(len(frames), int(slider.val) + 1)))

    timer = fig.canvas.new_timer(interval=100)

    def tick() -> None:
        if not state["playing"]:
            return
        index = int(slider.val) - 1
        if index >= len(frames) - 1:
            state["playing"] = False
            play_button.label.set_text("Play")
            fig.canvas.draw_idle()
            return
        slider.set_val(index + 2)
        delay = (frames[index + 1].timestamp - frames[index].timestamp) * 1000 / speed
        timer.interval = max(20, int(delay)) if delay > 0 else 200
        timer.start()

    timer.add_callback(tick)

    def toggle(event) -> None:
        state["playing"] = not state["playing"]
        play_button.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"]:
            if int(slider.val) == len(frames):
                slider.set_val(1)
            timer.start()
        else:
            timer.stop()
        fig.canvas.draw_idle()

    play_button.on_clicked(toggle)

    def on_key(event) -> None:
        if event.key == "left":
            slider.set_val(max(1, int(slider.val) - 1))
        elif event.key == "right":
            slider.set_val(min(len(frames), int(slider.val) + 1))
        elif event.key == " ":
            toggle(event)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show(0)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path,
                        default=Path(__file__).with_name("vp_boundary_20260910下午.log"))
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    replay(load_frames(args.log), args.speed)


if __name__ == "__main__":
    main()
