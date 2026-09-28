"""Replay boundary frames in global XY and camera XZ coordinates."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button, Slider
import numpy as np

from boundary_shape import (grass_inside_support, grass_reference_coverage,
                            orient_shape,
                            orient_tracking_reference, recognize, reference_side,
                            tracking_reference)


LOG_PATTERN = re.compile(
    r"^\[(?P<clock>[^]]+)\].*?boundary_xz:\s*\[(?P<points>.*?)\]"
    r"\s*(?:,\s*grass_xz:\s*\[(?P<grass>.*?)\])?"
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
    robot_rpy: np.ndarray
    robot_yaw: float
    camera_xz: np.ndarray
    world_points: np.ndarray
    grass_world_xy: Optional[np.ndarray] = None


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
    frame = Frame(match.group("clock"), pose[6], robot_xyz, np.asarray(pose[3:6]), pose[5],
                  camera_points[:, (0, 2)], world_points)
    if match.group("grass") is not None:
        grass_pairs = POINT_PATTERN.findall(match.group("grass"))
        if not grass_pairs and match.group("grass").strip():
            raise ValueError(f"Line {line_number}: could not parse grass points")
        grass_xz = np.asarray(grass_pairs, dtype=float).reshape(-1, 2)
        frame.grass_world_xy = camera_xz_to_global_xy(grass_xz, frame)
    return frame


def global_xy_to_camera_xz(world_xy: np.ndarray, frame: Frame) -> np.ndarray:
    """Map global XY positions to the camera's measured XZ plane (camera Y=0)."""
    robot_rotation = rotation(*frame.robot_rpy)
    camera_to_world = robot_rotation @ rotation(*CAMERA_RPY)
    world_origin = frame.robot_xyz + robot_rotation @ CAMERA_TRANSLATION
    xy_basis = camera_to_world[:2, (0, 2)]
    return np.linalg.solve(xy_basis, (world_xy - world_origin[:2]).T).T


def camera_xz_to_global_xy(camera_xz: np.ndarray, frame: Frame) -> np.ndarray:
    """Map camera XZ grass points to global XY using the frame pose."""
    robot_rotation = rotation(*frame.robot_rpy)
    camera_to_world = robot_rotation @ rotation(*CAMERA_RPY)
    world_origin = frame.robot_xyz + robot_rotation @ CAMERA_TRANSLATION
    xy_basis = camera_to_world[:2, (0, 2)]
    return camera_xz @ xy_basis.T + world_origin[:2]


def load_frames(path: Path) -> list[Frame]:
    if path.suffix.lower() == ".jsonl":
        return load_data_package(path)
    with path.open("r", encoding="utf-8") as source:
        frames = [parse_frame(line, number) for number, line in enumerate(source, 1)]
    if not frames:
        raise ValueError("The log is empty")
    return frames


def load_data_package(path: Path) -> list[Frame]:
    """Read JSONL records and pair visual samples with the latest pose."""
    frames = []
    pose = None
    with path.open("r", encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
                topic = record.get("topic")
                data = record.get("data", {})
                if topic == "location":
                    position = data["poseOI"]["position"]
                    attitude = data["poseOI"]["attitude"]
                    pose = (np.array([position["east"], position["north"], position["up"]], dtype=float),
                            np.array([attitude["roll"], attitude["pitch"], attitude["head"]], dtype=float))
                elif topic == "visual_boundary_points" and pose is not None:
                    def points(key, count_key):
                        count = int(data[count_key])
                        values = data[key]
                        if count < 0 or count > len(values):
                            raise ValueError(f"{count_key} exceeds available points")
                        return np.array([[p["cornerX"], p["cornerZ"]]
                                         for p in values[:count]], dtype=float).reshape(-1, 2)

                    boundary = points("vBoundaryPoints", "boundary_num")
                    grass = points("vGrassPoints", "grass_num")
                    xyz, rpy = pose
                    camera_points = np.zeros((len(boundary), 3))
                    camera_points[:, (0, 2)] = boundary
                    world = (camera_points @ rotation(*CAMERA_RPY).T + CAMERA_TRANSLATION)
                    world = world @ rotation(*rpy).T + xyz
                    timestamp = float(data.get("timestamp", record["system_ms"])) / 1000
                    clock = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")[:-3]
                    frame = Frame(clock, timestamp, xyz.copy(), rpy.copy(), rpy[2], boundary, world)
                    frame.grass_world_xy = camera_xz_to_global_xy(grass, frame)
                    frames.append(frame)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                raise ValueError(f"Line {number}: invalid navigation data: {error}") from error
    if not frames:
        raise ValueError("No visual boundary frames with location in the data package")
    return frames


def load_grass_csv(path: Path, frames: list[Frame]) -> None:
    """Attach grass points using frame,x,y or frame,camera_x,camera_z CSV columns."""
    points = [[] for _ in frames]
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or [])
        world_format = {"frame", "x", "y"} <= fields
        camera_format = {"frame", "camera_x", "camera_z"} <= fields
        if world_format == camera_format:
            raise ValueError("Grass CSV needs either frame,x,y or frame,camera_x,camera_z columns")
        for row_number, row in enumerate(reader, 2):
            frame_number = int(row["frame"])
            if not 1 <= frame_number <= len(frames):
                raise ValueError(f"Grass CSV row {row_number}: frame out of range")
            if world_format:
                point = np.array([float(row["x"]), float(row["y"])])
            else:
                camera_point = np.array([[float(row["camera_x"]), float(row["camera_z"])]])
                point = camera_xz_to_global_xy(camera_point, frames[frame_number - 1])[0]
            if not np.isfinite(point).all():
                raise ValueError(f"Grass CSV row {row_number}: nonfinite coordinate")
            points[frame_number - 1].append(point)
    for frame, cloud in zip(frames, points):
        frame.grass_world_xy = np.asarray(cloud, dtype=float).reshape(-1, 2) if cloud else None


def fit_world_shapes(frames: list[Frame], min_turn_radius=None):
    """Fit each moving frame and pool stable, stationary stretches of frames."""
    max_curvature = 1.5 if min_turn_radius is None else min(1.5, 1 / (min_turn_radius + 0.4))
    shapes = [orient_shape(recognize(frame.world_points, max_curvature=max_curvature),
                           frame.robot_xyz[:2])
              for frame in frames]
    index = 0
    while index < len(frames):
        end = index + 1
        anchor = frames[index]
        while (end < len(frames)
               and np.linalg.norm(frames[end].robot_xyz[:2] - anchor.robot_xyz[:2]) <= 0.03
               and abs((frames[end].robot_yaw - anchor.robot_yaw + np.pi) % (2*np.pi) - np.pi) <= 0.03):
            end += 1
        if end - index >= 10:
            clouds = [frame.world_points for frame in frames[index:end]
                      if len(frame.world_points)]
            pooled = recognize(np.vstack(clouds), max_curvature=max_curvature) if clouds else None
            if pooled is not None:
                for j in range(index, end):
                    shapes[j] = orient_shape(pooled, frames[j].robot_xyz[:2])
        index = end
    return shapes


def recorded_loop_direction(frames: list[Frame]):
    """Summarize a nearly closed recorded pose loop, or return unknown."""
    if len(frames) < 3:
        return "Recorded route: insufficient poses"
    positions = np.array([frame.robot_xyz[:2] for frame in frames])
    length = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    closure_gap = np.linalg.norm(positions[-1] - positions[0])
    if length < 1 or closure_gap > 0.1 * length:
        return "Recorded route: open or incomplete"
    yaw = np.unwrap([frame.robot_yaw for frame in frames])
    turn_deg = np.degrees(yaw[-1] - yaw[0])
    twice_area = np.sum(positions[:, 0] * np.roll(positions[:, 1], -1)
                        - np.roll(positions[:, 0], -1) * positions[:, 1])
    if abs(turn_deg) < 270 or turn_deg * twice_area <= 0:
        return "Recorded route: rotation inconclusive"
    direction = "CCW" if turn_deg > 0 else "CW"
    return f"Recorded route: {direction} | pose yaw {turn_deg:+.0f} deg | closure gap {closure_gap:.2f} m"


def reference_route_direction(frames: list[Frame], shapes):
    """Estimate traversal sense from the sequence of reference midpoints."""
    midpoints = []
    for frame, shape in zip(frames, shapes):
        if shape is None:
            continue
        reference = orient_tracking_reference(
            tracking_reference(shape, frame.robot_xyz[:2],
                               grass_points=frame.grass_world_xy), shape.centerline)
        midpoints.append(reference[len(reference) // 2])
    if len(midpoints) < 3:
        return "Reference route: insufficient samples"
    positions = np.asarray(midpoints)
    length = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    gap = np.linalg.norm(positions[-1] - positions[0])
    if length < 1 or gap > 0.1 * length:
        return "Reference route: open or incomplete"
    twice_area = np.sum(positions[:, 0] * np.roll(positions[:, 1], -1)
                        - np.roll(positions[:, 0], -1) * positions[:, 1])
    bounds = np.ptp(positions, axis=0)
    if abs(twice_area) < 0.02 * bounds[0] * bounds[1]:
        return "Reference route: rotation inconclusive"
    direction = "CCW" if twice_area > 0 else "CW"
    return f"Reference route: {direction} (sampled)"


def replay(frames: list[Frame], speed: float, min_turn_radius=None,
           robot_width=None, right_half_width=None,
           grass_support_radius=0.25) -> None:
    world_shapes = fit_world_shapes(frames, min_turn_radius)
    fig, (ax, camera_ax) = plt.subplots(1, 2, figsize=(15, 7))
    fig.subplots_adjust(bottom=0.22, top=0.78, wspace=0.28)
    fig.suptitle(recorded_loop_direction(frames) + "\n" +
                 reference_route_direction(frames, world_shapes), fontsize=10)
    all_xy = np.vstack([f.world_points[:, :2] for f in frames if len(f.world_points)] +
                       [f.grass_world_xy for f in frames
                        if f.grass_world_xy is not None and len(f.grass_world_xy)] +
                       [np.vstack([f.robot_xyz[:2] for f in frames])])
    low, high = all_xy.min(axis=0), all_xy.max(axis=0)
    center = (low + high) / 2
    half_range = max((high - low).max() / 2, 0.5) + 0.7
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Global X (m)")
    ax.set_ylabel("Global Y (m)")
    ax.grid(True, alpha=0.3)
    points_artist = ax.scatter([], [], s=12, c="tab:green", label="Boundary points")
    has_grass = any(f.grass_world_xy is not None and len(f.grass_world_xy)
                    for f in frames)
    grass_artist = ax.scatter([], [], s=5, c="#a6d854", alpha=0.35,
                              label="Grass points" if has_grass else "_nolegend_")
    robot_artist = ax.scatter([], [], s=65, marker="x", c="tab:red", label="Robot")
    heading_artist, = ax.plot([], [], c="tab:red", lw=2)
    trend_artist, = ax.plot([], [], c="tab:blue", lw=2, label="Estimated shape")
    reference_artist, = ax.plot([], [], c="tab:purple", lw=2.5, ls="--",
                                label="Tracking reference (0.4 m)")
    direction_color = "#ff5a00"
    direction_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                                      mutation_scale=22, color=direction_color, lw=3,
                                      zorder=5)
    ax.add_patch(direction_arrow)
    direction_arrow.set_visible(False)
    direction_text = ax.text(0.02, 0.04, "", transform=ax.transAxes, va="bottom",
                             color=direction_color, weight="bold", fontsize=10,
                             bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    status_text = ax.text(0.02, 0.11, "", transform=ax.transAxes, va="bottom",
                          color="black", fontsize=9,
                          bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    ax.legend(loc="upper right")

    camera_xz = np.vstack([f.camera_xz for f in frames if len(f.camera_xz)] +
                          [global_xy_to_camera_xz(f.grass_world_xy, f)
                           for f in frames if f.grass_world_xy is not None
                           and len(f.grass_world_xy)])
    camera_low, camera_high = camera_xz.min(axis=0), camera_xz.max(axis=0)
    camera_padding = np.maximum((camera_high - camera_low) * 0.05, 0.5)
    camera_ax.set_xlim(camera_low[0] - camera_padding[0], camera_high[0] + camera_padding[0])
    camera_ax.set_ylim(camera_low[1] - camera_padding[1], camera_high[1] + camera_padding[1])
    camera_ax.set_aspect("equal", adjustable="box")
    camera_ax.set_xlabel("Camera X (m)")
    camera_ax.set_ylabel("Camera Z (m)")
    camera_ax.grid(True, alpha=0.3)
    camera_artist = camera_ax.scatter([], [], s=12, c="tab:green", label="Boundary points")
    camera_grass_artist = camera_ax.scatter([], [], s=5, c="#a6d854", alpha=0.35,
                                            label="Grass points" if has_grass else "_nolegend_")
    camera_trend_artist, = camera_ax.plot([], [], c="tab:blue", lw=2,
                                          label="Estimated shape")
    camera_reference_artist, = camera_ax.plot([], [], c="tab:purple", lw=2.5,
                                               ls="--", label="Tracking reference (0.4 m)")
    camera_direction_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                                             mutation_scale=22, color=direction_color,
                                             lw=3, zorder=5)
    camera_ax.add_patch(camera_direction_arrow)
    camera_direction_arrow.set_visible(False)
    camera_direction_text = camera_ax.text(0.02, 0.04, "", transform=camera_ax.transAxes,
                                           va="bottom", color=direction_color,
                                           weight="bold", fontsize=10,
                                           bbox={"facecolor": "white", "alpha": 0.8,
                                                 "edgecolor": "none"})
    camera_ax.legend(loc="upper right")

    slider_ax = fig.add_axes((0.13, 0.12, 0.74, 0.03))
    slider = Slider(slider_ax, "Frame", 1, len(frames), valinit=1, valstep=1)
    previous = Button(fig.add_axes((0.18, 0.04, 0.15, 0.05)), "Previous")
    play_button = Button(fig.add_axes((0.42, 0.04, 0.15, 0.05)), "Play")
    next_button = Button(fig.add_axes((0.66, 0.04, 0.15, 0.05)), "Next")
    state = {"playing": False}

    def show_direction(line, arrow):
        if line is None:
            arrow.set_visible(False)
            return None
        middle = len(line) // 2
        direction = line[middle + 1] - line[middle - 1]
        direction /= np.linalg.norm(direction)
        position = line[middle]
        arrow.set_positions(position - 0.1 * direction,
                            position + 0.1 * direction)
        arrow.set_visible(True)
        return np.degrees(np.arctan2(direction[1], direction[0]))

    def show(index: int) -> None:
        frame = frames[index]
        points_artist.set_offsets(frame.world_points[:, :2])
        camera_artist.set_offsets(frame.camera_xz)
        grass = frame.grass_world_xy
        grass_artist.set_offsets(grass if grass is not None else np.empty((0, 2)))
        camera_grass_artist.set_offsets(
            global_xy_to_camera_xz(grass, frame) if grass is not None
            else np.empty((0, 2)))
        camera_shape = orient_shape(recognize(frame.camera_xz), (0.0, 0.0))
        if camera_shape is None:
            camera_trend_artist.set_data([], [])
            camera_detail = "insufficient shape data"
        else:
            camera_trend_artist.set_data(camera_shape.centerline[:, 0],
                                         camera_shape.centerline[:, 1])
            camera_detail = (f"{camera_shape.label} | length {camera_shape.length_m:.2f} m | "
                             f"turn {camera_shape.turn_deg:+.0f} deg")
        camera_ax.set_title(f"Camera XZ | {len(frame.camera_xz)} points\n{camera_detail}")
        robot_artist.set_offsets(frame.robot_xyz[:2].reshape(1, 2))
        origin = frame.robot_xyz[:2]
        # The pose yaw refers to the robot's lateral axis; its forward axis is 90° counterclockwise.
        heading = frame.robot_yaw + np.pi / 2
        tip = origin + 0.25 * np.array([np.cos(heading), np.sin(heading)])
        heading_artist.set_data([origin[0], tip[0]], [origin[1], tip[1]])
        shape = world_shapes[index]
        if shape is None:
            trend_artist.set_data([], [])
            reference_artist.set_data([], [])
            camera_reference_artist.set_data([], [])
            reference = None
            camera_reference = None
            detail = "insufficient shape data"
            status_text.set_text("")
        else:
            trend_artist.set_data(shape.centerline[:, 0], shape.centerline[:, 1])
            reference = orient_tracking_reference(
                tracking_reference(shape, origin, grass_points=grass), shape.centerline)
            reference_artist.set_data(reference[:, 0], reference[:, 1])
            camera_reference = global_xy_to_camera_xz(reference, frame)
            camera_reference_artist.set_data(camera_reference[:, 0],
                                             camera_reference[:, 1])
            _, side_source = reference_side(shape, origin, grass)
            detail = (f"{shape.label} | length {shape.length_m:.2f} m | "
                      f"turn {shape.turn_deg:+.0f} deg | "
                      f"lawn side: {side_source if side_source == 'grass' else 'robot estimate'}")
            status_lines = []
            support = grass_inside_support(reference, grass)
            if support is not None:
                status_lines.append(f"Grass left of path: {support:.0%}")
            coverage = grass_reference_coverage(reference, grass, grass_support_radius)
            if coverage is not None:
                status_lines.append(f"Grass near path: {coverage:.0%}")
            right_extent = (right_half_width if right_half_width is not None
                            else robot_width / 2 if robot_width is not None else None)
            if right_extent is not None:
                clearance = 0.4 - right_extent
                status_lines.append(f"Nominal right-edge clearance: {clearance:+.2f} m")
            status_text.set_text("\n".join(status_lines))
            status_text.set_color("tab:red" if right_extent is not None
                                  and right_extent >= 0.4 else "black")
        world_heading = show_direction(reference, direction_arrow)
        camera_heading = show_direction(camera_reference, camera_direction_arrow)
        direction_text.set_text("" if world_heading is None
                                else f"Tracking direction: {world_heading:.0f} deg")
        camera_direction_text.set_text("" if camera_heading is None
                                       else f"Tracking direction: {camera_heading:.0f} deg")
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
    parser.add_argument("--grass-csv", type=Path,
                        help="Optional grass points CSV: frame,x,y or frame,camera_x,camera_z")
    parser.add_argument("--min-turn-radius", type=float,
                        help="Robot minimum turning radius in meters; limits reference curvature")
    width_group = parser.add_mutually_exclusive_group()
    width_group.add_argument("--robot-width", type=float,
                             help="Robot body width in meters; reports nominal right-edge clearance")
    width_group.add_argument("--right-half-width", type=float,
                             help="Distance from robot center to right body edge in meters")
    parser.add_argument("--grass-support-radius", type=float, default=0.25,
                        help="Distance in meters used to count grass coverage near reference")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.min_turn_radius is not None and args.min_turn_radius < 0:
        parser.error("--min-turn-radius must be nonnegative")
    if args.robot_width is not None and args.robot_width <= 0:
        parser.error("--robot-width must be positive")
    if args.right_half_width is not None and args.right_half_width <= 0:
        parser.error("--right-half-width must be positive")
    if args.grass_support_radius <= 0:
        parser.error("--grass-support-radius must be positive")
    frames = load_frames(args.log)
    if args.grass_csv:
        load_grass_csv(args.grass_csv, frames)
    replay(frames, args.speed, args.min_turn_radius, args.robot_width,
           args.right_half_width, args.grass_support_radius)


if __name__ == "__main__":
    main()
