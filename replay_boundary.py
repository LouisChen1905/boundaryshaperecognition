"""Replay boundary frames in global XY and camera XZ coordinates."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from itertools import chain, islice
from pathlib import Path
import re
import warnings
from time import perf_counter_ns
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button, Slider
import numpy as np

from boundary_shape import (grass_inside_support, grass_reference_coverage,
                            orient_shape,
                            orient_tracking_reference, recognize, reference_side,
                            tracking_reference)
from boundary_exploration import BoundaryExplorer, ExplorationConfig
from goal_pose_labels import GoalPoseLabels, GoalPoseAnnotator


LOG_PATTERN = re.compile(
    r"^\[(?P<clock>[^]]+)\].*?boundary_xz:\s*\[(?P<points>.*?)\]"
    r"\s*(?:,\s*grass_xz:\s*\[(?P<grass>.*?)\])?"
    r"\s*,\s*location info:\s*(?P<pose>[^\r\n]+)$"
)
POINT_PATTERN = re.compile(r"\(\s*([^,()]+)\s*,\s*([^,()]+)\s*\)")
CAMERA_TRANSLATION = np.array([0.4, 0.0, 0.0])
CAMERA_RPY = (-1.57, 0.0, 0)  # roll, pitch, yaw (radians)
EXPLORATION_PERFORMANCE_FIELDS = [
    "frame", "timestamp", "status", "reader_ms", "transform_ms", "raster_ms",
    "evidence_ms", "safe_ms", "closure_ms", "goal_ms", "goal_reach_ms",
    "goal_frontier_ms", "goal_score_ms", "snapshot_ms", "update_ms",
    "artist_update_ms", "render_ms", "callback_ms", "grass_points", "boundary_points",
    "observed_grass_cells", "observed_boundary_cells", "known_grass_cells",
    "mapped_boundary_cells", "safe_cells", "reachable_cells", "frontier_cells",
    "candidate_pairs", "sight_checks", "sight_cache_hits",
    "goal_search_radius", "goal_visible_gain", "boundary_frontier_cells",
    "goal_boundary_gain", "goal_boundary_distance", "map_version",
    "observation_fused", "removed_boundary_cells", "goal_action",
    "goal_release_reason", "goal_visit_count", "pose_alignment",
    "pose_before_timestamp", "pose_after_timestamp", "pose_gap_ms", "display_skipped"]


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
    camera_boundary_xyz: Optional[np.ndarray] = None
    camera_grass_xyz: Optional[np.ndarray] = None
    pose_alignment: str = "embedded"
    pose_before_timestamp: Optional[float] = None
    pose_after_timestamp: Optional[float] = None


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


def camera_to_world_matrix(frame: Frame) -> np.ndarray:
    """Full camera XYZ to global XYZ transform for the exploration input."""
    robot_rotation = rotation(*frame.robot_rpy)
    transform = np.eye(4)
    transform[:3, :3] = robot_rotation @ rotation(*CAMERA_RPY)
    transform[:3, 3] = frame.robot_xyz + robot_rotation @ CAMERA_TRANSLATION
    return transform


def load_frames(path: Path) -> list[Frame]:
    if path.suffix.lower() == ".jsonl":
        return load_data_package(path)
    with path.open("r", encoding="utf-8") as source:
        frames = [parse_frame(line, number) for number, line in enumerate(source, 1)]
    if not frames:
        raise ValueError("The log is empty")
    return frames


def _sample_timestamp(record):
    """Both navigation topics use data.timestamp in milliseconds, never arrival time."""
    try:
        value = float(record.get("data", {}).get("timestamp"))
    except (TypeError, ValueError):
        return None
    return value / 1000 if np.isfinite(value) and value > 0 else None


def _interpolate_attitude(first, second, fraction):
    """Shortest-arc quaternion SLERP for Rz(yaw) Ry(pitch) Rx(roll)."""
    def quaternion(rpy):
        cr, cp, cy = np.cos(np.asarray(rpy) / 2)
        sr, sp, sy = np.sin(np.asarray(rpy) / 2)
        return np.array([cr * cp * cy + sr * sp * sy,
                         sr * cp * cy - cr * sp * sy,
                         cr * sp * cy + sr * cp * sy,
                         cr * cp * sy - sr * sp * cy])

    q0, q1 = quaternion(first), quaternion(second)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    dot = np.clip(dot, -1., 1.)
    if dot > .9995:
        q = (1 - fraction) * q0 + fraction * q1
    else:
        angle = np.arccos(dot)
        q = (np.sin((1 - fraction) * angle) * q0
             + np.sin(fraction * angle) * q1) / np.sin(angle)
    w, x, y, z = q / np.linalg.norm(q)
    matrix = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)]])
    pitch = np.arcsin(np.clip(-matrix[2, 0], -1., 1.))
    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = 0.
        yaw = np.arctan2(-matrix[0, 1], matrix[1, 1])
    return np.array([roll, pitch, yaw])


def iter_data_package(path: Path, max_records: Optional[int] = None,
                      pose_max_gap_s: float = 0.1, alignment_stats=None):
    """Align clouds to location.data.timestamp with bounded pose/pending buffers.

    Position is interpolated linearly and attitude by quaternion SLERP. Only
    exact matches or bracketing samples within pose_max_gap_s are accepted.
    No nearest/latest pose, extrapolation, or system_ms fallback is used.
    max_records bounds lines actually read, including lookahead for a bracket.
    """
    if not np.isfinite(pose_max_gap_s) or pose_max_gap_s <= 0:
        raise ValueError("pose_max_gap_s must be finite and positive")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    stats = alignment_stats if alignment_stats is not None else {}
    names = ("location_packets", "visual_packets", "aligned_frames",
             "exact_frames", "interpolated_frames", "skipped_invalid_timestamp",
             "skipped_invalid_pose", "skipped_backwards_visual",
             "skipped_pose_gap", "skipped_before_pose_history",
             "skipped_no_future_pose", "skipped_pending_overflow")
    stats.update({name: 0 for name in names})
    stats["raw_records_read"] = 0
    stats["raw_bytes_read"] = 0
    pose_times, poses = [], []
    pending = deque()
    previous_time = None
    warned = set()

    def skip(reason):
        stats["skipped_" + reason] += 1
        if reason not in warned:
            warnings.warn("Skipping navigation packet: " + reason
                          + " (alignment uses data.timestamp only)", RuntimeWarning)
            warned.add(reason)

    def ready_frames(final=False):
        while pending:
            timestamp, boundary, grass = pending[0]
            index = bisect_left(pose_times, timestamp)
            if index < len(pose_times) and pose_times[index] == timestamp:
                xyz, rpy = poses[index]
                before = after = timestamp
                method = "exact"
            elif 0 < index < len(pose_times):
                before, after = pose_times[index - 1], pose_times[index]
                if after - before > pose_max_gap_s + 1e-9:
                    pending.popleft()
                    skip("pose_gap")
                    continue
                fraction = (timestamp - before) / (after - before)
                p0, a0 = poses[index - 1]
                p1, a1 = poses[index]
                xyz = p0 + fraction * (p1 - p0)
                rpy = _interpolate_attitude(a0, a1, fraction)
                method = "interpolated"
            elif pose_times and index == 0:
                pending.popleft()
                skip("before_pose_history")
                continue
            elif final:
                pending.popleft()
                skip("no_future_pose")
                continue
            else:
                break
            pending.popleft()
            robot_rotation = rotation(*rpy)
            world = (boundary @ rotation(*CAMERA_RPY).T + CAMERA_TRANSLATION)
            world = world @ robot_rotation.T + xyz
            clock = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")[:-3]
            frame = Frame(clock, timestamp, xyz.copy(), rpy.copy(), rpy[2],
                          boundary[:, (0, 2)], world,
                          camera_boundary_xyz=boundary, camera_grass_xyz=grass,
                          pose_alignment=method, pose_before_timestamp=before,
                          pose_after_timestamp=after)
            grass_world = (grass @ rotation(*CAMERA_RPY).T + CAMERA_TRANSLATION)
            frame.grass_world_xy = (grass_world @ robot_rotation.T + xyz)[:, :2]
            stats["aligned_frames"] += 1
            stats[method + "_frames"] += 1
            yield frame

    with path.open("r", encoding="utf-8-sig") as source:
        records = source if max_records is None else islice(source, max_records)
        for number, line in enumerate(records, 1):
            stats["raw_records_read"] = number
            stats["raw_bytes_read"] += len(line.encode("utf-8"))
            try:
                record = json.loads(line)
                topic = record.get("topic")
                if topic not in ("location", "visual_boundary_points"):
                    continue
                data = record.get("data", {})
                stats["location_packets" if topic == "location" else "visual_packets"] += 1
                timestamp = _sample_timestamp(record)
                if timestamp is None:
                    skip("invalid_timestamp")
                    continue
                if topic == "location":
                    position = data["poseOI"]["position"]
                    attitude = data["poseOI"]["attitude"]
                    xyz = np.array([position["east"], position["north"], position["up"]], dtype=float)
                    rpy = np.array([attitude["roll"], attitude["pitch"], attitude["head"]], dtype=float)
                    if not np.isfinite(xyz).all() or not np.isfinite(rpy).all():
                        skip("invalid_pose")
                        continue
                    index = bisect_left(pose_times, timestamp)
                    if index < len(pose_times) and pose_times[index] == timestamp:
                        poses[index] = (xyz, rpy)
                    else:
                        pose_times.insert(index, timestamp)
                        poses.insert(index, (xyz, rpy))
                    # About 40 seconds at 50 Hz; bounded even for huge logs.
                    if len(pose_times) > 2048:
                        del pose_times[0]
                        del poses[0]
                else:
                    if previous_time is not None and timestamp < previous_time:
                        skip("backwards_visual")
                        continue
                    previous_time = timestamp

                    def points(key, count_key):
                        count = int(data[count_key])
                        values = data[key]
                        if count < 0 or count > len(values):
                            raise ValueError(f"{count_key} exceeds available points")
                        return np.array([[p["cornerX"], p["cornerY"], p["cornerZ"]]
                                         for p in values[:count]], dtype=float).reshape(-1, 3)

                    boundary = points("vBoundaryPoints", "boundary_num")
                    grass = points("vGrassPoints", "grass_num")
                    if len(pending) >= 128:
                        pending.popleft()
                        skip("pending_overflow")
                    pending.append((timestamp, boundary, grass))
                yield from ready_frames()
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                raise ValueError(f"Line {number}: invalid navigation data: {error}") from error
        # At EOF or the record limit, never consume extra records or extrapolate.
        yield from ready_frames(final=True)


def load_data_package(path: Path) -> list[Frame]:
    """Compatibility loader; use ``iter_data_package`` for large files."""
    frames = list(iter_data_package(path))
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


def update_exploration(explorer: BoundaryExplorer, frame: Frame,
                       performance=None):
    """Advance the map exactly once for a new frame."""
    if performance is not None:
        performance["pose_alignment"] = frame.pose_alignment
        performance["pose_before_timestamp"] = frame.pose_before_timestamp
        performance["pose_after_timestamp"] = frame.pose_after_timestamp
        performance["pose_gap_ms"] = (
            (frame.pose_after_timestamp - frame.pose_before_timestamp) * 1000
            if frame.pose_before_timestamp is not None and frame.pose_after_timestamp is not None
            else "")
    pose = (frame.robot_xyz[0], frame.robot_xyz[1], frame.robot_yaw + np.pi / 2)
    if frame.camera_boundary_xyz is not None:
        return explorer.update(frame.camera_grass_xyz, frame.camera_boundary_xyz,
                               pose, frame.timestamp,
                               sensor_to_world=camera_to_world_matrix(frame),
                               performance=performance)
    grass = frame.grass_world_xy
    return explorer.update(grass if grass is not None else np.empty((0, 2)),
                           frame.world_points[:, :2], pose, frame.timestamp,
                           performance=performance)


def _cell_polygons(cells, resolution):
    if not cells:
        return np.empty((0, 4, 2))
    origins = np.asarray(list(cells), dtype=float) * resolution
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * resolution
    return origins[:, None, :] + corners[None, :, :]


def same_robot_pose(first: Frame, second: Frame, position_tolerance=0.03,
                    angle_tolerance_deg=2.0):
    """Compare against a fixed anchor, including wrapped 3D attitude."""
    relative = rotation(*first.robot_rpy).T @ rotation(*second.robot_rpy)
    angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
    return (np.linalg.norm(first.robot_xyz - second.robot_xyz) <= position_tolerance
            and angle <= np.radians(angle_tolerance_deg))


def profile_exploration_stream(frames, config: ExplorationConfig,
                               performance_log: Path,
                               verified_start_radius=None, goal_scorer=None):
    """Run the same exploration updates without plotting, writing every frame."""
    explorer = BoundaryExplorer(config, goal_scorer=goal_scorer)
    count = 0
    totals = {name: 0.0 for name in EXPLORATION_PERFORMANCE_FIELDS
              if name.endswith("_ms")}
    maximum_update_ms = 0.0
    previous_time = None
    with Path(performance_log).open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=EXPLORATION_PERFORMANCE_FIELDS)
        writer.writeheader()
        source = iter(frames)
        try:
            while True:
                started = perf_counter_ns()
                frame = next(source, None)
                reader_ms = (perf_counter_ns() - started) / 1e6
                if frame is None:
                    break
                if previous_time is not None and frame.timestamp < previous_time:
                    raise ValueError("Navigation frame timestamps must be monotonic")
                if count == 0 and verified_start_radius is not None:
                    explorer.seed_verified_grass_disk(frame.robot_xyz[:2],
                                                      verified_start_radius)
                performance = {}
                update_exploration(explorer, frame, performance)
                count += 1
                performance.update(frame=count, timestamp=frame.timestamp,
                                   reader_ms=reader_ms,
                                   callback_ms=(perf_counter_ns() - started) / 1e6)
                writer.writerow({name: performance.get(name, "")
                                 for name in EXPLORATION_PERFORMANCE_FIELDS})
                for name in totals:
                    totals[name] += performance.get(name, 0.0)
                maximum_update_ms = max(maximum_update_ms,
                                        performance["update_ms"])
                previous_time = frame.timestamp
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()
    if count == 0:
        raise ValueError("No visual boundary frames with location in the data package")
    return count, totals, maximum_update_ms


def replay_exploration_stream(frames, speed: float, config: ExplorationConfig,
                              verified_start_radius=None, history_size=50,
                              performance_log=None, labels_path=None, source_path=None,
                              static_position_tolerance=0.03,
                              static_angle_tolerance_deg=2.0, goal_scorer=None) -> None:
    """Replay an iterator incrementally and display the explorer's live map.

    Previous/Next use a bounded cache. Revisiting a cached frame never feeds
    that observation into the explorer a second time.
    """
    if (labels_path is not None and performance_log is not None
            and Path(labels_path).resolve() == Path(performance_log).resolve()):
        raise ValueError("Goal labels and performance log must use different files")
    label_store = GoalPoseLabels(labels_path, source_path) if labels_path is not None else None
    frame_iter = iter(frames)
    first_read_started = perf_counter_ns()
    first = next(frame_iter, None)
    first_read_ms = (perf_counter_ns() - first_read_started) / 1e6
    if first is None:
        raise ValueError("No visual boundary frames with location in the data package")
    source = chain((first,), frame_iter)
    explorer = BoundaryExplorer(config, goal_scorer=goal_scorer)
    if verified_start_radius is not None:
        explorer.seed_verified_grass_disk(first.robot_xyz[:2], verified_start_radius)
    performance_file = None
    performance_writer = None
    if performance_log is not None:
        performance_file = Path(performance_log).open("w", encoding="utf-8", newline="")
        performance_writer = csv.DictWriter(performance_file,
                                            fieldnames=EXPLORATION_PERFORMANCE_FIELDS)
        performance_writer.writeheader()

    fig, (ax, camera_ax) = plt.subplots(1, 2, figsize=(15, 7))
    fig.subplots_adjust(bottom=0.16, top=0.88, wspace=0.28)
    fig.suptitle("Boundary exploration replay | gray: unknown, green: confirmed grass, "
                 "red: boundary, teal: safe", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Global X (m)")
    ax.set_ylabel("Global Y (m)")
    ax.set_facecolor("#eeeeee")
    ax.grid(True, alpha=0.2)
    grass_cells = PolyCollection([], facecolors="#a6d854", edgecolors="none",
                                 alpha=0.75, zorder=1, label="Confirmed grass")
    safe_cells = PolyCollection([], facecolors="#27aeae", edgecolors="none",
                                alpha=0.7, zorder=2, label="Robot-center safe")
    boundary_cells = PolyCollection([], facecolors="#cf4446", edgecolors="none",
                                    alpha=0.9, zorder=3, label="Observed boundary")
    for collection in (grass_cells, safe_cells, boundary_cells):
        ax.add_collection(collection)
    raw_points = ax.scatter([], [], s=8, c="darkgreen", alpha=0.5,
                            label="Current boundary points", zorder=4)
    robot = ax.scatter([], [], s=90, marker="x", c="black", label="Robot", zorder=6)
    goal = ax.scatter([], [], s=110, marker="*", c="#e69f00",
                      label="Next viewing pose", zorder=7)
    path_line, = ax.plot([], [], c="#e69f00", lw=2.5, label="Safe path", zorder=5)
    shape_line, = ax.plot([], [], c="tab:blue", lw=1.5,
                          label="Single-frame fit", zorder=5)
    status_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, va="bottom",
                          fontsize=9, bbox={"facecolor": "white", "alpha": 0.85,
                                             "edgecolor": "none"}, zorder=8)
    ax.legend(loc="upper right", fontsize=8)

    camera_ax.set_aspect("equal", adjustable="box")
    camera_ax.set_xlabel("Camera X (m)")
    camera_ax.set_ylabel("Camera Z (m)")
    camera_ax.grid(True, alpha=0.3)
    camera_points = camera_ax.scatter([], [], s=10, c="darkgreen", label="Boundary")
    camera_grass = camera_ax.scatter([], [], s=8, c="#a6d854", alpha=0.6,
                                     label="Grass")
    camera_shape, = camera_ax.plot([], [], c="tab:blue", lw=2,
                                   label="Single-frame fit")
    camera_ax.legend(loc="upper right", fontsize=8)

    previous_button = Button(fig.add_axes((0.18, 0.04, 0.15, 0.06)), "Previous")
    play_button = Button(fig.add_axes((0.42, 0.04, 0.15, 0.06)), "Pause")
    next_button = Button(fig.add_axes((0.66, 0.04, 0.15, 0.06)), "Next")
    skip_button = Button(fig.add_axes((0.42, 0.112, 0.18, 0.035)), "Skip static [S]")
    skip_text = fig.text(.02, .125, "", fontsize=9)
    history = deque(maxlen=history_size)
    state = {"index": -1, "playing": True, "ended": False, "number": 0,
             "previous_time": None, "skipping": False, "skip_count": 0}

    def cancel_skip():
        if state["skipping"]:
            skip_text.set_text("Skipped {} static frames (stopped)".format(state["skip_count"]))
        state["skipping"] = False
        skip_timer.stop()

    def pause_for_label():
        cancel_skip()
        state["playing"] = False
        timer.stop()
        play_button.label.set_text("Play")

    annotator = (GoalPoseAnnotator(fig, ax, label_store,
                                   pause_for_label) if label_store is not None else None)

    def show(entry, performance=None):
        artist_started = perf_counter_ns()
        number, frame, result = entry
        if annotator is not None:
            annotator.show(number, frame, result.goal_pose)
        resolution = config.resolution
        grass_cells.set_verts(_cell_polygons(result.known_grass, resolution))
        safe_cells.set_verts(_cell_polygons(result.safe_cells, resolution))
        boundary_cells.set_verts(_cell_polygons(result.boundary_cells, resolution))
        raw_points.set_offsets(frame.world_points[:, :2])
        robot.set_offsets(frame.robot_xyz[:2].reshape(1, 2))
        if result.goal_pose is None:
            goal.set_offsets(np.empty((0, 2)))
        else:
            goal.set_offsets(np.asarray(result.goal_pose[:2]).reshape(1, 2))
        path_line.set_data(result.path_xy[:, 0], result.path_xy[:, 1])
        shape = orient_shape(recognize(frame.world_points), frame.robot_xyz[:2])
        if shape is None:
            shape_line.set_data([], [])
        else:
            shape_line.set_data(shape.centerline[:, 0], shape.centerline[:, 1])
        camera_points.set_offsets(frame.camera_xz)
        camera_grass.set_offsets(frame.camera_grass_xyz[:, (0, 2)]
                                 if frame.camera_grass_xyz is not None
                                 else np.empty((0, 2)))
        camera_fit = orient_shape(recognize(frame.camera_xz), (0, 0))
        if camera_fit is None:
            camera_shape.set_data([], [])
        else:
            camera_shape.set_data(camera_fit.centerline[:, 0],
                                  camera_fit.centerline[:, 1])
        camera_ax.set_title(f"Camera XZ | {len(frame.camera_xz)} boundary points")
        camera_cloud = frame.camera_xz
        if frame.camera_grass_xyz is not None and len(frame.camera_grass_xyz):
            camera_cloud = np.vstack((camera_cloud,
                                      frame.camera_grass_xyz[:, (0, 2)]))
        if len(camera_cloud):
            low, high = camera_cloud.min(axis=0), camera_cloud.max(axis=0)
            camera_ax.set_xlim(low[0] - 0.3, high[0] + 0.3)
            camera_ax.set_ylim(low[1] - 0.3, high[1] + 0.3)
        status_text.set_text(
            f"{result.status}: {result.reason}\n"
            f"Grass {len(result.known_grass)} | Boundary {len(result.boundary_cells)} "
            f"| Safe {len(result.safe_cells)}\n"
            f"New grass {result.new_grass_cells} | New boundary {result.new_boundary_cells} "
            f"| Closed {result.boundary_closed}")
        ax.set_title(f"Frame {number} | {frame.clock} | exploration: {result.status}")
        extent = [frame.robot_xyz[:2]]
        if len(frame.world_points):
            extent.append(frame.world_points[:, :2])
        if result.known_grass or result.boundary_cells:
            extent.append(np.asarray(list(result.known_grass | result.boundary_cells))
                          * resolution)
        if result.goal_pose is not None:
            extent.append(np.asarray(result.goal_pose[:2]).reshape(1, 2))
        cloud = np.vstack(extent)
        low, high = cloud.min(axis=0), cloud.max(axis=0)
        center = (low + high) / 2
        radius = max(0.75, max(high - low) / 2 + 0.4)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        if performance is None:
            fig.canvas.draw_idle()
        else:
            performance["artist_update_ms"] = (perf_counter_ns() - artist_started) / 1e6
            render_started = perf_counter_ns()
            fig.canvas.draw()
            performance["render_ms"] = (perf_counter_ns() - render_started) / 1e6

    def advance(render=True):
        if state["index"] + 1 < len(history):
            state["index"] += 1
            if render:
                show(history[state["index"]])
            return True
        if state["ended"]:
            return False
        callback_started = perf_counter_ns()
        read_started = callback_started
        frame = next(source, None)
        reader_ms = (perf_counter_ns() - read_started) / 1e6
        if frame is None:
            state["ended"] = True
            return False
        performance = {} if performance_writer is not None else None
        result = update_exploration(explorer, frame, performance)
        state["number"] += 1
        history.append((state["number"], frame, result))
        state["index"] = len(history) - 1
        if state["previous_time"] is not None:
            delay = (frame.timestamp - state["previous_time"]) * 1000 / speed
            elapsed = (perf_counter_ns() - callback_started) / 1e6
            timer.interval = max(1, min(1000, int(delay - elapsed))) if delay > 0 else 1
        state["previous_time"] = frame.timestamp
        if render:
            show(history[-1], performance)
            if state["previous_time"] is not None and state["number"] > 1:
                elapsed = (perf_counter_ns() - callback_started) / 1e6
                timer.interval = max(1, min(1000, int(delay - elapsed))) if delay > 0 else 1
        elif performance is not None:
            performance["artist_update_ms"] = 0.0
            performance["render_ms"] = 0.0
        if performance_writer is not None:
            performance["display_skipped"] = int(not render)
            performance["frame"] = state["number"]
            performance["timestamp"] = frame.timestamp
            performance["reader_ms"] = (first_read_ms if state["number"] == 1
                                        else reader_ms)
            performance["callback_ms"] = ((perf_counter_ns() - callback_started) / 1e6
                                          + (first_read_ms if state["number"] == 1
                                             else 0.0))
            performance_writer.writerow({field: performance.get(field, "")
                                         for field in EXPLORATION_PERFORMANCE_FIELDS})
            performance_file.flush()
        return True

    timer = fig.canvas.new_timer(interval=max(20, int(100 / speed)))
    skip_timer = fig.canvas.new_timer(interval=1)
    state["last_skip_display_ns"] = 0

    def skip_chunk():
        if not state["skipping"]:
            return
        started = perf_counter_ns()
        # Yield to the GUI regularly so long stationary recordings can be cancelled.
        for _ in range(50):
            if not advance(render=False):
                cancel_skip()
                break
            current = history[state["index"]][1]
            if not same_robot_pose(state["skip_anchor"], current,
                                   static_position_tolerance,
                                   static_angle_tolerance_deg):
                cancel_skip()
                break
            state["skip_count"] += 1
            if (perf_counter_ns() - started) / 1e6 >= 25:
                break
        now = perf_counter_ns()
        if not state["skipping"] or now - state["last_skip_display_ns"] >= 250_000_000:
            skip_text.set_text("Skipped {} static frames{}".format(
                state["skip_count"], " (working...)" if state["skipping"] else ""))
            show(history[state["index"]])
            state["last_skip_display_ns"] = now
        if state["skipping"]:
            skip_timer.start()

    skip_timer.add_callback(skip_chunk)

    def skip_static(event=None):
        if state["skipping"]:
            cancel_skip()
            return
        pause_for_label()
        if annotator is not None and annotator.active:
            annotator.cancel()
        state["skip_anchor"] = history[state["index"]][1]
        state["skip_count"] = 0
        state["skipping"] = True
        skip_text.set_text("Skipping static frames...")
        skip_timer.start()
        fig.canvas.draw_idle()

    skip_button.on_clicked(skip_static)

    def tick():
        if state["playing"] and not advance():
            state["playing"] = False
            play_button.label.set_text("Play")
        if state["playing"]:
            timer.start()

    timer.add_callback(tick)

    def toggle(event):
        cancel_skip()
        if annotator is not None and annotator.active:
            annotator.cancel()
        state["playing"] = not state["playing"]
        play_button.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"]:
            timer.start()
        else:
            timer.stop()
        fig.canvas.draw_idle()

    def previous(event):
        cancel_skip()
        if state["index"] > 0:
            state["index"] -= 1
            show(history[state["index"]])

    play_button.on_clicked(toggle)
    previous_button.on_clicked(previous)
    def next_frame(event=None):
        cancel_skip()
        advance()

    next_button.on_clicked(next_frame)

    def on_key(event):
        if event.key == "left":
            previous(event)
        elif event.key == "right":
            next_frame()
        elif event.key == " ":
            toggle(event)
        elif event.key == "s":
            skip_static()
        elif event.key == "escape":
            cancel_skip()

    fig.canvas.mpl_connect("key_press_event", on_key)
    advance()
    timer.start()
    try:
        plt.show()
    finally:
        timer.stop()
        skip_timer.stop()
        if performance_file is not None:
            performance_file.close()
        close = getattr(frame_iter, "close", None)
        if close is not None:
            close()


def replay(frames: list[Frame], speed: float, min_turn_radius=None,
           robot_width=None, right_half_width=None,
           grass_support_radius=0.25, labels_path=None, source_path=None) -> None:
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

    def pause_for_label():
        state["playing"] = False
        timer.stop()
        play_button.label.set_text("Play")

    annotator = (GoalPoseAnnotator(fig, ax, GoalPoseLabels(labels_path, source_path),
                                   pause_for_label) if labels_path is not None else None)

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
        if annotator is not None:
            annotator.show(index + 1, frame)
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
        if annotator is not None and annotator.active:
            annotator.cancel()
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
    parser.add_argument("--max-records", type=int,
                        help="Read at most this many JSONL input records")
    parser.add_argument("--pose-max-gap-ms", type=float, default=100.0,
                        help="Maximum location bracket interval for timestamp interpolation (ms)")
    parser.add_argument("--exploration-robot-radius", type=float, default=0.2,
                        help="Exploration footprint radius in meters (demo default: 0.2)")
    parser.add_argument("--exploration-sensor-range", type=float, default=2.0,
                        help="Exploration sensor range in meters (demo default: 2.0)")
    parser.add_argument("--grid-resolution", type=float, default=0.1,
                        help="Exploration map cell size in meters")
    parser.add_argument("--goal-path-radii", type=float, nargs="+",
                        default=(1.0, 2.0, 3.0), metavar="R",
                        help="Increasing safe-path search radii in meters")
    parser.add_argument("--min-visible-gain", type=float, default=2.0,
                        help="Minimum visible frontier weight for an early goal")
    parser.add_argument("--boundary-frontier-distance", type=float, default=0.3,
                        help="Distance to observed boundary for priority frontiers (meters)")
    parser.add_argument("--boundary-distance-weight", type=float, default=2.0,
                        help="Viewpoint boundary-distance penalty per meter")
    parser.add_argument("--goal-reached-distance", type=float, default=0.15,
                        help="Distance in meters that marks an assigned goal reached")
    parser.add_argument("--goal-progress-timeout-s", type=float, default=15.0,
                        help="Release a goal after this many seconds without progress")
    parser.add_argument("--goal-progress-distance", type=float, default=0.1,
                        help="Cumulative distance improvement in meters that resets timeout")
    parser.add_argument("--goal-refresh-interval-s", type=float, default=2.0,
                        help="Re-score all candidate views at least this often (sensor seconds)")
    parser.add_argument("--goal-refresh-distance", type=float, default=0.3,
                        help="Robot displacement since last selection that refreshes the goal (m)")
    parser.add_argument("--goal-refresh-yaw-deg", type=float, default=20.0,
                        help="Wrapped heading change since last selection that refreshes the goal")
    parser.add_argument("--boundary-clear-observations", type=int, default=3,
                        help="Grass-only observations required to retract a boundary cell")
    parser.add_argument("--verified-start-radius", type=float,
                        help="Optional externally verified grass disk around first pose")
    parser.add_argument("--performance-log", type=Path,
                        help="Write per-frame JSONL replay timings to this CSV")
    parser.add_argument("--profile-only", action="store_true",
                        help="Process JSONL frames and write timing CSV without opening a plot")
    parser.add_argument("--goal-labels", type=Path,
                        help="Desired goal labels JSONL (default: <log stem>_goal_labels.jsonl in cwd)")
    parser.add_argument("--goal-score-model", type=Path,
                        help="Learned positive score residual JSON (JSONL exploration only; full search)")
    parser.add_argument("--static-position-tolerance", type=float, default=0.03,
                        help="Static-skip position tolerance in meters relative to its starting pose")
    parser.add_argument("--static-angle-tolerance-deg", type=float, default=2.0,
                        help="Static-skip 3D attitude tolerance in degrees")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if (not np.isfinite(args.static_position_tolerance) or args.static_position_tolerance < 0
            or not np.isfinite(args.static_angle_tolerance_deg)
            or not 0 <= args.static_angle_tolerance_deg <= 180):
        parser.error("Static tolerances must be finite: position >= 0, angle in [0, 180]")
    if args.min_turn_radius is not None and args.min_turn_radius < 0:
        parser.error("--min-turn-radius must be nonnegative")
    if args.robot_width is not None and args.robot_width <= 0:
        parser.error("--robot-width must be positive")
    if args.right_half_width is not None and args.right_half_width <= 0:
        parser.error("--right-half-width must be positive")
    if args.grass_support_radius <= 0:
        parser.error("--grass-support-radius must be positive")
    if args.max_records is not None and args.max_records <= 0:
        parser.error("--max-records must be positive")
    if not np.isfinite(args.pose_max_gap_ms) or args.pose_max_gap_ms <= 0:
        parser.error("--pose-max-gap-ms must be finite and positive")
    if args.verified_start_radius is not None and args.verified_start_radius <= 0:
        parser.error("--verified-start-radius must be positive")
    try:
        exploration_config = ExplorationConfig(
            robot_radius=args.exploration_robot_radius,
            sensor_range=args.exploration_sensor_range,
            resolution=args.grid_resolution,
            path_radii=args.goal_path_radii,
            min_visible_gain=args.min_visible_gain,
            boundary_frontier_distance=args.boundary_frontier_distance,
            boundary_distance_weight=args.boundary_distance_weight,
            goal_reached_distance=args.goal_reached_distance,
            goal_progress_timeout_s=args.goal_progress_timeout_s,
            goal_progress_distance=args.goal_progress_distance,
            goal_refresh_interval_s=args.goal_refresh_interval_s,
            goal_refresh_distance=args.goal_refresh_distance,
            goal_refresh_yaw_deg=args.goal_refresh_yaw_deg,
            boundary_clear_observations=args.boundary_clear_observations)
    except ValueError as error:
        parser.error(str(error))
    goal_scorer = None
    if args.goal_score_model is not None:
        if args.log.suffix.lower() != ".jsonl":
            parser.error("--goal-score-model requires a JSONL exploration replay")
        from goal_score_rl import GoalScorePolicy
        try:
            goal_scorer = GoalScorePolicy.load(args.goal_score_model)
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))
    if args.log.suffix.lower() == ".jsonl":
        if args.grass_csv:
            parser.error("--grass-csv is only supported for the text log replay")
        alignment_stats = {}
        frames = iter_data_package(args.log, args.max_records,
                                   pose_max_gap_s=args.pose_max_gap_ms / 1000,
                                   alignment_stats=alignment_stats)
        if args.profile_only:
            performance_log = args.performance_log or Path("exploration_performance.csv")
            count, totals, maximum = profile_exploration_stream(
                frames,
                exploration_config, performance_log, args.verified_start_radius,
                goal_scorer=goal_scorer)
            print(f"Profiled {count} visual frames; CSV: {performance_log}")
            print("Total ms by stage: " + ", ".join(
                f"{name}={totals[name]:.1f}" for name in
                ("reader_ms", "raster_ms", "safe_ms", "closure_ms",
                 "goal_ms", "update_ms")))
            print(f"Maximum update: {maximum:.1f} ms")
            print("Pose alignment: " + str(alignment_stats))
            return
        replay_exploration_stream(frames,
                                  args.speed, exploration_config,
                                  args.verified_start_radius,
                                  performance_log=args.performance_log,
                                  labels_path=args.goal_labels or Path(args.log.stem + "_goal_labels.jsonl"),
                                  source_path=args.log,
                                  static_position_tolerance=args.static_position_tolerance,
                                  static_angle_tolerance_deg=args.static_angle_tolerance_deg,
                                  goal_scorer=goal_scorer)
        return
    if args.profile_only:
        parser.error("--profile-only requires a JSONL log")
    frames = load_frames(args.log)
    if args.grass_csv:
        load_grass_csv(args.grass_csv, frames)
    replay(frames, args.speed, args.min_turn_radius, args.robot_width,
           args.right_half_width, args.grass_support_radius,
           labels_path=args.goal_labels or Path(args.log.stem + "_goal_labels.jsonl"),
           source_path=args.log)


if __name__ == "__main__":
    main()
