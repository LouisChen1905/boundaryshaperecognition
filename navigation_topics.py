"""Bounded navigation parsing and goal-score trajectory-label export."""

import argparse
from bisect import bisect_right
from collections import Counter, deque
from dataclasses import dataclass
from itertools import islice
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


def finite_number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("expected a finite number")
    return value


def counted(values, count):
    size = int(count)
    if size != count or size < 0 or size > len(values):
        raise ValueError("invalid effective array count")
    return values[:size]


@dataclass
class ObstacleObject:
    data_type: int
    object_id: int
    dynamic: int
    camera_xyz: np.ndarray


@dataclass
class ObstacleObservation:
    timestamp: float
    sensor_id: int
    state: int
    objects: tuple

    @property
    def camera_xyz(self):
        if not self.objects:
            return np.empty((0, 3))
        return np.concatenate([obj.camera_xyz for obj in self.objects])


def parse_obstacles(data):
    objects = []
    for obj in counted(data["vObs"], data["sum"]):
        points = np.array([[finite_number(p[k]) for k in
                            ("cornerX", "cornerY", "cornerZ")]
                           for p in counted(obj["vInfo"], obj["sum"])],
                          dtype=float).reshape(-1, 3)
        objects.append(ObstacleObject(int(obj["dataType"]), int(obj["objectID"]),
                                      int(obj["dynamic"]), points))
    timestamp = finite_number(data["timestamp"]) / 1000
    if timestamp <= 0:
        raise ValueError("obstacle timestamp must be positive")
    return ObstacleObservation(timestamp, int(data["sensorID"]), int(data["state"]),
                               tuple(objects))


class AuxiliaryTopics:
    """Bounded histories. Obstacle joins are exact; RPM uses causal arrival time.

    motor_cut timestamps are in a different clock domain. system_ms is the
    explicit fallback for motor_cut and cut_ctrl, never for cloud/pose alignment.
    Missing or stale RPM is unknown, rather than zero or active.
    """

    def __init__(self, max_age_s=.5, capacity=2048):
        if not math.isfinite(max_age_s) or max_age_s <= 0 or capacity < 1:
            raise ValueError("positive age and capacity required")
        self.max_age_s = max_age_s
        self.capacity = capacity
        self.obstacles = {}
        self.states = {"motor_cut": [], "cut_ctrl": []}

    def ingest(self, record):
        topic, data = record.get("topic"), record.get("data", {})
        if topic == "visual_obstacle_front":
            try:
                timestamp = finite_number(data.get("timestamp"))
            except (ValueError, TypeError):
                return False
            if timestamp <= 0:
                return False
            observation = parse_obstacles(data)
            self.obstacles[observation.timestamp] = observation
            if len(self.obstacles) > 128:
                del self.obstacles[min(self.obstacles)]
        elif topic in self.states:
            try:
                timestamp = finite_number(record.get("system_ms")) / 1000
            except (ValueError, TypeError):
                return False
            if timestamp <= 0:
                return False
            sample = dict(timestamp=timestamp, rpm=finite_number(data["rpm"]),
                          timestamp_basis="system_ms",
                          data_timestamp=data.get("timestamp"), status=data.get("status"))
            history = self.states[topic]
            index = bisect_right([s["timestamp"] for s in history], timestamp)
            history.insert(index, sample)
            if topic == "motor_cut":
                # Preserve run identity across visual frames, including a stop
                # and restart that occur entirely between two camera samples.
                for i in range(index, len(history)):
                    current = history[i]
                    previous = history[i - 1] if i else None
                    continuous = (previous is not None and abs(previous["rpm"]) > 0
                                  and current["timestamp"] - previous["timestamp"] <= self.max_age_s + 1e-9)
                    current["active_since"] = (
                        previous["active_since"] if continuous else current["timestamp"]
                    ) if abs(current["rpm"]) > 0 else None
            del history[:-self.capacity]
        return True

    def state_at(self, topic, timestamp):
        history = self.states[topic]
        index = bisect_right([s["timestamp"] for s in history], timestamp) - 1
        if index < 0 or timestamp - history[index]["timestamp"] > self.max_age_s + 1e-9:
            return None
        return dict(history[index])


def export_motor_labels(source, output, max_records=6000, max_age_s=.5):
    """Export observed locations, not desired exploration goals, atomically."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("label output must differ from input")
    if max_records < 1:
        raise ValueError("max_records must be positive")
    history = AuxiliaryTopics(max_age_s)
    stats = dict(raw_records_read=0, raw_bytes_read=0, labels=0,
                 location_unknown_motor=0, location_inactive_motor=0,
                 skipped_auxiliary_timestamp=0, skipped_location_timestamp=0)
    topics = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with source.open(encoding="utf-8-sig") as stream, tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(output.parent), delete=False) as target:
            temporary = target.name
            for number, line in enumerate(islice(stream, max_records), 1):
                stats["raw_records_read"] = number
                stats["raw_bytes_read"] += len(line.encode("utf-8"))
                try:
                    record = json.loads(line)
                    topic = record.get("topic")
                    topics[topic] += 1
                    if not history.ingest(record):
                        stats["skipped_auxiliary_timestamp"] += 1
                    if topic != "location":
                        continue
                    data = record["data"]
                    try:
                        timestamp = finite_number(data.get("timestamp")) / 1000
                    except (ValueError, TypeError):
                        stats["skipped_location_timestamp"] += 1
                        continue
                    if timestamp <= 0:
                        stats["skipped_location_timestamp"] += 1
                        continue
                    motor = history.state_at("motor_cut", timestamp)
                    if motor is None:
                        stats["location_unknown_motor"] += 1
                        continue
                    if abs(motor["rpm"]) == 0:
                        stats["location_inactive_motor"] += 1
                        continue
                    position, attitude = data["poseOI"]["position"], data["poseOI"]["attitude"]
                    xyz = [finite_number(position[k]) for k in ("east", "north", "up")]
                    rpy = [finite_number(attitude[k]) for k in ("roll", "pitch", "head")]
                    label = dict(schema_version=1, label_type="motor_cut_active_location",
                                 source=str(source.resolve()), record=number,
                                 sequence=record.get("sequence"), timestamp=timestamp,
                                 coordinate_frame="poseOI_world", training_label=True,
                                 robot_xyz=xyz, robot_rpy=rpy,
                                 robot_pose_world=dict(x=xyz[0], y=xyz[1], yaw=rpy[2] + math.pi / 2),
                                 motor_cut=motor, cut_ctrl=history.state_at("cut_ctrl", timestamp),
                                 flag_rule="abs(motor_cut.rpm) > 0",
                                 motor_age_s=timestamp - motor["timestamp"])
                    target.write(json.dumps(label, allow_nan=False) + "\n")
                    stats["labels"] += 1
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError("Line {}: {}".format(number, error)) from error
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, str(output))
        temporary = None
    finally:
        if temporary is not None:
            os.remove(temporary)
    stats.update(topics=dict(topics), source=str(source.resolve()), output=str(output.resolve()),
                 max_records=max_records, motor_max_age_s=max_age_s,
                 motor_timestamp_basis="system_ms; causal relative to location.data.timestamp")
    return stats


def export_goal_labels(source, output, max_records=6000, max_frames=800,
                       max_labels=128, lookahead_distance=1., max_horizon_s=10.,
                       label_interval_s=1., max_frame_gap_s=.5,
                       pose_max_gap_s=.1, max_age_s=.5):
    """Pair an aligned visual frame with a future motor-active trajectory pose.

    Targets are observed future poses, not the current pose and not manual
    desired goals. Replay owns frame numbering, pose interpolation and heading.
    Only a bounded queue of small pose records is retained, never cloud frames.
    """
    # Local import avoids the replay -> navigation_topics import cycle.
    from replay_boundary import iter_data_package

    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("label output must differ from input")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (max_records, max_frames, max_labels)):
        raise ValueError("record, frame and label budgets must be positive integers")
    for value in (lookahead_distance, max_horizon_s, label_interval_s,
                  max_frame_gap_s, pose_max_gap_s, max_age_s):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("distances, intervals and gaps must be finite and positive")
    alignment = {}
    stats = dict(labels=0, frames_read=0, inactive_or_unknown_frames=0,
                 skipped_discontinuity=0, skipped_horizon=0,
                 skipped_pending_overflow=0, skipped_no_future_target=0)
    pending = deque()
    previous, last_origin_time = None, None
    travelled = 0.
    temporary = None
    options = dict(lookahead_distance=lookahead_distance, max_horizon_s=max_horizon_s,
                   label_interval_s=label_interval_s, max_frame_gap_s=max_frame_gap_s,
                   motor_max_age_s=max_age_s, pose_max_gap_s=pose_max_gap_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = iter_data_package(source, max_records, pose_max_gap_s, alignment,
                               motor_max_age_s=max_age_s)
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=str(output.parent), delete=False) as target:
            temporary = target.name
            for number, frame in enumerate(islice(stream, max_frames), 1):
                stats["frames_read"] = number
                if frame.training_label is not True:
                    stats["inactive_or_unknown_frames"] += 1
                    stats["skipped_discontinuity"] += len(pending)
                    pending.clear()
                    previous, last_origin_time = None, None
                    continue
                current = dict(frame=number, timestamp=float(frame.timestamp), clock=frame.clock,
                               pose=[float(frame.robot_xyz[0]), float(frame.robot_xyz[1]),
                                     float(frame.robot_yaw + math.pi / 2)],
                               motor_cut=frame.motor_cut, cut_ctrl=frame.cut_ctrl,
                               pose_alignment=frame.pose_alignment)
                if previous is not None:
                    gap = current["timestamp"] - previous["timestamp"]
                    if (gap <= 0 or gap > max_frame_gap_s + 1e-9
                            or current["motor_cut"]["active_since"] != previous["motor_cut"]["active_since"]):
                        stats["skipped_discontinuity"] += len(pending)
                        pending.clear()
                        previous, last_origin_time = None, None
                if previous is None:
                    travelled = 0.
                else:
                    travelled += float(np.linalg.norm(np.subtract(current["pose"][:2], previous["pose"][:2])))
                current["travelled"] = travelled
                while pending and current["timestamp"] - pending[0]["timestamp"] > max_horizon_s + 1e-9:
                    pending.popleft()
                    stats["skipped_horizon"] += 1
                remaining = deque()
                for origin_index, origin in enumerate(pending):
                    distance = travelled - origin["travelled"]
                    displacement = float(np.linalg.norm(np.subtract(current["pose"][:2], origin["pose"][:2])))
                    if distance + 1e-9 < lookahead_distance or displacement < min(.2, lookahead_distance):
                        remaining.append(origin)
                        continue
                    label = dict(schema_version=1, label_type="motor_cut_trajectory_goal",
                                 source=str(source.resolve()), frame=origin["frame"],
                                 timestamp=origin["timestamp"], clock=origin["clock"],
                                 coordinate_frame="world_xy", yaw_unit="radians",
                                 yaw_convention="CCW from world +X; robot forward is head + pi/2",
                                 robot_pose_world=dict(zip(("x", "y", "yaw"), origin["pose"])),
                                 desired_goal_world=dict(zip(("x", "y", "yaw"), current["pose"])),
                                 algorithm_goal_world=None,
                                 motor_cut=origin["motor_cut"],
                                 provenance=dict(method="future_active_trajectory_distance",
                                     target_frame=number, target_timestamp=current["timestamp"],
                                     target_motor_cut=current["motor_cut"],
                                     trajectory_distance=distance, displacement=displacement,
                                     origin_pose_alignment=origin["pose_alignment"],
                                     target_pose_alignment=current["pose_alignment"],
                                     export_options=options))
                    target.write(json.dumps(label, allow_nan=False) + "\n")
                    stats["labels"] += 1
                    if stats["labels"] >= max_labels:
                        remaining.extend(list(pending)[origin_index + 1:])
                        break
                pending = remaining
                if stats["labels"] >= max_labels:
                    break
                if last_origin_time is None or frame.timestamp - last_origin_time >= label_interval_s - 1e-9:
                    if len(pending) >= 2048:
                        pending.popleft()
                        stats["skipped_pending_overflow"] += 1
                    pending.append(current)
                    last_origin_time = frame.timestamp
                previous = current
            stats["stopped_at_label_budget"] = stats["labels"] >= max_labels
            stats["pending_targets_at_stop"] = len(pending)
            stats["skipped_no_future_target"] = 0 if stats["stopped_at_label_budget"] else len(pending)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, str(output))
        temporary = None
    finally:
        stream.close()
        if temporary is not None:
            os.remove(temporary)
    stats.update(source=str(source.resolve()), output=str(output.resolve()),
                 label_type="motor_cut_trajectory_goal", alignment=alignment,
                 budgets=dict(max_records=max_records, max_frames=max_frames, max_labels=max_labels),
                 export_options=options)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=6000)
    parser.add_argument("--motor-max-age-s", type=float, default=.5)
    parser.add_argument("--label-format", choices=("goal", "location"), default="goal",
                        help="goal (default) imports directly into goal_score_rl.py")
    parser.add_argument("--max-frames", type=int, default=800)
    parser.add_argument("--max-labels", type=int, default=128)
    parser.add_argument("--lookahead-distance", type=float, default=1.)
    parser.add_argument("--max-horizon-s", type=float, default=10.)
    parser.add_argument("--label-interval-s", type=float, default=1.)
    parser.add_argument("--max-frame-gap-s", type=float, default=.5)
    parser.add_argument("--pose-max-gap-s", type=float, default=.1)
    args = parser.parse_args()
    if args.label_format == "location":
        report = export_motor_labels(args.source, args.output, args.max_records, args.motor_max_age_s)
    else:
        report = export_goal_labels(args.source, args.output, args.max_records, args.max_frames,
                                   args.max_labels, args.lookahead_distance, args.max_horizon_s,
                                   args.label_interval_s, args.max_frame_gap_s, args.pose_max_gap_s,
                                   args.motor_max_age_s)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
