"""Manual desired exploration poses, independent of explorer decisions."""

import json
import math
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button


class GoalPoseLabels:
    """One current label per source/frame/timestamp, saved atomically as JSONL."""

    def __init__(self, path, source):
        self.path = Path(path)
        self.source = str(Path(source).resolve()) if source is not None else "unspecified"
        if source is not None and self.path.resolve() == Path(source).resolve():
            raise ValueError("Label file must differ from the replay input")
        self.records = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        key = self.key(record["source"], record["frame"], record["timestamp"])
                        if record["schema_version"] != 1:
                            raise ValueError("unsupported label schema")
                        pose = record["desired_goal_world"]
                        if not all(math.isfinite(float(pose[name])) for name in ("x", "y", "yaw")):
                            raise ValueError("nonfinite goal")
                        self.records[key] = record
                    except (ValueError, TypeError, KeyError) as error:
                        raise ValueError("Invalid goal label at line {}: {}".format(number, error)) from error

    @staticmethod
    def key(source, frame, timestamp):
        return source, int(frame), float(timestamp)

    def get(self, number, frame):
        return self.records.get(self.key(self.source, number, frame.timestamp))

    def _write(self, records):
        # A failed write leaves both the previous file and in-memory labels intact.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=str(self.path.parent), delete=False) as stream:
                temporary = stream.name
                for key in sorted(records):
                    stream.write(json.dumps(records[key], ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, str(self.path))
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.remove(temporary)
        self.records = records

    def save(self, number, frame, goal, algorithm_goal=None):
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("goal must be finite (x, y, yaw)")
        heading = float(frame.robot_yaw + math.pi / 2)
        dx, dy = goal[:2] - frame.robot_xyz[:2]
        local = (math.cos(heading) * dx + math.sin(heading) * dy,
                 -math.sin(heading) * dx + math.cos(heading) * dy,
                 math.atan2(math.sin(goal[2] - heading), math.cos(goal[2] - heading)))
        record = dict(schema_version=1, source=self.source, frame=int(number),
                      timestamp=float(frame.timestamp), clock=frame.clock,
                      coordinate_frame="world_xy", yaw_unit="radians",
                      yaw_convention="CCW from world +X; local +X is robot forward",
                      robot_pose_world=dict(x=float(frame.robot_xyz[0]), y=float(frame.robot_xyz[1]),
                                            yaw=heading),
                      desired_goal_world=dict(zip(("x", "y", "yaw"), map(float, goal))),
                      desired_goal_robot=dict(zip(("x", "y", "yaw"), map(float, local))),
                      algorithm_goal_world=(dict(zip(("x", "y", "yaw"), map(float, algorithm_goal)))
                                            if algorithm_goal is not None else None),
                      label_type="manual_desired_pose",
                      updated_at=datetime.now(timezone.utc).isoformat())
        records = self.records.copy()
        records[self.key(self.source, number, frame.timestamp)] = record
        self._write(records)
        return record

    def delete(self, number, frame):
        key = self.key(self.source, number, frame.timestamp)
        if key in self.records:
            records = self.records.copy()
            del records[key]
            self._write(records)


class GoalPoseAnnotator:
    """Two clicks in global XY select a position and a forward heading."""

    def __init__(self, fig, ax, labels, pause):
        self.fig, self.ax, self.labels, self.pause = fig, ax, labels, pause
        self.current = None
        self.active = False
        self.position = None
        self.arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                                      color="#d000d0", lw=3, mutation_scale=22, zorder=10)
        ax.add_patch(self.arrow)
        self.arrow.set_visible(False)
        self.marker = ax.scatter([], [], marker="+", s=120, c="#d000d0", zorder=10)
        self.text = fig.text(.02, .905, "", fontsize=9, color="#a000a0")
        self.button = Button(fig.add_axes((.02, .04, .13, .06)), "Label goal [A]")
        self.delete_button = Button(fig.add_axes((.84, .04, .14, .06)), "Delete label")
        self.button.on_clicked(self.begin)
        self.delete_button.on_clicked(self.delete)
        fig.canvas.mpl_connect("button_press_event", self.click)
        fig.canvas.mpl_connect("key_press_event", self.key)

    def show(self, number, frame, algorithm_goal=None):
        self.current = number, frame, algorithm_goal
        self.active = False
        self.position = None
        self.refresh(draw=False)

    def refresh(self, message=None, draw=True):
        self.arrow.set_visible(False)
        self.marker.set_offsets(np.empty((0, 2)))
        record = self.labels.get(*self.current[:2]) if self.current else None
        if record:
            pose = record["desired_goal_world"]
            start = np.array([pose["x"], pose["y"]])
            end = start + .3 * np.array([math.cos(pose["yaw"]), math.sin(pose["yaw"])])
            self.arrow.set_positions(start, end)
            self.arrow.set_visible(True)
            self.marker.set_offsets(start.reshape(1, 2))
        self.text.set_text(message or ("Manual goal saved (magenta). A: replace; Delete: remove."
                                      if record else "A: label desired goal (position, then heading)."))
        if draw:
            self.fig.canvas.draw_idle()

    def begin(self, event=None):
        if self.current is None:
            return
        self.pause()
        self.active = True
        self.position = None
        self.refresh("Click goal position in Global XY. Esc: cancel.")

    def cancel(self):
        self.active = False
        self.position = None
        self.refresh()

    def click(self, event):
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if (not self.active or event.inaxes is not self.ax or event.button != 1
                or (toolbar is not None and toolbar.mode)
                or event.xdata is None or event.ydata is None):
            return
        point = np.array([event.xdata, event.ydata], dtype=float)
        if not np.isfinite(point).all():
            return
        if self.position is None:
            self.position = point
            self.marker.set_offsets(point.reshape(1, 2))
            self.text.set_text("Click forward direction from the goal. Esc: cancel.")
            self.fig.canvas.draw_idle()
            return
        delta = point - self.position
        if np.linalg.norm(delta) < 1e-6:
            return
        pose = (*self.position, math.atan2(delta[1], delta[0]))
        try:
            self.labels.save(*self.current[:2], pose, self.current[2])
        except (OSError, ValueError) as error:
            self.text.set_text("Save failed: {}. Click again to retry.".format(error))
            self.fig.canvas.draw_idle()
            return
        self.cancel()

    def delete(self, event=None):
        if self.current is not None:
            self.pause()
            try:
                self.labels.delete(*self.current[:2])
            except OSError as error:
                self.refresh("Delete failed: {}".format(error))
                return
            self.cancel()

    def key(self, event):
        if event.key == "a":
            self.begin()
        elif event.key == "escape":
            self.cancel()
        elif event.key == "delete":
            self.delete()
