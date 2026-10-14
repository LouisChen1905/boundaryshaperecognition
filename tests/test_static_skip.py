import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from boundary_exploration import ExplorationConfig
from replay_boundary import Frame, replay_exploration_stream, same_robot_pose


def make_frame(x=0., yaw=0., timestamp=1., roll=0.):
    return Frame("00:00:01", timestamp, np.array([x, 0., 0.]),
                 np.array([roll, 0., yaw]), yaw, np.empty((0, 2)),
                 np.array([[1., 1., 0.]]), grass_world_xy=np.array([[.05, .05]]))


class ManualTimer:
    def __init__(self):
        self.running = False
    def add_callback(self, callback):
        self.callback = callback
    def start(self):
        self.running = True
    def stop(self):
        self.running = False


class StaticSkipTests(unittest.TestCase):
    def test_playback_subtracts_callback_time_from_wait(self):
        import time
        timers = []
        def new_timer(*args, **kwargs):
            timer = ManualTimer()
            timers.append(timer)
            return timer
        def slow_draw(canvas, *args, **kwargs):
            time.sleep(.02)
        def interact():
            timers[0].callback()
            # Frames are 10 ms apart but drawing alone takes 20 ms.
            self.assertEqual(timers[0].interval, 1)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(FigureCanvasAgg, "new_timer", side_effect=new_timer), patch.object(FigureCanvasAgg, "draw", slow_draw), patch.object(plt, "show", side_effect=interact):
                replay_exploration_stream(
                    [make_frame(timestamp=1.), make_frame(timestamp=1.01)],
                    1., ExplorationConfig(.1, 2.),
                    performance_log=Path(folder) / "timing.csv")
        plt.close("all")

    def test_wrapped_attitude_and_anchor_drift(self):
        self.assertTrue(same_robot_pose(make_frame(yaw=np.pi-.01),
                                       make_frame(yaw=-np.pi+.01)))
        self.assertFalse(same_robot_pose(make_frame(), make_frame(roll=.1)))
        self.assertFalse(same_robot_pose(make_frame(), make_frame(x=.031)))
        self.assertTrue(same_robot_pose(make_frame(x=.02), make_frame(x=.031)))

    def test_skip_fuses_static_frames_and_stops_at_first_moving_frame(self):
        timers = []
        consumed = []
        frames = [make_frame(x=x, timestamp=i+1.) for i, x in enumerate((0., .01, .02, .031, .5))]
        def source():
            for f in frames:
                consumed.append(f)
                yield f
        def new_timer(*args, **kwargs):
            timer = ManualTimer()
            timers.append(timer)
            return timer
        def interact():
            canvas = plt.gcf().canvas
            canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key="s"))
            for _ in range(10):
                if not timers[1].running:
                    break
                timers[1].callback()
            self.assertFalse(timers[0].running)
            self.assertFalse(timers[1].running)
            self.assertIn("Frame 4", plt.gcf().axes[0].get_title())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "performance.csv"
            with patch.object(FigureCanvasAgg, "new_timer", side_effect=new_timer), patch.object(plt, "show", side_effect=interact):
                replay_exploration_stream(source(), 1., ExplorationConfig(.1, 2.), performance_log=path)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(consumed), 4)
        self.assertEqual([row["frame"] for row in rows], ["1", "2", "3", "4"])
        self.assertEqual([row["observation_fused"] for row in rows], ["1"]*4)
        self.assertTrue(all(float(row["render_ms"]) == 0 for row in rows[1:]))
        self.assertEqual(rows[-1]["known_grass_cells"], "1")
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
