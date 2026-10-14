import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
import numpy as np

from goal_pose_labels import GoalPoseLabels, GoalPoseAnnotator
from boundary_exploration import ExplorationConfig
from replay_boundary import Frame, replay_exploration_stream


def frame(timestamp=1.0):
    return SimpleNamespace(timestamp=timestamp, clock="00:00:01",
                           robot_xyz=np.array([2., 3., 0.]), robot_yaw=0.)


class GoalLabelTests(unittest.TestCase):
    def test_roundtrip_replace_and_delete_preserve_other_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.jsonl"
            labels = GoalPoseLabels(path, "capture.jsonl")
            labels.save(1, frame(), (2., 4., math.pi / 2), (3., 4., 0.))
            labels.save(2, frame(2.), (5., 6., 0.))
            record = GoalPoseLabels(path, "capture.jsonl").get(1, frame())
            np.testing.assert_allclose(list(record["desired_goal_robot"].values()), [1., 0., 0.], atol=1e-12)
            self.assertEqual(record["algorithm_goal_world"]["x"], 3.)
            labels.save(1, frame(), (7., 8., 0.))
            self.assertEqual(len(labels.records), 2)
            labels.delete(1, frame())
            restored = GoalPoseLabels(path, "capture.jsonl")
            self.assertIsNone(restored.get(1, frame()))
            self.assertIsNotNone(restored.get(2, frame(2.)))
            self.assertIsNone(restored.get(2, frame(3.)))

    def test_failed_save_preserves_file_and_existing_label(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.jsonl"
            labels = GoalPoseLabels(path, "capture.jsonl")
            labels.save(1, frame(), (2., 4., 0.))
            before = path.read_bytes()
            with patch("goal_pose_labels.os.replace", side_effect=OSError("disk unavailable")):
                with self.assertRaises(OSError):
                    labels.save(1, frame(), (9., 9., 0.))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(labels.get(1, frame())["desired_goal_world"]["x"], 2.)
            with self.assertRaises(ValueError):
                GoalPoseLabels(path, path)
            with self.assertRaises(ValueError):
                replay_exploration_stream(iter(()), 1., ExplorationConfig(.1, 2.),
                                          performance_log=path, labels_path=path)
            self.assertEqual(path.read_bytes(), before)

    def test_two_clicks_save_paused_frame_and_navigation_cancels_draft(self):
        with tempfile.TemporaryDirectory() as folder:
            labels = GoalPoseLabels(Path(folder) / "labels.jsonl", "capture.jsonl")
            fig, ax = plt.subplots()
            pause = Mock()
            annotator = GoalPoseAnnotator(fig, ax, labels, pause)
            click = lambda x, y: SimpleNamespace(inaxes=ax, button=1, xdata=x, ydata=y)
            annotator.show(1, frame(), (3., 4., 0.))
            annotator.begin()
            pause.assert_called_once()
            annotator.click(click(2., 4.))
            annotator.click(click(2., 5.))
            self.assertAlmostEqual(labels.get(1, frame())["desired_goal_world"]["yaw"], math.pi / 2)
            self.assertFalse(annotator.active)
            annotator.begin()
            annotator.click(click(9., 9.))
            annotator.show(2, frame(2.))
            annotator.click(click(10., 9.))
            self.assertIsNone(labels.get(2, frame(2.)))
            annotator.show(1, frame())
            self.assertTrue(annotator.arrow.get_visible())
            annotator.delete()
            self.assertIsNone(labels.get(1, frame()))
            plt.close(fig)

    def test_replay_keyboard_and_mouse_label_the_visible_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "labels.jsonl"
            frames = [Frame("00:00:01", timestamp, np.zeros(3), np.zeros(3), 0.,
                            np.empty((0, 2)), np.empty((0, 3))) for timestamp in (1., 2.)]

            def interact():
                fig = plt.gcf()
                canvas = fig.canvas
                canvas.draw()
                canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key="a"))
                for point in ((.2, .2), (.2, .5)):
                    x, y = fig.axes[0].transData.transform(point)
                    canvas.callbacks.process("button_press_event",
                                             MouseEvent("button_press_event", canvas, x, y, button=1))
                canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key="right"))

            with patch.object(plt, "show", side_effect=interact):
                replay_exploration_stream(iter(frames), 1., ExplorationConfig(.1, 2.),
                                          labels_path=path, source_path="capture.jsonl")
            labels = GoalPoseLabels(path, "capture.jsonl")
            self.assertIsNotNone(labels.get(1, frames[0]))
            self.assertIsNone(labels.get(2, frames[1]))
            self.assertIn("Frame 2", plt.gcf().axes[0].get_title())
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
