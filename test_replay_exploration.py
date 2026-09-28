"""Integration checks for incremental JSONL exploration replay."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from boundary_exploration import BoundaryExplorer, ExplorationConfig
from replay_boundary import (camera_to_world_matrix, iter_data_package,
                             replay_exploration_stream, update_exploration)


class ReplayExplorationTests(unittest.TestCase):
    @staticmethod
    def _records():
        location = {"topic": "location", "system_ms": 1000,
                    "data": {"poseOI": {
                        "position": {"east": 1, "north": 2, "up": 0},
                        "attitude": {"roll": 0, "pitch": 0, "head": 0}}}}
        point = {"cornerX": 0.2, "cornerY": 0.4, "cornerZ": 1.0}
        visual = {"topic": "visual_boundary_points", "system_ms": 1100,
                  "data": {"timestamp": 0, "boundary_num": 1,
                           "grass_num": 1, "vBoundaryPoints": [point],
                           "vGrassPoints": [point]}}
        return location, visual

    def test_iterator_limits_records_and_retains_camera_y(self):
        location, visual = self._records()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "small.jsonl"
            with path.open("w", encoding="utf-8") as target:
                for record in (location, visual, visual,
                               {"topic": "visual_boundary_points", "data": {}}):
                    target.write(json.dumps(record) + "\n")
            frames = list(iter_data_package(path, max_records=3))
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].timestamp, 1.1)
        self.assertEqual(frames[0].camera_boundary_xyz.shape, (1, 3))
        expected = np.array([0.2, 0.4, 1.0, 1.0]) @ camera_to_world_matrix(frames[0]).T
        np.testing.assert_allclose(frames[0].world_points[0], expected[:3])
        explorer = BoundaryExplorer(ExplorationConfig(0.1, 2.0, grass_observations=1))
        result = update_exploration(explorer, frames[0])
        self.assertIn(explorer.cell(expected[:2]), result.boundary_cells)

    def test_replay_draws_first_frame_without_consuming_following_frames(self):
        location, visual = self._records()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "small.jsonl"
            with path.open("w", encoding="utf-8") as target:
                for record in (location, visual, visual):
                    target.write(json.dumps(record) + "\n")
            frames = iter_data_package(path)
            with patch.object(plt, "show"):
                replay_exploration_stream(frames, 1.0,
                                          ExplorationConfig(0.1, 2.0))
            figure = plt.gcf()
            self.assertIn("Frame 1", figure.axes[0].get_title())
            self.assertIn("waiting_for_observation",
                          figure.axes[0].get_title())
            self.assertEqual(len(figure.axes[0].collections[2].get_paths()), 1)
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
