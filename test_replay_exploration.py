"""Integration checks for incremental JSONL exploration replay."""

import json
import csv
import copy
import sys
import warnings
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
                             main,
                             _interpolate_attitude, rotation,
                             profile_exploration_stream,
                             replay_exploration_stream, update_exploration)


class ReplayExplorationTests(unittest.TestCase):
    @staticmethod
    def _records():
        location = {"topic": "location", "system_ms": 1000,
                    "data": {"timestamp": 1100, "poseOI": {
                        "position": {"east": 1, "north": 2, "up": 0},
                        "attitude": {"roll": 0, "pitch": 0, "head": 0}}}}
        point = {"cornerX": 0.2, "cornerY": 0.4, "cornerZ": 1.0}
        visual = {"topic": "visual_boundary_points", "system_ms": 1100,
                  "data": {"timestamp": 1100, "boundary_num": 1,
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
            timing_path = Path(folder) / "timing.csv"
            with patch.object(plt, "show"):
                replay_exploration_stream(frames, 1.0,
                                          ExplorationConfig(0.1, 2.0),
                                          performance_log=timing_path)
            figure = plt.gcf()
            self.assertIn("Frame 1", figure.axes[0].get_title())
            self.assertIn("waiting_for_observation",
                          figure.axes[0].get_title())
            self.assertEqual(len(figure.axes[0].collections[2].get_paths()), 1)
            with timing_path.open("r", encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 1)
            self.assertGreater(float(rows[0]["render_ms"]), 0)
            self.assertGreater(float(rows[0]["update_ms"]), 0)
            plt.close(figure)

    def test_headless_profile_writes_every_visual_frame(self):
        location, visual = self._records()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "small.jsonl"
            timing_path = Path(folder) / "timing.csv"
            with path.open("w", encoding="utf-8") as target:
                for record in (location, visual, visual):
                    target.write(json.dumps(record) + "\n")
            count, totals, maximum = profile_exploration_stream(
                iter_data_package(path), ExplorationConfig(0.1, 2.0), timing_path)
            with timing_path.open("r", encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(count, 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["render_ms"], "")
        self.assertGreater(totals["raster_ms"], 0)
        self.assertGreater(maximum, 0)
        self.assertEqual([row["observation_fused"] for row in rows], ["1", "0"])
        self.assertEqual(rows[0]["map_version"], rows[1]["map_version"])
        for name in ("removed_boundary_cells", "goal_action", "goal_release_reason",
                     "goal_visit_count", "pose_alignment", "pose_before_timestamp",
                     "pose_after_timestamp", "pose_gap_ms"):
            self.assertIn(name, rows[0])
        self.assertEqual(rows[0]["pose_alignment"], "exact")
        self.assertEqual(float(rows[0]["pose_gap_ms"]), 0.)

    def test_replay_skips_backwards_packets_and_keeps_duplicates_without_fusing(self):
        location, visual = self._records()
        records = [location]
        for timestamp in (1200, 1300):
            record = copy.deepcopy(location)
            record["data"]["timestamp"] = timestamp
            records.append(record)
        for timestamp in (1200, 1100, 1200, 1300):
            record = copy.deepcopy(visual)
            record["data"]["timestamp"] = timestamp
            records.append(record)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "small.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            with self.assertWarnsRegex(RuntimeWarning, "backwards_visual"):
                frames = list(iter_data_package(path))
        self.assertEqual([f.timestamp for f in frames], [1.2, 1.2, 1.3])
        explorer = BoundaryExplorer(ExplorationConfig(.1, 2.))
        fused = []
        for frame in frames:
            metrics = {}
            update_exploration(explorer, frame, metrics)
            fused.append(metrics["observation_fused"])
        self.assertEqual(fused, [1, 0, 1])
        self.assertTrue(all(hits == 2 for hits in explorer.boundary_hits.values()))

    def test_invalid_sensor_timestamps_never_use_system_time(self):
        location, visual = self._records()
        for timestamp in (None, 0, -1, float("nan"), float("inf"), "bad"):
            with self.subTest(timestamp=timestamp), tempfile.TemporaryDirectory() as folder:
                record = copy.deepcopy(visual)
                record["data"]["timestamp"] = timestamp
                path = Path(folder) / "small.jsonl"
                path.write_text("\n".join(json.dumps(r) for r in (location, record)), encoding="utf-8")
                stats = {}
                with self.assertWarnsRegex(RuntimeWarning, "invalid_timestamp"):
                    frames = list(iter_data_package(path, alignment_stats=stats))
                self.assertEqual(frames, [])
                self.assertEqual(stats["skipped_invalid_timestamp"], 1)

    def test_cli_passes_lifecycle_configuration_to_replay(self):
        arguments = ["replay_boundary.py", "synthetic.jsonl", "--goal-reached-distance", "0.2",
                     "--goal-progress-timeout-s", "8", "--goal-progress-distance", "0.05",
                     "--boundary-clear-observations", "4", "--goal-refresh-interval-s", "3",
                     "--goal-refresh-distance", "0.4", "--goal-refresh-yaw-deg", "30"]
        with patch.object(sys, "argv", arguments), patch("replay_boundary.replay_exploration_stream") as replay:
            main()
        config = replay.call_args[0][2]
        self.assertEqual(config.goal_reached_distance, .2)
        self.assertEqual(config.goal_progress_timeout_s, 8.)
        self.assertEqual(config.goal_progress_distance, .05)
        self.assertEqual(config.boundary_clear_observations, 4)
        self.assertEqual(config.goal_refresh_interval_s, 3.)
        self.assertEqual(config.goal_refresh_distance, .4)
        self.assertEqual(config.goal_refresh_yaw_deg, 30.)

    def location(self, timestamp, x=0., yaw=0.):
        location, _ = self._records()
        location["data"]["timestamp"] = timestamp
        location["system_ms"] = 99999999  # Arrival time must never select the pose.
        location["data"]["poseOI"]["position"] = {"east": x, "north": 0., "up": 0.}
        location["data"]["poseOI"]["attitude"] = {"roll": 0., "pitch": 0., "head": yaw}
        return location

    def visual(self, timestamp):
        _, visual = self._records()
        visual["data"]["timestamp"] = timestamp
        visual["system_ms"] = 88888888
        return visual

    def aligned(self, records, **options):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "aligned.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            return list(iter_data_package(path, **options))

    def test_delayed_cloud_interpolates_sample_time_not_latest_pose(self):
        frames = self.aligned([
            self.location(1000, 0., np.deg2rad(170)),
            self.location(1040, 4., np.deg2rad(-170)),
            self.location(1100, 100., 0.), self.visual(1020)])
        frame = frames[0]
        self.assertEqual(frame.timestamp, 1.02)
        np.testing.assert_allclose(frame.robot_xyz, [2., 0., 0.])
        self.assertAlmostEqual(abs(frame.robot_yaw), np.pi)
        self.assertEqual(frame.pose_alignment, "interpolated")
        self.assertEqual((frame.pose_before_timestamp, frame.pose_after_timestamp), (1., 1.04))
        # At yaw pi, the entire camera transform (including camera Y) rotates
        # around the interpolated robot position, not the newest pose at x=100.
        camera_transform = camera_to_world_matrix(frame)
        expected = np.column_stack((frame.camera_boundary_xyz, np.ones(1))) @ camera_transform.T
        np.testing.assert_allclose(frame.world_points, expected[:, :3])
        expected_grass = np.column_stack((frame.camera_grass_xyz, np.ones(1))) @ camera_transform.T
        np.testing.assert_allclose(frame.grass_world_xy, expected_grass[:, :2])

    def test_waits_for_future_bracket_and_does_not_read_following_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "aligned.jsonl"
            records = [self.location(1000), self.visual(1020), self.location(1040, 4.)]
            path.write_text("\n".join(json.dumps(r) for r in records) + "\nNOT JSON", encoding="utf-8")
            frames = iter_data_package(path)
            try:
                frame = next(frames)
                np.testing.assert_allclose(frame.robot_xyz, [2., 0., 0.])
            finally:
                frames.close()

    def test_record_limit_does_not_read_extra_pose_or_extrapolate(self):
        records = [self.location(1000), self.visual(1020), self.location(1040, 4.)]
        stats = {}
        with self.assertWarnsRegex(RuntimeWarning, "no_future_pose"):
            frames = self.aligned(records, max_records=2, alignment_stats=stats)
        self.assertEqual(frames, [])
        self.assertEqual(stats["location_packets"], 1)
        self.assertEqual(stats["skipped_no_future_pose"], 1)

    def test_missing_or_stale_pose_is_skipped_and_gap_is_configurable(self):
        records = [self.location(1000), self.location(1300, 3.), self.visual(1100)]
        with self.assertWarnsRegex(RuntimeWarning, "pose_gap"):
            self.assertEqual(self.aligned(records), [])
        frame = self.aligned(records, pose_max_gap_s=.5)[0]
        self.assertAlmostEqual(frame.robot_xyz[0], 1.)
        with self.assertWarnsRegex(RuntimeWarning, "before_pose_history"):
            self.assertEqual(self.aligned([self.location(1100), self.visual(1000)]), [])
        location = self.location(1000)
        del location["data"]["timestamp"]
        stats = {}
        with warnings.catch_warnings(record=True):
            self.assertEqual(self.aligned([location, self.visual(1000)], alignment_stats=stats), [])
        self.assertEqual(stats["skipped_invalid_timestamp"], 1)

    def test_out_of_order_location_history_and_exact_match(self):
        frames = self.aligned([self.location(1040, 4.), self.location(1000), self.visual(1020),
                               self.visual(1040)])
        self.assertEqual([frame.pose_alignment for frame in frames], ["interpolated", "exact"])
        np.testing.assert_allclose(frames[0].robot_xyz, [2., 0., 0.])
        np.testing.assert_allclose(frames[1].robot_xyz, [4., 0., 0.])

    def test_attitude_interpolation_preserves_full_rotation_and_singularity(self):
        first, second = np.array([.4, -.2, 2.8]), np.array([-.3, .5, -2.9])
        np.testing.assert_allclose(rotation(*_interpolate_attitude(first, second, 0.)),
                                   rotation(*first), atol=1e-12)
        np.testing.assert_allclose(rotation(*_interpolate_attitude(first, second, 1.)),
                                   rotation(*second), atol=1e-12)
        r0, r1 = rotation(*first), rotation(*second)
        mid = rotation(*_interpolate_attitude(first, second, .5))
        left, right = r0.T @ mid, mid.T @ r1
        np.testing.assert_allclose(left, right, atol=1e-12)
        singular = np.array([.3, np.pi / 2, .7])
        np.testing.assert_allclose(rotation(*_interpolate_attitude(singular, singular, .5)),
                                   rotation(*singular), atol=1e-8)

    def test_alignment_rejects_invalid_gap_limit(self):
        for gap in (0, -1, float("nan"), float("inf")):
            with self.subTest(gap=gap), self.assertRaises(ValueError):
                self.aligned([], pose_max_gap_s=gap)


if __name__ == "__main__":
    unittest.main()
