"""Auxiliary clock domains, effective counts and measured-motor labels."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from boundary_exploration import BoundaryExplorer, ExplorationConfig
from navigation_topics import AuxiliaryTopics, export_goal_labels, export_motor_labels, parse_obstacles
from replay_boundary import iter_data_package, update_exploration, camera_to_world_matrix


def obstacle(timestamp=1100):
    return dict(timestamp=timestamp, sensorID=0, state=1, sum=1,
                vObs=[dict(sum=1, dataType=2, objectID=7, dynamic=1,
                           vInfo=[dict(cornerX=.2, cornerY=.4, cornerZ=1.),
                                  {"invalid_padding": True}]), {"invalid_padding": True}])


def location(timestamp):
    return dict(topic="location", system_ms=timestamp + 5,
                data=dict(timestamp=timestamp, poseOI=dict(
                    position=dict(east=1., north=2., up=0.),
                    attitude=dict(roll=0., pitch=0., head=0.))))


def motor(timestamp, rpm, topic="motor_cut"):
    return dict(topic=topic, system_ms=timestamp,
                data=dict(timestamp=123, rpm=rpm, status=0))


class NavigationTopicsTests(unittest.TestCase):
    @staticmethod
    def trajectory_records(count=12, step=.25):
        records = []
        grass = [dict(cornerX=float(x), cornerY=0., cornerZ=float(z))
                 for x in np.arange(-1.2, 1.21, .1) for z in np.arange(-.8, 1.81, .1)]
        for i in range(count):
            stamp = 1000 + i * 100
            pose = location(stamp)
            pose["data"]["poseOI"]["position"]["north"] += i * step
            records.extend([motor(stamp - 10, -500 if i % 2 else 500), pose,
                            dict(topic="visual_boundary_points", data=dict(
                                timestamp=stamp, boundary_num=0, grass_num=len(grass),
                                vBoundaryPoints=[], vGrassPoints=grass))])
        return records

    def test_goal_export_loads_and_prepares_without_format_conversion(self):
        from goal_score_rl import load_labels, prepare
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, labels, dataset = root / "raw.jsonl", root / "goals.jsonl", root / "dataset.json"
            source.write_text("\n".join(json.dumps(r) for r in self.trajectory_records()), encoding="utf-8")
            report = export_goal_labels(source, labels, max_records=36, max_frames=12,
                                        max_labels=3, label_interval_s=.2)
            loaded = load_labels(labels, source, max_labels=3)
            self.assertEqual(report["labels"], 3)
            self.assertEqual([r["frame"] for r in loaded], [1, 3, 5])
            for label in loaded:
                self.assertEqual(label["label_type"], "motor_cut_trajectory_goal")
                self.assertAlmostEqual(label["desired_goal_world"]["y"] - label["robot_pose_world"]["y"], 1.)
                self.assertAlmostEqual(label["desired_goal_world"]["yaw"], np.pi / 2)
                self.assertEqual(label["provenance"]["target_frame"], label["frame"] + 4)
            summary = prepare(source, labels, dataset, ExplorationConfig(.05, 2., grass_observations=1),
                              max_records=36, max_frames=12, max_labels=3)
            self.assertEqual(summary["episodes"], 3)
            prepared = json.loads(dataset.read_text())
            self.assertEqual(prepared["episodes"][0]["label_type"], "motor_cut_trajectory_goal")
            self.assertEqual(prepared["episodes"][0]["label_provenance"], loaded[0]["provenance"])
            loaded[0]["provenance"]["target_motor_cut"]["active_since"] = 999.
            labels.write_text(json.dumps(loaded[0]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_labels(labels, source)

    def test_goal_export_does_not_bridge_stop_restart_between_visual_frames(self):
        records = self.trajectory_records()
        records[12:12] = [motor(1350, 0), motor(1375, -500)]
        with tempfile.TemporaryDirectory() as folder:
            source, labels = Path(folder) / "raw.jsonl", Path(folder) / "goals.jsonl"
            source.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            report = export_goal_labels(source, labels, max_records=38, label_interval_s=.1)
            loaded = [json.loads(s) for s in labels.read_text().splitlines()]
            self.assertGreater(len(loaded), 0)
            self.assertGreater(report["skipped_discontinuity"], 0)
            self.assertTrue(all(r["frame"] >= 5 for r in loaded))

    def test_goal_export_budgets_stationary_horizon_and_atomic_failure(self):
        records = self.trajectory_records(count=8, step=0.)
        with tempfile.TemporaryDirectory() as folder:
            source, labels = Path(folder) / "raw.jsonl", Path(folder) / "goals.jsonl"
            source.write_text("\n".join(json.dumps(r) for r in records) + "\nBROKEN", encoding="utf-8")
            report = export_goal_labels(source, labels, max_records=24, max_horizon_s=.2,
                                        label_interval_s=.1)
            self.assertEqual(report["labels"], 0)
            self.assertGreater(report["skipped_horizon"], 0)
            self.assertEqual(report["alignment"]["raw_records_read"], 24)
            labels.write_text("previous", encoding="utf-8")
            with self.assertRaises(ValueError):
                export_goal_labels(source, labels, max_records=25)
            self.assertEqual(labels.read_text(), "previous")
            with self.assertRaises(ValueError):
                export_goal_labels(source, source)
            for option in (dict(lookahead_distance=0), dict(max_frames=0), dict(max_age_s=float("nan"))):
                with self.assertRaises(ValueError):
                    export_goal_labels(source, labels, **option)

    def test_active_run_identity_survives_history_trimming_and_splits_stale_gap(self):
        history = AuxiliaryTopics(capacity=2)
        for stamp in (1000, 1100, 1200, 1300):
            history.ingest(motor(stamp, 500))
        self.assertEqual(history.state_at("motor_cut", 1.3)["active_since"], 1.)
        history.ingest(motor(1900, -500))
        self.assertEqual(history.state_at("motor_cut", 1.9)["active_since"], 1.9)

    def test_counts_exclude_padding_and_preserve_objects(self):
        observation = parse_obstacles(obstacle())
        self.assertEqual(len(observation.objects), 1)
        obj = observation.objects[0]
        self.assertEqual((obj.data_type, obj.object_id, obj.dynamic), (2, 7, 1))
        np.testing.assert_allclose(observation.camera_xyz, [[.2, .4, 1.]])
        data = obstacle()
        data["sum"] = 3
        with self.assertRaises(ValueError):
            parse_obstacles(data)
        data = obstacle()
        data["vObs"][0]["sum"] = -1
        with self.assertRaises(ValueError):
            parse_obstacles(data)

    def test_clock_domain_causality_staleness_and_bounds(self):
        history = AuxiliaryTopics(capacity=2)
        for stamp, rpm in ((1000, -100), (1100, 0), (1200, 200)):
            history.ingest(motor(stamp, rpm))
        self.assertEqual(len(history.states["motor_cut"]), 2)
        self.assertIsNone(history.state_at("motor_cut", 1.05))
        self.assertEqual(history.state_at("motor_cut", 1.15)["rpm"], 0)
        self.assertEqual(history.state_at("motor_cut", 1.2)["rpm"], 200)
        self.assertIsNone(history.state_at("motor_cut", 1.701))

    def test_invalid_auxiliary_timestamp_is_skipped_without_stale_reuse(self):
        history = AuxiliaryTopics()
        self.assertFalse(history.ingest(dict(topic="visual_obstacle_front", data=obstacle(0))))
        self.assertEqual(history.obstacles, {})
        self.assertFalse(history.ingest(dict(topic="motor_cut", data=dict(rpm=10))))
        empty = obstacle()
        empty["sum"] = 0
        self.assertEqual(parse_obstacles(empty).camera_xyz.shape, (0, 3))

    def test_export_measured_positive_and_negative_rpm_only(self):
        records = [location(900), motor(1000, -500), motor(1000, 0, "cut_ctrl"),
                   location(1050), motor(1100, 0), location(1150),
                   motor(1200, 500), location(1250), location(1800),
                   motor(1900, float("nan"))]
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source.jsonl", Path(folder) / "labels.jsonl"
            source.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            stats = export_motor_labels(source, target, max_records=9)
            labels = [json.loads(s) for s in target.read_text().splitlines()]
            self.assertEqual([r["record"] for r in labels], [4, 8])
            self.assertEqual([r["motor_cut"]["rpm"] for r in labels], [-500, 500])
            self.assertEqual(labels[0]["cut_ctrl"]["rpm"], 0)
            self.assertEqual(stats["location_unknown_motor"], 2)
            self.assertEqual(stats["location_inactive_motor"], 1)
            self.assertEqual(stats["raw_records_read"], 9)
            previous = target.read_bytes()
            with self.assertRaises(ValueError):
                export_motor_labels(source, target, max_records=10)
            self.assertEqual(target.read_bytes(), previous)
            with self.assertRaises(ValueError):
                export_motor_labels(source, source)

    def test_replay_passes_same_sample_obstacles_and_clears_missing(self):
        records = [motor(1000, -10), motor(1000, 100, "cut_ctrl"), location(1100),
                   dict(topic="visual_obstacle_front", data=obstacle()),
                   dict(topic="visual_boundary_points", data=dict(
                       timestamp=1100, boundary_num=0, grass_num=0,
                       vBoundaryPoints=[], vGrassPoints=[])),
                   location(1200), dict(topic="visual_boundary_points", data=dict(
                       timestamp=1200, boundary_num=0, grass_num=0,
                       vBoundaryPoints=[], vGrassPoints=[]))]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.jsonl"
            source.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            stats = {}
            frames = list(iter_data_package(source, alignment_stats=stats))
        first, second = frames
        self.assertTrue(first.training_label)
        self.assertEqual(first.cut_ctrl["rpm"], 100)
        explorer = BoundaryExplorer(ExplorationConfig(.1, 2.))
        update_exploration(explorer, first)
        expected = np.array([[.2, .4, 1., 1.]]) @ camera_to_world_matrix(first).T
        np.testing.assert_allclose(explorer.obstacle_world_xy, expected[:, :2])
        self.assertEqual(explorer.visual_obstacle_front.objects[0].object_id, 7)
        self.assertEqual(explorer.boundary_hits, {})
        update_exploration(explorer, second)
        self.assertIsNone(explorer.visual_obstacle_front)
        self.assertEqual(explorer.obstacle_world_xy.shape, (0, 2))
        self.assertEqual(stats["frames_with_obstacles"], 1)


if __name__ == "__main__":
    unittest.main()
