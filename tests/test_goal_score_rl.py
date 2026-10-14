"""Behavioral checks for score learning, safe integration and bounded preparation."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from boundary_exploration import BoundaryExplorer, ExplorationConfig, GoalScoreCandidate
from goal_score_rl import (GoalScoreBanditEnv, GoalScorePolicy, bandit_gradient, evaluate, features,
                           load_labels, pose_errors, prepare, split_episodes, train)


def candidate(x, y, score=1., heading=0., cell=None):
    return GoalScoreCandidate(cell or (round(x*10), round(y*10)), (x, y, heading),
        (0., 0., 0.), 3., 0., math.hypot(x, y), 0., 0, score, 2.)


def episode(number, timestamp):
    return dict(frame=number, timestamp=timestamp, desired_pose=[0., 1., 0.],
        candidates=[asdict(candidate(1., 0., 1.1)), asdict(candidate(0., 1., 1.))], eligible_count=2)


class GoalScoreRLTests(unittest.TestCase):
    def test_bandit_episode_is_single_step_and_label_free_observation(self):
        env = GoalScoreBanditEnv([episode(1, 1.)])
        observation, info = env.reset()
        self.assertEqual(set(observation), {"features", "baseline_logits", "action_mask"})
        self.assertEqual(info["frame"], 1)
        with self.assertRaises(ValueError):
            env.step(2)
        _, reward, terminated, truncated, _ = env.step(1)
        self.assertEqual(reward, 0.)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        with self.assertRaises(RuntimeError):
            env.step(1)

    def test_wrapped_yaw_error_and_robot_feature_rotation(self):
        distance, yaw = pose_errors([[0., 0., -math.pi+.01]], [0., 0., math.pi-.01])
        self.assertAlmostEqual(yaw[0], .02)
        self.assertEqual(distance[0], 0.)
        c = candidate(0., 2.)
        c = GoalScoreCandidate(**{**asdict(c), "robot_pose": (0., 0., math.pi/2)})
        self.assertAlmostEqual(features(c)[5], 1.)
        self.assertAlmostEqual(features(c)[6], 0.)

    def test_policy_gradient_matches_expected_reward_and_kl_derivative(self):
        x = np.array([[1., -.3], [.1, .8], [-.2, -.5]])
        weights, baseline, reward = np.array([.2, -.4]), np.array([.2, -.1, .8]), np.array([-.9, -.1, -2.])
        temperature, kl = .7, .13
        actual, _ = bandit_gradient(weights, x, baseline, reward, temperature, kl)

        def objective(w):
            logits = (baseline + x @ w) / temperature
            log_p = logits - np.logaddexp.reduce(logits)
            log_q = baseline / temperature
            log_q -= np.logaddexp.reduce(log_q)
            return np.exp(log_p) @ (reward - kl*(log_p-log_q))

        epsilon = 1e-6
        expected = np.array([(objective(weights + np.eye(2)[i]*epsilon)
                              - objective(weights - np.eye(2)[i]*epsilon))/(2*epsilon) for i in range(2)])
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)

    def test_chronological_gap_excludes_nearby_validation_frames(self):
        train_set, validation = split_episodes([episode(1, 1.), episode(2, 2.), episode(3, 2.1),
                                                episode(4, 5.)], train_fraction=.5, gap_s=1.)
        self.assertEqual([e["frame"] for e in train_set], [1, 2])
        self.assertEqual([e["frame"] for e in validation], [4])
        with self.assertRaises(ValueError):
            split_episodes([episode(1, 1.), episode(2, 1.1)], gap_s=1.)

    def test_training_improves_heldout_label_reward_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, model, report = (root / name for name in ("data.json", "model.json", "report.json"))
            dataset.write_text(json.dumps(dict(schema_version=1, task="label_reward_contextual_bandit",
                exploration_config={}, episodes=[episode(1, 1.), episode(2, 3.), episode(3, 6.)])))
            result = train(dataset, model, report, epochs=100)
            self.assertGreater(result["validation"]["policy_reward"], result["validation"]["baseline_reward"])
            policy = GoalScorePolicy.load(model)
            self.assertGreater(policy(candidate(0., 1.)), policy(candidate(1., 0., 1.1)))
            self.assertEqual(json.loads(report.read_text())["metadata"]["training_frames"], [1, 2])
            self.assertEqual(evaluate([episode(3, 6.)], policy)["episodes"], 1)

    def test_model_feature_version_and_nonfinite_weights_rejected(self):
        with self.assertRaises(ValueError):
            GoalScorePolicy([math.nan]*9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.json"
            GoalScorePolicy().save(path)
            record = json.loads(path.read_text())
            record["feature_names"].reverse()
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                GoalScorePolicy.load(path)

    def test_learned_goal_and_enumerated_paths_remain_safe(self):
        policy = GoalScorePolicy(np.array([0., 0., 0., 0., 0., 2., -1., 0., 0.]))
        explorer = BoundaryExplorer(ExplorationConfig(robot_radius=.1, sensor_range=1., pose_margin=0.,
                                                      boundary_clearance=.1), goal_scorer=policy)
        grass = np.array([(x*.1+.05, y*.1+.05) for x in range(-10, 11) for y in range(-10, 11)])
        pose = (.05, .05, 0.)
        explorer.update(grass, [], pose, 1.)
        result = explorer.update(grass, [], pose, 2.)
        self.assertEqual(result.status, "exploring")
        self.assertTrue(explorer.trajectory_is_safe(np.column_stack((result.path_xy, np.zeros(len(result.path_xy))))))
        snapshot = (explorer.active_goal, dict(explorer.goal_visits), explorer.last_timestamp)
        candidates = explorer.goal_candidates(pose)
        self.assertTrue(candidates)
        self.assertTrue(all(c.cell in result.safe_cells and c.baseline_score > 0 for c in candidates))
        expected = max(candidates, key=lambda c: (c.boundary_gain > 0, policy(c), -c.path_cost, c.cell))
        np.testing.assert_allclose(result.goal_pose, expected.pose)
        self.assertEqual(snapshot, (explorer.active_goal, explorer.goal_visits, explorer.last_timestamp))
        for c in candidates:
            explorer.goal_visits[c.cell] = 100
        self.assertEqual(explorer.goal_candidates(pose), [])

    def test_unknown_grass_never_becomes_action_and_invalid_scorer_rejected(self):
        explorer = BoundaryExplorer(ExplorationConfig(robot_radius=.1, sensor_range=1.),
                                    goal_scorer=GoalScorePolicy(np.full(9, 1e3)))
        self.assertEqual(explorer.goal_candidates((0., 0., 0.)), [])
        explorer.seed_verified_grass_disk((.05, .05), 1.)
        explorer.goal_scorer = lambda _: float("nan")
        with self.assertRaisesRegex(ValueError, "goal_scorer"):
            explorer.goal_candidates((.05, .05, 0.))

    def test_prepare_closes_bounded_stream_and_rejects_timestamp_mismatch(self):
        from replay_boundary import Frame
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, labels, output = root/"raw.jsonl", root/"labels.jsonl", root/"data.json"
            record = dict(schema_version=1, source=str(source), frame=2, timestamp=2.,
                coordinate_frame="world_xy", yaw_unit="radians", label_type="manual_desired_pose",
                robot_pose_world=dict(x=0., y=0., yaw=math.pi/2), desired_goal_world=dict(x=1., y=0., yaw=0.))
            labels.write_text(json.dumps(record)+"\n", encoding="utf-8")
            consumed, closed = [], []

            def stream(path, max_records, pose_gap, alignment):
                alignment["aligned_frames"] = 0
                try:
                    for i in range(1, 6):
                        consumed.append(i)
                        alignment["aligned_frames"] += 1
                        yield Frame(str(i), float(i), np.zeros(3), np.zeros(3), 0., np.empty((0, 2)),
                                    np.empty((0, 3)))
                finally:
                    closed.append(True)

            config = ExplorationConfig(robot_radius=.1, sensor_range=1.)
            with patch("replay_boundary.iter_data_package", stream):
                summary = prepare(source, labels, output, config, max_frames=1)
                self.assertEqual(consumed, [1])
                self.assertEqual(closed, [True])
                self.assertEqual(summary["episodes"], 0)
                self.assertIn("budget", summary["skipped"][0]["reason"])
                record["timestamp"] = 2.5
                labels.write_text(json.dumps(record)+"\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "alignment mismatch"):
                    prepare(source, labels, output, config, max_frames=2)

    def test_labels_source_and_prefix_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, path = root/"raw.jsonl", root/"labels.jsonl"
            label = dict(schema_version=1, source=str(source), frame=1, timestamp=1., coordinate_frame="world_xy",
                yaw_unit="radians", label_type="manual_desired_pose", robot_pose_world=dict(x=0., y=0., yaw=0.),
                desired_goal_world=dict(x=1., y=0., yaw=0.))
            path.write_text(json.dumps(label)+"\nBROKEN OUTSIDE BUDGET\n", encoding="utf-8")
            self.assertEqual(len(load_labels(path, source, max_label_records=1)), 1)
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                load_labels(path, root/"different.jsonl", max_label_records=1)


if __name__ == "__main__":
    unittest.main()
