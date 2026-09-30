"""Behavioral checks for the independent grass-only boundary explorer."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from boundary_exploration import (BoundaryExplorer, ExplorationConfig,
                                  points_to_world_xy)


class BoundaryExplorerTests(unittest.TestCase):
    @staticmethod
    def explorer():
        return BoundaryExplorer(ExplorationConfig(robot_radius=0.1,
                                                  sensor_range=1.0,
                                                  pose_margin=0.0,
                                                  boundary_clearance=0.1))

    @staticmethod
    def grass_square(low=-1.0, high=1.0):
        values = np.arange(low, high + 0.001, 0.1)
        return np.array([(x + 0.05, y + 0.05) for x in values for y in values])

    def test_unknown_remains_blocked_until_repeated_grass_observation(self):
        explorer = self.explorer()
        grass = self.grass_square()
        first = explorer.update(grass, np.empty((0, 2)), (0., 0., 0.), 0.0)
        self.assertEqual(first.status, "waiting_for_observation")
        self.assertEqual(explorer.cell_state(explorer.cell((0., 0.))), "unknown")
        second = explorer.update(grass, np.empty((0, 2)), (0., 0., 0.), 0.1)
        self.assertEqual(second.status, "exploring")
        self.assertEqual(explorer.cell_state(explorer.cell((0., 0.))), "grass")
        self.assertEqual(explorer.cell_state(explorer.cell((5., 5.))), "unknown")
        self.assertTrue(all(explorer.cell(point) in second.safe_cells
                            for point in second.path_xy))
        self.assertTrue(explorer.trajectory_is_safe(np.array([[0., 0., 0.],
                                                               [0.2, 0., 0.]])))
        self.assertFalse(explorer.trajectory_is_safe(np.array([[0., 0., 0.],
                                                                [2., 0., 0.]])))
        self.assertTrue(explorer.command_is_safe((0., 0., 0.), 0.1, 0., 1.0))
        self.assertFalse(explorer.command_is_safe((0., 0., 0.), 1.0, 0., 2.0))

    def test_verified_start_patch_is_explicit_and_never_implied_by_pose(self):
        explorer = self.explorer()
        empty = np.empty((0, 2))
        waiting = explorer.update(empty, empty, (0., 0., 0.), 0.0)
        self.assertEqual(waiting.status, "waiting_for_observation")
        self.assertEqual(explorer.update([], [], (0., 0., 0.), 0.1).status,
                         "waiting_for_observation")
        explorer.seed_verified_grass_disk((0., 0.), 0.4)
        self.assertEqual(explorer.cell_state(explorer.cell((0., 0.))), "grass")
        self.assertTrue(explorer.trajectory_is_safe(np.array([[0., 0., 0.]])))
        self.assertEqual(explorer.cell_state(explorer.cell((1., 0.))), "unknown")

    def test_boundary_blocks_crossing_and_large_point_gaps_remain_unknown(self):
        explorer = self.explorer()
        grass = self.grass_square(-1.0, 1.5)
        # The gap in this ordered sample is too large to draw a line across.
        boundary = np.array([[-0.5, 0.], [-0.4, 0.], [0.4, 0.], [0.5, 0.]])
        explorer.update(grass, boundary, (0., -0.5, 0.), 0.0)
        result = explorer.update(grass, boundary, (0., -0.5, 0.), 0.1)
        self.assertEqual(explorer.cell_state(explorer.cell((-0.45, 0.))), "boundary")
        self.assertNotEqual(explorer.cell_state(explorer.cell((0., 0.))), "boundary")
        self.assertTrue(all(cell not in result.safe_cells for cell in result.boundary_cells))
        self.assertFalse(explorer.trajectory_is_safe(np.array([[-0.5, -0.5, 0.],
                                                                [-0.5, 0.5, 0.]])))

    def test_disconnected_grass_island_is_not_a_route(self):
        explorer = self.explorer()
        grass = np.vstack((self.grass_square(-1., 1.),
                           self.grass_square(3., 4.)))
        explorer.update(grass, np.empty((0, 2)), (0., 0., 0.), 0.0)
        result = explorer.update(grass, np.empty((0, 2)), (0., 0., 0.), 0.1)
        self.assertIsNotNone(result.goal_pose)
        self.assertTrue(all(point[0] < 2. for point in result.path_xy))

    def test_confirmed_boundary_loop_reports_complete(self):
        explorer = self.explorer()
        grass = self.grass_square(-0.8, 0.8)
        v = np.linspace(-1., 1., 21)
        boundary = np.vstack((np.column_stack((v, np.full_like(v, -1.))),
                              np.column_stack((np.full_like(v, 1.), v)),
                              np.column_stack((v[::-1], np.full_like(v, 1.))),
                              np.column_stack((np.full_like(v, -1.), v[::-1]))))
        explorer.update(grass, boundary, (0., 0., 0.), 0.0)
        result = explorer.update(grass, boundary, (0., 0., 0.), 0.1)
        self.assertTrue(result.boundary_closed)
        self.assertEqual(result.status, "complete")
        self.assertIsNone(result.goal_pose)

    def test_camera_y_is_used_during_full_3d_transform(self):
        transform = np.eye(4)
        transform[0, 1] = 1.0
        result = points_to_world_xy(np.array([[1., 0.2, 2.]]), transform)
        np.testing.assert_allclose(result, [[1.2, 0.2]])
        with self.assertRaises(ValueError):
            points_to_world_xy(np.array([[1., 0.2, 2.]]))

    def test_fast_sight_check_matches_rasterized_visibility(self):
        explorer = self.explorer()
        random = np.random.RandomState(7)
        for _ in range(300):
            start = explorer.center(tuple(random.randint(-12, 13, 2)))
            end = explorer.center(tuple(random.randint(-12, 13, 2)))
            blocked = {tuple(cell) for cell in random.randint(-12, 13, (15, 2))}
            expected = not (explorer._raster_segment(start, end) & blocked)
            self.assertEqual(explorer._sight_clear(start, end, blocked), expected)

    def test_visibility_cache_reuses_clearance_and_invalidates_on_new_boundary(self):
        explorer = self.explorer()
        grass = self.grass_square()
        empty = np.empty((0, 2))
        explorer.update(grass, empty, (0., 0., 0.), 0.0)
        first = {}
        explorer.update(grass, empty, (0., 0., 0.), 0.1, performance=first)
        self.assertGreater(first["sight_checks"], 0)
        self.assertEqual(first["sight_cache_hits"], 0)
        repeated = {}
        explorer.update(grass, empty, (0., 0., 0.), 0.2, performance=repeated)
        self.assertEqual(repeated["sight_cache_hits"], repeated["sight_checks"])
        changed = {}
        explorer.update(grass, np.array([[0.8, 0.8]]), (0., 0., 0.), 0.3,
                        performance=changed)
        self.assertEqual(changed["sight_cache_hits"], 0)

    def test_path_radius_configuration_requires_increasing_positive_finite_values(self):
        for radii in ((), (0, 1), (2, 1), (1, 1), (1, float("inf")),
                      (float("nan"),), None):
            with self.subTest(radii=radii), self.assertRaises(ValueError):
                ExplorationConfig(0.1, 1.0, path_radii=radii)
        for gain in (0, -1, float("inf"), float("nan")):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                ExplorationConfig(0.1, 1.0, min_visible_gain=gain)

    def test_boundary_preference_configuration(self):
        for name, values in (("boundary_frontier_distance", (0, -1, float("nan"), float("inf"))),
                             ("boundary_distance_weight", (-1, float("nan"), float("inf")))):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ExplorationConfig(0.1, 1.0, **{name: value})

    def test_boundary_frontier_expands_past_local_generic_frontier(self):
        explorer = BoundaryExplorer(ExplorationConfig(
            0.1, 1.1, resolution=1.0, goal_spacing=0.1,
            boundary_frontier_distance=1.0, min_visible_gain=1.0))
        safe = {(x, 0) for x in range(7)}
        performance = {}
        with patch.object(explorer, "_frontiers", return_value={(1, 0), (6, 0)}):
            goal, path, _ = explorer._goal((0.5, 0.5, 0.), safe, {(6, 1)}, safe,
                                          performance)
        self.assertGreaterEqual(explorer.cell(goal[:2])[0], 5)
        self.assertEqual(performance["goal_boundary_gain"], 2.0)
        self.assertEqual(performance["goal_search_radius"], "full")
        self.assertTrue(all(explorer.cell(point) in safe for point in path))

    def test_boundary_distance_penalty_moves_goal_closer_to_line(self):
        goals = []
        for weight in (0.0, 2.0):
            explorer = BoundaryExplorer(ExplorationConfig(
                0.1, 5.0, resolution=1.0, goal_spacing=0.1, path_radii=(4.0,),
                boundary_frontier_distance=1.0, boundary_distance_weight=weight))
            safe = {(x, 0) for x in range(5)}
            with patch.object(explorer, "_frontiers", return_value={(4, 0)}):
                goal, path, _ = explorer._goal((0.5, 0.5, 0.), safe, {(4, 1)}, safe)
            goals.append(explorer.cell(goal[:2]))
            self.assertTrue(all(explorer.cell(point) in safe for point in path))
        self.assertEqual(goals[0], (1, 0))
        self.assertGreater(goals[1][0], goals[0][0])

    def test_occluded_boundary_frontier_falls_back_to_generic_view(self):
        explorer = BoundaryExplorer(ExplorationConfig(
            0.1, 5.0, resolution=1.0, goal_spacing=0.1,
            boundary_frontier_distance=1.0))
        safe = {(0, 0), (1, 0)}
        boundary = {(2, y) for y in range(-3, 4)} | {(4, 1)}
        performance = {}
        with patch.object(explorer, "_frontiers", return_value={(0, 1), (4, 0)}):
            goal, path, _ = explorer._goal((0.5, 0.5, 0.), safe, boundary, safe,
                                          performance)
        self.assertIsNotNone(goal)
        self.assertEqual(performance["boundary_frontier_cells"], 1)
        self.assertEqual(performance["goal_boundary_gain"], 0)
        self.assertTrue(all(explorer.cell(point) in safe for point in path))

    def test_heading_prioritizes_boundary_and_spans_angle_wrap(self):
        explorer = BoundaryExplorer(ExplorationConfig(0.1, 5.0, sensor_fov_deg=40))
        def target(degrees, weight):
            angle = math.radians(degrees)
            return ((math.cos(angle), math.sin(angle)), weight)
        visible = [target(-170, 2.0), target(170, 2.0)]
        visible += [target(degrees, 1.0) for degrees in range(-10, 11)]
        heading, gain, boundary_gain = explorer._view_heading((0., 0.), visible)
        self.assertAlmostEqual(abs(heading), math.pi)
        self.assertEqual((gain, boundary_gain), (4.0, 4.0))
        explorer = BoundaryExplorer(ExplorationConfig(0.1, 5.0, sensor_fov_deg=360))
        _, gain, boundary_gain = explorer._view_heading((0., 0.), visible)
        self.assertEqual((gain, boundary_gain), (25.0, 4.0))

    def test_search_stops_at_first_qualified_path_radius_and_falls_back(self):
        for frontier_x, expected_goal, expected_cells in ((2, 1, 2),
                                                           (3, 2, 3),
                                                           (6, 5, 7)):
            with self.subTest(frontier_x=frontier_x):
                explorer = BoundaryExplorer(ExplorationConfig(
                    0.1, 1.1, resolution=1.0, goal_spacing=0.1,
                    path_radii=(1.0, 2.0, 3.0), min_visible_gain=1.0))
                safe = {(x, 0) for x in range(7)}
                frontier = (frontier_x, 0)
                performance = {}
                with patch.object(explorer, "_frontiers", return_value={frontier}):
                    goal, path, _ = explorer._goal(
                        (0.5, 0.5, 0.), safe, set(), safe, performance)
                self.assertEqual(explorer.cell(goal[:2]), (expected_goal, 0))
                self.assertEqual(performance["reachable_cells"], expected_cells)
                self.assertEqual(performance["goal_search_radius"],
                                 1.0 if frontier_x == 2 else
                                 2.0 if frontier_x == 3 else "full")
                self.assertTrue(all(explorer.cell(point) in safe for point in path))

    def test_low_gain_and_repeat_visits_expand_search(self):
        explorer = BoundaryExplorer(ExplorationConfig(
            0.1, 2.1, resolution=1.0, goal_spacing=0.1,
            path_radii=(1.0, 2.0, 3.0), min_visible_gain=2.0))
        safe = {(x, 0) for x in range(5)}
        with patch.object(explorer, "_frontiers", return_value={(3, 0), (4, 0)}):
            goal, path, _ = explorer._goal((0.5, 0.5, 0.), safe, set(), safe)
        self.assertEqual(explorer.cell(goal[:2]), (2, 0))
        self.assertTrue(all(explorer.cell(point) in safe for point in path))

    def test_distant_frontiers_are_not_scored_against_local_candidates(self):
        explorer = BoundaryExplorer(ExplorationConfig(
            0.1, 1.1, resolution=1.0, goal_spacing=0.1,
            path_radii=(1.0, 2.0, 3.0), min_visible_gain=1.0))
        safe = {(x, 0) for x in range(11)}
        frontiers = {(2, 0)} | {(100 + x, 0) for x in range(100)}
        performance = {}
        with patch.object(explorer, "_frontiers", return_value=frontiers):
            goal, _, _ = explorer._goal((0.5, 0.5, 0.), safe, set(), safe,
                                        performance)
        self.assertEqual(explorer.cell(goal[:2]), (1, 0))
        self.assertLess(performance["candidate_pairs"], 10)

        explorer = BoundaryExplorer(ExplorationConfig(
            0.1, 1.1, resolution=1.0, goal_spacing=0.1,
            path_radii=(1.0, 2.0, 3.0), min_visible_gain=1.0))
        explorer.goal_visits[(1, 0)] = 1
        with patch.object(explorer, "_frontiers", return_value={(2, 0)}):
            goal, path, _ = explorer._goal((0.5, 0.5, 0.), safe, set(), safe)
        self.assertEqual(explorer.cell(goal[:2]), (2, 0))
        self.assertTrue(all(explorer.cell(point) in safe for point in path))

    def test_radius_uses_safe_path_distance_around_unknown_cells(self):
        explorer = BoundaryExplorer(ExplorationConfig(
            0.1, 0.1, resolution=1.0, goal_spacing=0.1,
            path_radii=(2.0, 3.0), min_visible_gain=1.0))
        safe = {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)}
        performance = {}
        with patch.object(explorer, "_frontiers", return_value={(0, 2)}):
            goal, path, _ = explorer._goal((0.5, 0.5, 0.), safe, set(), safe,
                                          performance)
        self.assertEqual(explorer.cell(goal[:2]), (0, 2))
        self.assertEqual(len(path) - 1, 6)
        self.assertEqual(performance["reachable_cells"], len(safe))
        self.assertTrue(all(explorer.cell(point) in safe for point in path))


class ExplorationLifecycleTests(unittest.TestCase):
    def mapped_explorer(self, **options):
        config = dict(robot_radius=0.1, sensor_range=1.0, pose_margin=0.0,
                      boundary_clearance=0.0, goal_refresh_interval_s=1000.,
                      goal_refresh_distance=10., goal_refresh_yaw_deg=180.)
        config.update(options)
        explorer = BoundaryExplorer(ExplorationConfig(**config))
        grass = BoundaryExplorerTests.grass_square()
        explorer.update(grass, [], (0.05, 0.05, 0.), 0.)
        result = explorer.update(grass, [], (0.05, 0.05, 0.), 1.)
        self.assertEqual(result.status, "exploring")
        return explorer, result, grass

    def test_arrival_is_counted_before_reselection_once_per_assignment(self):
        explorer, first, grass = self.mapped_explorer()
        old_cell = explorer.cell(first.goal_pose[:2])
        version = explorer.map_version
        metrics = {}
        second = explorer.update(grass, [], first.goal_pose, 2., performance=metrics)
        self.assertEqual(explorer.goal_visits[old_cell], 1)
        self.assertEqual(metrics["goal_release_reason"], "arrived")
        self.assertEqual(explorer.map_version, version)
        self.assertNotEqual(explorer.cell(second.goal_pose[:2]), old_cell)
        duplicate = explorer.update(grass, [], first.goal_pose, 2., performance=metrics)
        self.assertEqual(explorer.goal_visits[old_cell], 1)
        self.assertEqual(duplicate.goal_pose, second.goal_pose)
        self.assertEqual(metrics["observation_fused"], 0)
        self.assertEqual(metrics["goal_action"], "held")

    def test_goal_is_held_and_path_starts_at_actual_pose(self):
        explorer, first, grass = self.mapped_explorer()
        pose = (0.071, 0.063, 0.8)
        metrics = {}
        result = explorer.update(grass, [], pose, 2., performance=metrics)
        self.assertEqual(result.goal_pose, first.goal_pose)
        np.testing.assert_array_equal(result.path_xy[0], pose[:2])
        self.assertEqual(metrics["goal_search_radius"], "held")
        self.assertTrue(explorer.trajectory_is_safe(
            np.column_stack((result.path_xy, np.zeros(len(result.path_xy))))))

    def test_timeout_excludes_old_goal_and_selects_another(self):
        explorer, first, grass = self.mapped_explorer(goal_progress_timeout_s=2.)
        metrics = {}
        result = explorer.update(grass, [], (0.05, 0.05, 0.), 3., performance=metrics)
        self.assertEqual(metrics["goal_release_reason"], "progress_timeout")
        old_cell = explorer.cell(first.goal_pose[:2])
        self.assertIn(old_cell, explorer._excluded_goals)
        self.assertIsNotNone(result.goal_pose)
        self.assertNotEqual(explorer.cell(result.goal_pose[:2]), old_cell)

    def test_cumulative_motion_resets_timeout(self):
        explorer, first, grass = self.mapped_explorer(goal_progress_timeout_s=2.)
        start = np.array([0.05, 0.05])
        direction = np.array(first.goal_pose[:2]) - start
        direction /= np.linalg.norm(direction)
        for timestamp, movement in ((2., 0.06), (2.9, 0.12), (4., 0.13)):
            pose = (*tuple(start + movement * direction), 0.)
            result = explorer.update(grass, [], pose, timestamp)
            self.assertEqual(result.goal_pose, first.goal_pose)
        self.assertFalse(explorer._excluded_goals)

    def test_classification_progress_resets_penalties_but_extra_hits_do_not(self):
        explorer, _, grass = self.mapped_explorer()
        explorer.goal_visits[(50, 50)] = 1
        explorer._excluded_goals.add((51, 51))
        version = explorer.map_version
        explorer.update(grass, [], (0.05, 0.05, 0.), 2.)
        self.assertEqual(explorer.map_version, version)
        self.assertEqual(explorer.goal_visits[(50, 50)], 1)
        # An unconfirmed new observation does not change cell classification.
        explorer.update([[3.05, 3.05]], [], (0.05, 0.05, 0.), 3.)
        self.assertEqual(explorer.map_version, version)
        explorer.update([[3.05, 3.05]], [], (0.05, 0.05, 0.), 4.)
        self.assertEqual(explorer.map_version, version + 1)
        self.assertFalse(explorer.goal_visits)
        self.assertFalse(explorer._excluded_goals)

    def test_stalled_and_recovery_after_new_confirmed_grass(self):
        explorer, first, grass = self.mapped_explorer()
        explorer.goal_visits.update({cell: 10000 for cell in first.safe_cells})
        result = explorer.update(grass, [], (0.05, 0.05, 0.), 2.)
        self.assertEqual(result.status, "stalled")
        self.assertIsNone(result.goal_pose)
        self.assertEqual(result.path_xy.shape, (0, 2))
        explorer.update([[3.05, 3.05]], [], (0.05, 0.05, 0.), 3.)
        result = explorer.update([[3.05, 3.05]], [], (0.05, 0.05, 0.), 4.)
        self.assertEqual(result.status, "exploring")

    def test_all_timed_out_viewpoints_eventually_stall(self):
        # This smaller map has only a few admissible viewpoints.
        explorer = BoundaryExplorer(ExplorationConfig(.01, 1., resolution=.1,
            pose_margin=0., boundary_clearance=0., grass_observations=1,
            goal_spacing=.1, goal_reached_distance=.01, goal_progress_timeout_s=1.))
        grass = np.array([explorer.center((x, y)) for x in range(-2, 3)
                          for y in range(-2, 3)])
        for time in range(12):
            result = explorer.update(grass, [], (.05, .05, 0.), float(time))
            if result.status == "stalled":
                break
        self.assertEqual(result.status, "stalled")
        self.assertGreater(len(explorer._excluded_goals), 0)

    def test_new_boundary_invalidates_held_goal(self):
        explorer, first, _ = self.mapped_explorer()
        metrics = {}
        result = explorer.update([], [first.goal_pose[:2]], (.05, .05, 0.), 2.,
                                 performance=metrics)
        self.assertNotEqual(result.goal_pose, first.goal_pose)
        self.assertTrue(metrics["goal_release_reason"].startswith("goal_invalid:"))

    def test_periodic_refresh_reselects_without_resetting_same_goal_timeout(self):
        explorer, first, grass = self.mapped_explorer(goal_refresh_interval_s=.5,
                                                     goal_progress_timeout_s=3.)
        for timestamp in (1.5, 2., 2.5, 3., 3.5):
            metrics = {}
            result = explorer.update(grass, [], (.05, .05, 0.), timestamp, performance=metrics)
            self.assertEqual(metrics["goal_action"], "selected")
            self.assertEqual(metrics["goal_release_reason"], "refresh_interval")
            self.assertNotEqual(metrics["goal_search_radius"], "held")
            self.assertEqual(result.goal_pose, first.goal_pose)
            self.assertFalse(explorer.goal_visits)
        metrics = {}
        result = explorer.update(grass, [], (.05, .05, 0.), 4., performance=metrics)
        self.assertEqual(metrics["goal_release_reason"], "progress_timeout")
        self.assertNotEqual(explorer.cell(result.goal_pose[:2]), explorer.cell(first.goal_pose[:2]))

    def test_displacement_refresh_is_measured_from_last_selection(self):
        explorer, first, grass = self.mapped_explorer(goal_refresh_distance=.1)
        # Move away from the goal, so arrival cannot cause this reselection.
        direction = np.array([.05, .05]) - np.asarray(first.goal_pose[:2])
        direction /= np.linalg.norm(direction)
        for timestamp, offset, action in ((1.1, .06, "held"), (1.2, .12, "selected"),
                                          (1.3, .14, "held")):
            pose = (*tuple(np.array([.05, .05]) + offset * direction), 0.)
            metrics = {}
            explorer.update(grass, [], pose, timestamp, performance=metrics)
            self.assertEqual(metrics["goal_action"], action)
            if action == "selected":
                self.assertEqual(metrics["goal_release_reason"], "refresh_distance")

    def test_yaw_refresh_wraps_at_pi_and_duplicate_time_can_refresh_pose(self):
        explorer, _, grass = self.mapped_explorer(goal_refresh_yaw_deg=20.)
        metrics = {}
        explorer.update(grass, [], (.05, .05, math.radians(179)), 1.1, performance=metrics)
        self.assertEqual(metrics["goal_release_reason"], "refresh_yaw")
        explorer.update(grass, [], (.05, .05, math.radians(-179)), 1.1, performance=metrics)
        self.assertEqual(metrics["goal_action"], "held")
        self.assertEqual(metrics["observation_fused"], 0)
        explorer.update(grass, [], (.05, .05, math.radians(-150)), 1.1, performance=metrics)
        self.assertEqual(metrics["goal_release_reason"], "refresh_yaw")
        self.assertEqual(metrics["goal_action"], "selected")

    def test_map_changes_do_not_postpone_periodic_refresh(self):
        explorer, _, _ = self.mapped_explorer(goal_refresh_interval_s=2.)
        for timestamp in (2., 3.):
            metrics = {}
            explorer.update([], [[10.05 + timestamp, 10.05]], (.05, .05, 0.),
                            timestamp, performance=metrics)
        self.assertEqual(metrics["goal_release_reason"], "refresh_interval")
        self.assertEqual(metrics["goal_action"], "selected")

    def test_positive_generic_goal_beats_nonpositive_boundary_goal(self):
        explorer = BoundaryExplorer(ExplorationConfig(.1, 1.1, resolution=1.,
            goal_spacing=.1, boundary_frontier_distance=1., min_visible_gain=1.))
        safe = {(x, 0) for x in range(7)}
        explorer.goal_visits.update({(5, 0): 100, (6, 0): 100})
        metrics = {}
        with patch.object(explorer, "_frontiers", return_value={(1, 0), (6, 0)}):
            goal, _, _ = explorer._goal((.5, .5, 0.), safe, {(6, 1)}, safe, metrics)
        self.assertIsNotNone(goal)
        self.assertLess(explorer.cell(goal[:2])[0], 5)
        self.assertEqual(metrics["goal_boundary_gain"], 0)

    def test_synthetic_closed_loop_moves_and_confirms_more_grass(self):
        explorer = BoundaryExplorer(ExplorationConfig(.1, .8, pose_margin=0.,
            boundary_clearance=0., sensor_fov_deg=360., goal_spacing=.2))
        ground = np.array([explorer.center((x, y)) for x in range(-15, 16)
                           for y in range(-15, 16)])
        pose = np.array([.05, .05, 0.])
        initial_count = None
        traveled = 0.
        for tick in range(60):
            observed = ground[np.linalg.norm(ground - pose[:2], axis=1) <= .8]
            result = explorer.update(observed, [], pose, tick * .2)
            if initial_count is None and result.status == "exploring":
                initial_count = len(result.known_grass)
            if result.goal_pose is None:
                continue
            # Execute one short step of the actual returned path, then observe
            # from that position. This is a feedback loop, not a logged route.
            for target in result.path_xy[1:]:
                delta = target - pose[:2]
                length = np.linalg.norm(delta)
                if length > 1e-9:
                    next_pose = np.array([*(pose[:2] + delta * min(1., .08 / length)),
                                          result.goal_pose[2]])
                    self.assertTrue(explorer.trajectory_is_safe(np.array([pose, next_pose])))
                    traveled += np.linalg.norm(next_pose[:2] - pose[:2])
                    pose = next_pose
                    break
        self.assertGreater(traveled, .5)
        self.assertGreater(len(result.known_grass), initial_count + 30)


class ExplorationSafetyAndEvidenceTests(unittest.TestCase):
    def test_new_configuration_validation(self):
        for field in ("goal_reached_distance", "goal_progress_timeout_s", "goal_progress_distance",
                      "goal_refresh_interval_s", "goal_refresh_distance", "goal_refresh_yaw_deg"):
            for value in (0, -1, math.inf, math.nan):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    ExplorationConfig(.1, 1., **{field: value})
        for value in (0, -1, 1.5, True, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ExplorationConfig(.1, 1., boundary_clear_observations=value)

    def test_offcenter_footprint_regression_and_touching_unknown(self):
        explorer = BoundaryExplorer(ExplorationConfig(.2, 2.))
        explorer.grass_hits = {(x, y): 2 for x in range(-10, 11)
                               for y in range(-10, 11) if (x, y) != (3, 2)}
        self.assertLess(math.hypot(.3 - .099, .2 - .099), .25)
        self.assertFalse(explorer.trajectory_is_safe(np.array([[.099, .099, 0.]])))
        # At the right edge of cell (0, 0), radius .2 touches cell (3, 0).
        explorer = BoundaryExplorer(ExplorationConfig(.2, 2., pose_margin=0.))
        explorer.grass_hits = {(x, y): 2 for x in range(-10, 11)
                               for y in range(-10, 11) if (x, y) != (3, 0)}
        self.assertNotIn((0, 0), explorer._safe_cells(0.)[2])

    def test_supercover_corners_edges_and_short_cell_intersections(self):
        explorer = BoundaryExplorer(ExplorationConfig(.1, 2., resolution=1.))
        cells = explorer._segment_cells((.5, .5), (1.5, 1.5))
        self.assertEqual(cells, {(0, 0), (1, 0), (0, 1), (1, 1)})
        self.assertFalse(explorer._path_is_safe([(.5, .5), (1.5, 1.5)], {(0, 0), (1, 1)}))
        self.assertEqual(explorer._segment_cells((0., .2), (0., 1.8)),
                         {(-1, 0), (0, 0), (-1, 1), (0, 1)})
        cells = explorer._segment_cells((.1, .1), (2.1, 2.099))
        self.assertIn((1, 0), cells)
        self.assertIn((2, 1), cells)

    def test_narrow_corridor_is_rejected_without_weakening_safety(self):
        explorer = BoundaryExplorer(ExplorationConfig(.1, 2., pose_margin=0.,
                                                     boundary_clearance=0.))
        explorer.grass_hits = {(x, y): 2 for x in range(-10, 11) for y in range(-1, 3)}
        self.assertFalse(explorer.trajectory_is_safe(np.array([[.05, .05, 0.], [.55, .05, 0.]])))
        explorer.grass_hits = {(x, y): 2 for x in range(-10, 11) for y in range(-4, 5)}
        self.assertTrue(explorer.trajectory_is_safe(np.array([[.05, .05, 0.], [.55, .05, 0.]])))

    def test_arc_rejected_when_endpoint_chord_is_safe(self):
        explorer = BoundaryExplorer(ExplorationConfig(.001, 2., pose_margin=0.,
                                                     boundary_clearance=0.))
        explorer.grass_hits = {(x, y): 2 for x in range(-15, 16)
                               for y in range(-15, 16) if (x, y) != (3, 3)}
        chord = np.array([[.05, .05, 0.], [.05, .05 + 2 / math.pi, math.pi]])
        self.assertTrue(explorer.trajectory_is_safe(chord))
        self.assertFalse(explorer.command_is_safe(chord[0], 1., math.pi, 1., step_s=10.))
        self.assertTrue(explorer.command_is_safe(chord[0], 0., math.pi, 1.))

    def test_arc_sagitta_margin_rejects_near_threshold_clearance(self):
        explorer = BoundaryExplorer(ExplorationConfig(.099, 2., pose_margin=0.,
                                                     boundary_clearance=0.))
        explorer.grass_hits = {(x, y): 2 for x in range(-10, 11)
                               for y in range(-10, 11) if (x, y) != (2, 0)}
        angle = math.pi / 8
        endpoint = (.05 + .1 * math.sin(angle), .05 + .1 * (1 - math.cos(angle)), angle)
        self.assertTrue(explorer.trajectory_is_safe(np.array([[.05, .05, 0.], endpoint])))
        self.assertFalse(explorer.command_is_safe((.05, .05, 0.), .2, 2., angle / 2, step_s=10.))

    def test_duplicate_timestamp_cannot_confirm_grass_or_boundary(self):
        explorer = BoundaryExplorer(ExplorationConfig(.1, 1.))
        grass = [[.05, .05]]
        boundary = [[1.05, .05]]
        explorer.update(grass, boundary, (0., 0., 0.), 1.)
        metrics = {}
        result = explorer.update(grass, boundary, (0., 0., 0.), 1., performance=metrics)
        self.assertEqual(explorer.grass_hits[(0, 0)], 1)
        self.assertEqual(explorer.boundary_hits[(10, 0)], 1)
        self.assertEqual(metrics["observation_fused"], 0)
        self.assertEqual((result.new_grass_cells, result.new_boundary_cells), (0, 0))
        with self.assertRaises(ValueError):
            explorer.update(grass, boundary, (0., 0., 0.), .9)

    def test_boundary_conflict_retraction_and_cache_invalidation(self):
        explorer = BoundaryExplorer(ExplorationConfig(.1, 1.))
        point = [[.05, .05]]
        pose = (5., 5., 0.)
        explorer.update(point, [], pose, 0.)
        explorer.update(point, [], pose, 1.)
        self.assertEqual(explorer.cell_state((0, 0)), "grass")
        explorer.update(point, point, pose, 2.)
        self.assertNotIn((0, 0), explorer.grass_hits)
        self.assertEqual(explorer.cell_state((0, 0)), "boundary")
        first_counterevidence = explorer.update(point, [], pose, 3.)
        self.assertEqual(first_counterevidence.new_grass_cells, 1)
        explorer.update(point, [], pose, 3.)  # duplicate is not counterevidence
        explorer.update([], [], pose, 4.)  # missing does not reset it
        next_counterevidence = explorer.update(point, [], pose, 5.)
        self.assertEqual(next_counterevidence.new_grass_cells, 0)
        self.assertEqual(explorer.cell_state((0, 0)), "boundary")
        explorer._visibility_cache[((5, 5), (6, 6))] = False
        metrics = {}
        explorer.update(point, [], pose, 6., performance=metrics)
        self.assertEqual(explorer.cell_state((0, 0)), "grass")
        self.assertEqual(metrics["removed_boundary_cells"], 1)
        self.assertFalse(explorer._visibility_cache)
        explorer.update([], point, pose, 7.)
        explorer.update(point, [], pose, 8.)
        explorer.update(point, point, pose, 9.)  # restart counterevidence
        explorer.update(point, [], pose, 10.)
        explorer.update(point, [], pose, 11.)
        self.assertEqual(explorer.cell_state((0, 0)), "boundary")

    def test_complete_is_not_latched_after_boundary_retraction(self):
        explorer = BoundaryExplorer(ExplorationConfig(.1, 1.))
        cells = {(x, y) for x in range(-5, 6) for y in range(-5, 6)
                 if abs(x) == 5 or abs(y) == 5}
        boundary = [explorer.center(cell) for cell in sorted(cells)]
        explorer.update([], boundary, (.05, .05, 0.), 0.)
        result = explorer.update([], boundary, (.05, .05, 0.), 1.)
        self.assertEqual(result.status, "complete")
        self.assertIn("encloses robot on grid", result.reason)
        hole = [explorer.center((5, 0))]
        for time in (2., 3., 4.):
            result = explorer.update(hole, [], (.05, .05, 0.), time)
        self.assertFalse(result.boundary_closed)
        self.assertNotEqual(result.status, "complete")


if __name__ == "__main__":
    unittest.main()
