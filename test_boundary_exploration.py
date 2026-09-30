"""Behavioral checks for the independent grass-only boundary explorer."""

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


if __name__ == "__main__":
    unittest.main()
