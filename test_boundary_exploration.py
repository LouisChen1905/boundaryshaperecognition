"""Behavioral checks for the independent grass-only boundary explorer."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
