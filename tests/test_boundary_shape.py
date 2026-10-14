"""Focused geometry and input-format checks for tracking reference generation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from boundary_shape import (grass_inside_support, grass_reference_coverage,
                            orient_tracking_reference, recognize, reference_side,
                            tracking_reference)
from replay_boundary import (Frame, camera_xz_to_global_xy,
                             global_xy_to_camera_xz, load_grass_csv,
                             parse_frame, recorded_loop_direction)


class TrackingReferenceTests(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0, 2, 80)
        self.shape = recognize(np.column_stack((x, np.zeros_like(x))))

    def test_grass_overrides_robot_and_boundary_is_right(self):
        x = np.linspace(0.2, 1.8, 20)
        grass = np.column_stack((x, np.full_like(x, 0.8)))
        self.assertEqual(reference_side(self.shape, (0, -1), grass), (1, "grass"))
        reference = orient_tracking_reference(
            tracking_reference(self.shape, (0, -1), grass_points=grass),
            self.shape.centerline)
        self.assertAlmostEqual(float(np.median(reference[:, 1])), 0.4, places=6)
        self.assertGreater(reference[-1, 0], reference[0, 0])
        tangent = np.gradient(reference, axis=0)
        delta = self.shape.centerline - reference
        cross = tangent[:, 0] * delta[:, 1] - tangent[:, 1] * delta[:, 0]
        self.assertTrue(np.all(cross < 0))
        self.assertAlmostEqual(grass_inside_support(reference, grass), 1.0)

    def test_grass_inside_support_detects_shallow_grass_samples(self):
        x = np.linspace(0.2, 1.8, 20)
        grass = np.column_stack((x, np.full_like(x, 0.1)))
        reference = orient_tracking_reference(
            tracking_reference(self.shape, (0, 1)), self.shape.centerline)
        self.assertAlmostEqual(grass_inside_support(reference, grass), 0.0)

    def test_grass_coverage_measures_support_near_reference(self):
        reference = orient_tracking_reference(
            tracking_reference(self.shape, (0, 1)), self.shape.centerline)
        grass_near = reference.copy()
        grass_far = grass_near + np.array([0, 1])
        self.assertAlmostEqual(grass_reference_coverage(reference, grass_near), 1.0)
        self.assertAlmostEqual(grass_reference_coverage(reference, grass_far), 0.0)

    def test_robot_fallback_orients_opposite_side(self):
        reference = orient_tracking_reference(
            tracking_reference(self.shape, (0, -1)), self.shape.centerline)
        self.assertAlmostEqual(float(np.median(reference[:, 1])), -0.4, places=6)
        self.assertLess(reference[-1, 0], reference[0, 0])

    def test_turn_radius_limit_simplifies_tight_curve(self):
        x = np.linspace(-1, 1, 120)
        cloud = np.column_stack((x, 0.8 * x * x))
        unconstrained = recognize(cloud, max_curvature=10)
        constrained = recognize(cloud, max_curvature=1 / (0.5 + 0.4))
        self.assertGreater(abs(unconstrained.turn_deg), abs(constrained.turn_deg))
        self.assertLess(abs(constrained.turn_deg), 1)

    def test_optional_grass_field_and_csv_use_camera_coordinates(self):
        line = ("[2026-09-10 16:15:00.053] timestamp(ms): 0.000000, "
                "boundary_xz: [(0.5, 1.0); (0.6, 1.1)], "
                "grass_xz: [(0.4, 1.0); (0.3, 1.1)], "
                "location info: 1,2,0,0,0,0,1")
        frame = parse_frame(line, 1)
        restored = global_xy_to_camera_xz(frame.grass_world_xy, frame)
        np.testing.assert_allclose(restored, [[0.4, 1.0], [0.3, 1.1]], atol=1e-12)
        np.testing.assert_allclose(
            camera_xz_to_global_xy(restored, frame), frame.grass_world_xy, atol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grass.csv"
            path.write_text("frame,camera_x,camera_z\n1,0.4,1.0\n", encoding="utf-8")
            load_grass_csv(path, [frame])
        np.testing.assert_allclose(frame.grass_world_xy,
                                   camera_xz_to_global_xy([[0.4, 1.0]], frame))

    def test_recorded_route_requires_closure_and_consistent_rotation(self):
        positions = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        frames = [Frame("", float(i), np.array([x, y, 0.0]),
                        np.array([0.0, 0.0, i * np.pi / 2]), i * np.pi / 2,
                        np.empty((0, 2)), np.empty((0, 3)))
                  for i, (x, y) in enumerate(positions)]
        self.assertIn("CCW", recorded_loop_direction(frames))
        self.assertIn("open", recorded_loop_direction(frames[:-1]))


if __name__ == "__main__":
    unittest.main()
