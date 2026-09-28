"""Estimate a smooth boundary centerline and its local direction from XY points."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Shape:
    centerline: np.ndarray
    directions_deg: np.ndarray
    length_m: float
    turn_deg: float
    label: str
    median_error_m: float


def recognize(points_xyz, bins=24, max_curvature=1.5):
    """Return a robust XY trend, or None if the cloud has too little support.

    The centerline runs from its endpoint nearer the robot to the farther one
    when robot_xy is supplied separately by orient_shape(). Without that cue,
    its sign follows the dominant global X (or Y) axis.
    """
    xy = np.asarray(points_xyz, dtype=float)[:, :2]
    if max_curvature <= 0:
        raise ValueError("max_curvature must be positive")
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 12:
        return None

    center = np.median(xy, axis=0)
    centered = xy - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]])
    along = centered @ axis
    across = centered @ normal
    lower, upper = np.percentile(along, [2, 98])
    if upper - lower < 0.25:
        return None

    edges = np.linspace(lower, upper, bins + 1)
    t, offsets = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (along >= left) & (along < right if right < upper else along <= right)
        if mask.sum() >= 3:
            t.append(float(np.median(along[mask])))
            offsets.append(float(np.median(across[mask])))
    if len(t) < 5:
        return None

    t = np.asarray(t)
    offsets = np.asarray(offsets)
    # Reject isolated transverse clutter, then smooth with a short weighted kernel.
    residual = across - np.interp(along, t, offsets)
    mad = np.median(np.abs(residual - np.median(residual)))
    threshold = max(0.06, 3 * 1.4826 * mad)
    good = np.abs(residual) <= threshold
    if good.sum() >= 12 and good.sum() < len(xy):
        along_good, across_good = along[good], across[good]
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (along_good >= left) & (along_good < right if right < upper else along_good <= right)
            if mask.sum() >= 3:
                j = np.argmin(np.abs(t - np.median(along_good[mask])))
                offsets[j] = np.median(across_good[mask])
    # Fit the most expressive curve whose radius of curvature stays safely
    # above the 0.4 m tracking offset. An inner offset of a tighter bend can
    # form a cusp or loop, so fall back to a simpler trend when necessary.
    normalized_t = 2 * (t - t[0]) / (t[-1] - t[0]) - 1
    sample_t = np.linspace(t[0], t[-1], 64)
    sample_s = 2 * (sample_t - t[0]) / (t[-1] - t[0]) - 1
    dt_scale = 2 / (t[-1] - t[0])
    for degree in (3, 2, 1):
        coefficients = np.polyfit(normalized_t, offsets, deg=degree)
        slope = np.polyval(np.polyder(coefficients), sample_s) * dt_scale
        bend = np.polyval(np.polyder(coefficients, 2), sample_s) * dt_scale ** 2
        curvature = np.abs(bend) / (1 + slope ** 2) ** 1.5
        if np.max(curvature) <= max_curvature or degree == 1:
            break
    smooth = np.polyval(coefficients, sample_s)
    line = center + sample_t[:, None] * axis + smooth[:, None] * normal
    segments = np.diff(line, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1e-6):
        return None
    angles = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    directions = np.degrees(angles)
    turn = float(np.degrees(angles[-1] - angles[0]))
    # Compare turning in the two halves of the fitted curve.
    middle = len(angles) // 2
    first_turn = np.degrees(angles[middle] - angles[0])
    second_turn = np.degrees(angles[-1] - angles[middle])
    chord = line[-1] - line[0]
    chord_normal = np.array([-chord[1], chord[0]]) / np.linalg.norm(chord)
    deviation = np.max(np.abs((line - line[0]) @ chord_normal))
    if abs(turn) < 12 and deviation < 0.08:
        label = "straight"
    elif first_turn * second_turn < 0 and min(abs(first_turn), abs(second_turn)) > 15 and deviation > 0.08:
        label = "S bend"
    elif abs(turn) < 12:
        label = "straight"
    elif turn > 0:
        label = "left bend"
    else:
        label = "right bend"
    return Shape(line, directions, float(lengths.sum()), turn, label,
                 float(np.median(np.abs(residual))))


def orient_shape(shape, robot_xy):
    """Choose line direction from the endpoint closer to the robot."""
    if shape is None:
        return None
    robot_xy = np.asarray(robot_xy, dtype=float)[:2]
    if np.linalg.norm(shape.centerline[0] - robot_xy) <= np.linalg.norm(shape.centerline[-1] - robot_xy):
        return shape
    return Shape(shape.centerline[::-1].copy(),
                 (shape.directions_deg[::-1] % 360) - 180,
                 shape.length_m, -shape.turn_deg,
                 {"left bend": "right bend", "right bend": "left bend"}.get(shape.label, shape.label),
                 shape.median_error_m)


def reference_side(shape, robot_xy, grass_points=None):
    """Choose the likely lawn side of a boundary; return (sign, evidence).

    Grass points, when supplied in the same XY plane, take priority. Without
    them the robot position is used as a provisional interior-side cue.
    """
    line = shape.centerline
    chord = line[-1] - line[0]
    axis = chord / np.linalg.norm(chord)
    normal = np.array([-axis[1], axis[0]])
    if grass_points is not None:
        grass = np.asarray(grass_points, dtype=float)
        if grass.size == 0:
            grass = np.empty((0, 2))
        elif grass.ndim != 2 or grass.shape[1] < 2:
            raise ValueError("grass_points must contain XY coordinates")
        grass = grass[:, :2]
        grass = grass[np.isfinite(grass).all(axis=1)]
        along = (grass - line[0]) @ axis
        usable = grass[(along >= -0.2) & (along <= np.linalg.norm(chord) + 0.2)]
        if len(usable) >= 5:
            signed = (usable - line[0]) @ normal
            significant = signed[np.abs(signed) >= 0.05]
            if len(significant) >= 5:
                positive_fraction = np.mean(significant > 0)
                if positive_fraction >= 0.7:
                    return 1, "grass"
                if positive_fraction <= 0.3:
                    return -1, "grass"
    side = np.dot(np.asarray(robot_xy, dtype=float)[:2] - line[0], normal)
    return (1 if side >= 0 else -1), "robot"


def tracking_reference(shape, robot_xy, offset_m=0.4, grass_points=None):
    """Offset each fitted XY point along its normal toward the lawn interior.

    The side is selected by grass points when available, otherwise by the
    robot position. Returns None when no shape is available.
    """
    if shape is None:
        return None
    if offset_m < 0:
        raise ValueError("offset_m must be nonnegative")
    line = shape.centerline
    tangent = np.gradient(line, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    left_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    robot_xy = np.asarray(robot_xy, dtype=float)[:2]
    sign, _ = reference_side(shape, robot_xy, grass_points)
    return line + sign * offset_m * left_normal


def orient_tracking_reference(reference, boundary):
    """Order the reference so the corresponding boundary stays on its right."""
    if reference is None:
        return None
    tangent = np.gradient(reference, axis=0)
    toward_boundary = boundary - reference
    cross = tangent[:, 0] * toward_boundary[:, 1] - tangent[:, 1] * toward_boundary[:, 0]
    return reference[::-1].copy() if np.median(cross) > 0 else reference


def grass_inside_support(reference, grass_points):
    """Fraction of nearby grass samples left of the directed reference.

    Returns None when fewer than five usable grass points are available. This
    is observational support, not proof of continuous lawn coverage.
    """
    if reference is None or grass_points is None:
        return None
    grass = np.asarray(grass_points, dtype=float)
    if grass.size == 0:
        return None
    if grass.ndim != 2 or grass.shape[1] < 2:
        raise ValueError("grass_points must contain XY coordinates")
    grass = grass[np.isfinite(grass[:, :2]).all(axis=1), :2]
    if len(grass) < 5:
        return None
    chord = reference[-1] - reference[0]
    length = np.linalg.norm(chord)
    axis = chord / length
    along = (grass - reference[0]) @ axis
    grass = grass[(along >= -0.2) & (along <= length + 0.2)]
    if len(grass) < 5:
        return None
    distances = np.linalg.norm(grass[:, None, :] - reference[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    tangents = np.gradient(reference, axis=0)[nearest]
    offsets = grass - reference[nearest]
    cross = tangents[:, 0] * offsets[:, 1] - tangents[:, 1] * offsets[:, 0]
    significant = cross[np.abs(cross) > 0.05 * np.linalg.norm(tangents, axis=1)]
    return float(np.mean(significant > 0)) if len(significant) >= 5 else None


def grass_reference_coverage(reference, grass_points, radius_m=0.25):
    """Fraction of reference samples with a grass point within radius_m."""
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if reference is None or grass_points is None:
        return None
    grass = np.asarray(grass_points, dtype=float)
    if grass.size == 0:
        return None
    if grass.ndim != 2 or grass.shape[1] < 2:
        raise ValueError("grass_points must contain XY coordinates")
    grass = grass[np.isfinite(grass[:, :2]).all(axis=1), :2]
    if len(grass) < 5:
        return None
    radius_sq = radius_m ** 2
    covered = sum(np.min(np.sum((grass - point) ** 2, axis=1)) <= radius_sq
                  for point in reference)
    return covered / len(reference)
