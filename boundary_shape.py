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


def recognize(points_xyz, bins=24):
    """Return a robust XY trend, or None if the cloud has too little support.

    The centerline runs from its endpoint nearer the robot to the farther one
    when robot_xy is supplied separately by orient_shape(). Without that cue,
    its sign follows the dominant global X (or Y) axis.
    """
    xy = np.asarray(points_xyz, dtype=float)[:, :2]
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
    # Fit a low-order curve to bin medians so individual noisy bins cannot
    # create artificial turns or S bends.
    normalized_t = 2 * (t - t[0]) / (t[-1] - t[0]) - 1
    coefficients = np.polyfit(normalized_t, offsets, deg=3)
    smooth = np.polyval(coefficients, normalized_t)
    line = center + t[:, None] * axis + smooth[:, None] * normal
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
