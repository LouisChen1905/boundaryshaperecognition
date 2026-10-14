"""Conservative grass-only grid exploration for mapping a boundary contour.

The caller supplies time-aligned world XY observations, or camera XYZ points
with a complete camera-to-world transform. Unknown cells are never traversable.
This module does not generate wheel commands; a local planner must check its
candidate motion with ``trajectory_is_safe`` before execution.
"""

from collections import deque
from dataclasses import dataclass
import heapq
import math
from time import perf_counter_ns

import numpy as np


@dataclass(frozen=True)
class ExplorationConfig:
    robot_radius: float
    sensor_range: float
    resolution: float = 0.1
    boundary_clearance: float = 0.1
    pose_margin: float = 0.05
    grass_observations: int = 2
    boundary_link_gap: float = 0.12
    goal_spacing: float = 0.3
    sensor_fov_deg: float = 120.0
    path_radii: tuple = (1.0, 2.0, 3.0)
    min_visible_gain: float = 2.0
    boundary_frontier_distance: float = 0.3
    boundary_distance_weight: float = 2.0
    goal_reached_distance: float = 0.15
    goal_progress_timeout_s: float = 15.0
    goal_progress_distance: float = 0.1
    boundary_clear_observations: int = 3
    goal_refresh_interval_s: float = 2.0
    goal_refresh_distance: float = 0.3
    goal_refresh_yaw_deg: float = 20.0

    def __post_init__(self):
        positive = (self.robot_radius, self.sensor_range, self.resolution,
                    self.boundary_link_gap, self.goal_reached_distance,
                    self.goal_progress_timeout_s, self.goal_progress_distance,
                    self.goal_refresh_interval_s, self.goal_refresh_distance,
                    self.goal_refresh_yaw_deg)
        nonnegative = (self.boundary_clearance, self.pose_margin, self.goal_spacing)
        if (any(not math.isfinite(v) or v <= 0 for v in positive)
                or any(not math.isfinite(v) or v < 0 for v in nonnegative)
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                       or v < 1 for v in (self.grass_observations,
                                          self.boundary_clear_observations))):
            raise ValueError("exploration limits must be finite; counts must be positive integers")
        if (self.robot_radius <= 0 or self.sensor_range <= 0 or self.resolution <= 0
                or self.boundary_clearance < 0 or self.pose_margin < 0
                or self.grass_observations < 1 or self.boundary_link_gap <= 0
                or self.goal_spacing < 0
                or not 0 < self.sensor_fov_deg <= 360
                or self.goal_refresh_yaw_deg > 180
                or not math.isfinite(self.min_visible_gain)
                or self.min_visible_gain <= 0
                or not math.isfinite(self.boundary_frontier_distance)
                or self.boundary_frontier_distance <= 0
                or not math.isfinite(self.boundary_distance_weight)
                or self.boundary_distance_weight < 0):
            raise ValueError("exploration configuration contains an invalid limit")
        try:
            radii = tuple(self.path_radii)
            if (not radii or any(not math.isfinite(radius) or radius <= 0
                                 for radius in radii)
                    or any(a >= b for a, b in zip(radii, radii[1:]))):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("path_radii must be finite, positive and strictly increasing") from None
        object.__setattr__(self, "path_radii", radii)


@dataclass
class ExplorationResult:
    status: str
    reason: str
    goal_pose: object
    path_xy: np.ndarray
    known_grass: frozenset
    boundary_cells: frozenset
    safe_cells: frozenset
    boundary_closed: bool
    new_grass_cells: int
    new_boundary_cells: int


def points_to_world_xy(points, sensor_to_world=None):
    """Convert world XY or camera XYZ samples to global XY.

    A 3D input requires a finite 4x4 transform; camera Y is retained during
    the transformation instead of being silently set to zero.
    """
    array = np.asarray(points, dtype=float)
    if array.size == 0 and array.ndim == 1:
        array = array.reshape(0, 3 if sensor_to_world is not None else 2)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError("points must have shape (N, 2) or (N, 3)")
    if array.shape[1] == 2:
        if sensor_to_world is not None:
            raise ValueError("2D points must already be in global XY")
        return array[np.isfinite(array).all(axis=1)].copy()
    transform = np.asarray(sensor_to_world, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])):
        raise ValueError("camera XYZ requires a finite 4x4 sensor_to_world transform")
    finite = array[np.isfinite(array).all(axis=1)]
    homogeneous = np.column_stack((finite, np.ones(len(finite))))
    world = homogeneous @ transform.T
    return world[:, :2] / world[:, 3, None]


def _bearing_difference(first, second):
    return abs(math.atan2(math.sin(first - second),
                          math.cos(first - second)))


@dataclass(frozen=True)
class GoalScoreCandidate:
    """Geometry available to a scorer; manual targets are never input features."""

    cell: tuple
    pose: tuple
    robot_pose: tuple
    gain: float
    boundary_gain: float
    path_cost: float
    boundary_distance: float
    visits: int
    baseline_score: float
    sensor_range: float
    boundary_tangent: float = None
    boundary_direction_confidence: float = 0.0


class BoundaryExplorer:
    """Build a sparse evidence grid and select safe boundary-viewing poses."""

    def __init__(self, config, goal_scorer=None):
        self.config = config
        self.goal_scorer = goal_scorer
        self.grass_hits = {}
        self.boundary_hits = {}
        self.last_seen = {}
        self.last_timestamp = None
        self.last_goal = None
        self.goal_visits = {}
        self._visibility_cache = {}
        self._offset_cache = {}
        self._grass_counterevidence = {}
        self.map_version = 0
        self._map_signature = (frozenset(), frozenset())
        self.active_goal = None
        self._excluded_goals = set()
        self._goal_progress_time = None
        self._goal_progress_distance = None
        self._goal_refresh_time = None
        self._goal_refresh_pose = None

    def cell(self, xy):
        x, y = np.asarray(xy, dtype=float)[:2]
        return (int(math.floor(x / self.config.resolution)),
                int(math.floor(y / self.config.resolution)))

    def center(self, cell):
        return np.array([(cell[0] + 0.5) * self.config.resolution,
                         (cell[1] + 0.5) * self.config.resolution])

    def cell_state(self, cell):
        """Return boundary, grass, or unknown; boundary takes precedence."""
        if self.boundary_hits.get(cell, 0):
            return "boundary"
        if self.grass_hits.get(cell, 0) >= self.config.grass_observations:
            return "grass"
        return "unknown"

    def seed_verified_grass_disk(self, center_xy, radius):
        """Add an externally verified grass patch around the starting pose.

        This is never called automatically: a detected pose alone is not
        evidence that its entire footprint is traversable grass.
        """
        center = np.asarray(center_xy, dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("center_xy must be finite XY")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be positive")
        extent = int(math.ceil(radius / self.config.resolution)) + 1
        cx, cy = self.cell(center)
        half_diagonal = self.config.resolution / math.sqrt(2)
        for x in range(cx - extent, cx + extent + 1):
            for y in range(cy - extent, cy + extent + 1):
                cell = (x, y)
                if np.linalg.norm(self.center(cell) - center) + half_diagonal <= radius:
                    self.grass_hits[cell] = max(self.grass_hits.get(cell, 0),
                                                self.config.grass_observations)

    def _raster_segment(self, start, end):
        length = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(length / (self.config.resolution / 2))))
        return {self.cell(start + (end - start) * (i / count))
                for i in range(count + 1)}

    def _boundary_observation_cells(self, points):
        cells = {self.cell(point) for point in points}
        # The sensor's sequence is locally ordered in the sample data, but
        # large jumps separate fragments and must never be bridged.
        for start, end in zip(points[:-1], points[1:]):
            if np.linalg.norm(end - start) <= self.config.boundary_link_gap:
                cells.update(self._raster_segment(start, end))
        return cells

    def _disk_offsets(self, radius):
        # Minimum square-to-square distance: every center in the source cell
        # must have its entire disk clear. Touching an unknown square is unsafe.
        if radius not in self._offset_cache:
            resolution = self.config.resolution
            cells = int(math.ceil(radius / resolution)) + 1
            offsets = tuple(
                (dx, dy) for dx in range(-cells, cells + 1)
                for dy in range(-cells, cells + 1)
                if math.hypot(max(abs(dx) - 1, 0), max(abs(dy) - 1, 0))
                * resolution <= radius + 1e-12)
            if len(self._offset_cache) >= 128:
                self._offset_cache.pop(next(iter(self._offset_cache)))
            self._offset_cache[radius] = offsets
        return self._offset_cache[radius]

    def _safe_cells(self, position_std, extra_margin=0.0):
        known = {cell for cell, hits in self.grass_hits.items()
                 if hits >= self.config.grass_observations and cell not in self.boundary_hits}
        boundary = set(self.boundary_hits)
        footprint = self.config.robot_radius + max(self.config.pose_margin,
                                                   2 * position_std) + extra_margin
        body_offsets = self._disk_offsets(footprint)
        boundary_offsets = self._disk_offsets(footprint + self.config.boundary_clearance)
        # Apply the identical footprint tests in bulk. Bound memory for sparse
        # maps whose coordinates span a very large area; use the sparse fallback.
        if known:
            coordinates = np.asarray(list(known), dtype=np.int64)
            low = coordinates.min(axis=0)
            span = coordinates.max(axis=0) - low + 1
            pad = max(max(abs(dx), abs(dy)) for dx, dy in boundary_offsets)
            shape = tuple(int(n) + 2 * pad for n in span)
            if shape[0] * shape[1] <= 4_000_000:
                grass_grid = np.zeros(shape, dtype=bool)
                boundary_grid = np.zeros(shape, dtype=bool)
                indices = coordinates - low + pad
                x, y = indices.T
                grass_grid[x, y] = True
                if boundary:
                    b = np.asarray(list(boundary), dtype=np.int64) - low + pad
                    b = b[np.all((b >= 0) & (b < shape), axis=1)]
                    boundary_grid[b[:, 0], b[:, 1]] = True
                valid = np.ones(len(coordinates), dtype=bool)
                for dx, dy in body_offsets:
                    valid &= grass_grid[x + dx, y + dy]
                for dx, dy in boundary_offsets:
                    valid &= ~boundary_grid[x + dx, y + dy]
                return known, boundary, set(map(tuple, coordinates[valid].tolist()))
        safe = set()
        for x, y in known:
            if any((x + dx, y + dy) not in known for dx, dy in body_offsets):
                continue
            if any((x + dx, y + dy) in boundary for dx, dy in boundary_offsets):
                continue
            safe.add((x, y))
        return known, boundary, safe

    def _segment_cells(self, start, end):
        """Supercover of a closed segment, including edge and corner contacts.

        Grid crossings partition the segment into intervals contained in one
        cell (or on an edge). Check crossings and interval midpoints, so no
        short intersection can fall between fixed-distance samples.
        """
        a = np.asarray(start, dtype=float)[:2] / self.config.resolution
        b = np.asarray(end, dtype=float)[:2] / self.config.resolution
        delta = b - a
        times = {0.0, 1.0}
        for axis in (0, 1):
            if delta[axis] == 0:
                continue
            low, high = sorted((a[axis], b[axis]))
            for gridline in range(math.ceil(low), math.floor(high) + 1):
                t = (gridline - a[axis]) / delta[axis]
                if 0 <= t <= 1:
                    times.add(float(t))
        ordered = sorted(times)
        samples = ordered + [(x + y) / 2 for x, y in zip(ordered, ordered[1:])]
        cells = set()
        for t in samples:
            point = a + t * delta
            axes = []
            for value in point:
                nearest = round(float(value))
                axes.append((nearest - 1, nearest) if abs(value - nearest) <= 1e-10
                            else (math.floor(value),))
            cells.update((x, y) for x in axes[0] for y in axes[1])
        return cells

    def _path_is_safe(self, points, safe):
        if not len(points):
            return False
        if len(points) == 1:
            return self._segment_cells(points[0], points[0]) <= safe
        return all(self._segment_cells(a, b) <= safe
                   for a, b in zip(points[:-1], points[1:]))

    def _build_path(self, pose, target, start, previous):
        cells = [target]
        while cells[-1] != start:
            cells.append(previous[cells[-1]])
        cells.reverse()
        centers = np.array([self.center(cell) for cell in cells])
        if np.array_equal(np.asarray(pose[:2]), centers[0]):
            return centers
        return np.vstack((pose[:2], centers))

    @staticmethod
    def _neighbors(cell, allowed):
        x, y = cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if not (dx or dy) or (x + dx, y + dy) not in allowed:
                    continue
                # Prevent a diagonal route from cutting an unknown corner.
                if dx and dy and ((x + dx, y) not in allowed
                                  or (x, y + dy) not in allowed):
                    continue
                yield (x + dx, y + dy), math.hypot(dx, dy)

    def _reachable_paths(self, start, safe, previous):
        """Yield settled safe cells in path-distance order from one Dijkstra queue."""
        costs = {start: 0.0}
        queue = [(0.0, start)]
        while queue:
            cost, cell = heapq.heappop(queue)
            if cost > costs[cell]:
                continue
            yield cell, cost
            for neighbor, step in self._neighbors(cell, safe):
                new_cost = cost + step * self.config.resolution
                if new_cost < costs.get(neighbor, float("inf")):
                    costs[neighbor] = new_cost
                    previous[neighbor] = cell
                    heapq.heappush(queue, (new_cost, neighbor))

    def _frontiers(self, known, boundary):
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
        return {cell for cell in known
                if any((cell[0] + dx, cell[1] + dy) not in known
                       and (cell[0] + dx, cell[1] + dy) not in boundary
                       for dx, dy in offsets)}

    def _sight_clear(self, start, end, boundary):
        """Test the same half-cell samples as _raster_segment, stopping at a hit.

        Goal scoring calls this thousands of times per frame. Avoid allocating
        a NumPy point and a set for every sample along every sightline.
        """
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        count = max(1, int(math.ceil(math.hypot(dx, dy) /
                                     (self.config.resolution / 2))))
        x0 = float(start[0])
        y0 = float(start[1])
        resolution = self.config.resolution
        for index in range(count + 1):
            fraction = index / count
            cell = (math.floor((x0 + dx * fraction) / resolution),
                    math.floor((y0 + dy * fraction) / resolution))
            if cell in boundary:
                return False
        return True

    def _view_heading(self, point, visible):
        """Maximize boundary-frontier count, then total gain in one angular window."""
        angles = sorted((math.atan2(target[1] - point[1], target[0] - point[0]),
                         weight) for target, weight in visible)
        width = math.radians(self.config.sensor_fov_deg)
        count = len(angles)
        angles += [(angle + 2 * math.pi, weight) for angle, weight in angles]
        end = 0
        gain = boundary_gain = 0.0
        best = None
        for start in range(count):
            while (end < start + count
                   and angles[end][0] - angles[start][0] <= width + 1e-12):
                weight = angles[end][1]
                gain += weight
                boundary_gain += weight if weight == 2.0 else 0.0
                end += 1
            heading = (angles[start][0] + angles[end - 1][0]) / 2
            rank = (boundary_gain, gain)
            if best is None or rank > best[:2]:
                best = (boundary_gain, gain, math.atan2(math.sin(heading),
                                                       math.cos(heading)))
            weight = angles[start][1]
            gain -= weight
            boundary_gain -= weight if weight == 2.0 else 0.0
        return best[2], best[1], best[0]

    def goal_candidates(self, robot_pose, position_std=0.0):
        """Enumerate baseline-positive safe views without assigning a goal.

        Full search is intentional for training. Map evidence/visits are unchanged;
        visibility caches may be populated. Every returned path passes the same
        safety check as an assigned goal. Boundary priority remains a hard rule.
        """
        pose = np.asarray(robot_pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("robot_pose must be finite (x, y, yaw)")
        if not math.isfinite(position_std) or position_std < 0:
            raise ValueError("position_std must be finite and nonnegative")
        known, boundary, safe = self._safe_cells(position_std)
        if self._boundary_closed(self.cell(pose[:2])):
            return []
        candidates = []
        self._goal(pose, known, boundary, safe, candidates=candidates)
        return candidates

    def nearby_boundary_direction(self, pose):
        """Fit an unoriented local line using accumulated map cells only."""
        points = np.array([self.center(c) for c in self.boundary_hits], dtype=float).reshape(-1, 2)
        if len(points) < 3:
            return None, 0.0
        distances = np.linalg.norm(points - np.asarray(pose[:2]), axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] > self.config.sensor_range:
            return None, 0.0
        # Fit locally around the closest boundary, avoiding remote parallel walls.
        radius = min(self.config.sensor_range, max(.3, 4 * self.config.resolution))
        local = points[np.linalg.norm(points - points[nearest], axis=1) <= radius]
        if len(local) < 3:
            return None, 0.0
        centered = local - local.mean(axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered / len(local))
        if values[1] < self.config.resolution ** 2:
            return None, 0.0
        confidence = float((values[1] - values[0]) / (values[1] + values[0]))
        if confidence < .6:
            return None, 0.0
        direction = vectors[:, 1]
        return math.atan2(direction[1], direction[0]), confidence

    def _goal(self, pose, known, boundary, safe, performance=None, preferred_goal=None,
              candidates=None):
        started = perf_counter_ns()
        if performance is not None:
            performance["boundary_frontier_cells"] = 0
            performance["goal_boundary_gain"] = 0.0
            performance["goal_boundary_distance"] = ""
        start = self.cell(pose[:2])
        if start not in known or start in boundary:
            if performance is not None:
                performance["goal_reach_ms"] = 0.0
                performance["goal_frontier_ms"] = 0.0
                performance["goal_score_ms"] = 0.0
                performance["reachable_cells"] = 0
                performance["frontier_cells"] = 0
                performance["candidate_pairs"] = 0
                performance["sight_checks"] = 0
                performance["sight_cache_hits"] = 0
                performance["goal_search_radius"] = ""
                performance["goal_visible_gain"] = 0.0
            return None, np.empty((0, 2)), "robot position lacks confirmed grass"
        previous = {}
        # Only the confirmed-grass start is exempt from footprint clearance.
        path_cells = safe | {start}
        paths = self._reachable_paths(start, path_cells, previous)
        reach_ns = 0
        reachable_cells = 0
        started = perf_counter_ns()
        frontiers = self._frontiers(known, boundary)
        if performance is not None:
            performance["frontier_cells"] = len(frontiers)
        if not frontiers:
            if performance is not None:
                performance["goal_reach_ms"] = 0.0
                performance["reachable_cells"] = 0
                performance["goal_frontier_ms"] = (perf_counter_ns() - started) / 1e6
                performance["goal_score_ms"] = 0.0
                performance["candidate_pairs"] = 0
                performance["sight_checks"] = 0
                performance["sight_cache_hits"] = 0
                performance["goal_search_radius"] = ""
                performance["goal_visible_gain"] = 0.0
            return None, np.empty((0, 2)), "no grass-to-unknown frontier"
        sensor_range = self.config.sensor_range
        goal_spacing = self.config.goal_spacing
        bucket_width = max(sensor_range, self.config.boundary_frontier_distance)
        frontier_buckets = {}
        boundary_buckets = {}
        for cell in boundary:
            point = tuple(self.center(cell))
            bucket = (math.floor(point[0] / bucket_width),
                      math.floor(point[1] / bucket_width))
            boundary_buckets.setdefault(bucket, []).append(point)
        boundary_frontier_count = 0
        for cell in sorted(frontiers):
            point = tuple(self.center(cell))
            bucket = (math.floor(point[0] / bucket_width),
                      math.floor(point[1] / bucket_width))
            near_boundary = any(
                math.hypot(point[0] - b[0], point[1] - b[1])
                <= self.config.boundary_frontier_distance + 1e-12
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                for b in boundary_buckets.get((bucket[0] + dx,
                                               bucket[1] + dy), ()))
            frontier_buckets.setdefault(bucket, []).append(
                (cell, point, 2.0 if near_boundary else 1.0))
            boundary_frontier_count += int(near_boundary)
        if performance is not None:
            performance["boundary_frontier_cells"] = boundary_frontier_count
            performance["goal_frontier_ms"] = (perf_counter_ns() - started) / 1e6
        started = perf_counter_ns()
        best = None
        best_gain = 0.0
        best_boundary_gain = 0.0
        best_distance = ""
        selected_radius = "full"
        candidate_pairs = 0
        sight_checks = 0
        cache_hits = 0
        scored_ns = 0
        geometric_view = False
        preferred_cell = self.cell(preferred_goal[:2]) if preferred_goal is not None else None

        tangent, direction_confidence = self.nearby_boundary_direction(pose)

        def score_cell(cell, path_cost):
            nonlocal candidate_pairs, sight_checks, cache_hits
            if cell not in safe:
                return None
            point = tuple(self.center(cell))
            if preferred_goal is None:
                distance_from_pose = math.hypot(point[0] - pose[0], point[1] - pose[1])
                if (distance_from_pose < goal_spacing
                        or distance_from_pose <= self.config.goal_reached_distance):
                    return None
            visible = []
            bucket_x = math.floor(point[0] / bucket_width)
            bucket_y = math.floor(point[1] / bucket_width)
            nearby = (frontier for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      for frontier in frontier_buckets.get((bucket_x + dx,
                                                            bucket_y + dy), ()))
            for target_cell, target, weight in nearby:
                candidate_pairs += 1
                if math.hypot(target[0] - point[0],
                              target[1] - point[1]) > sensor_range:
                    continue
                sight_checks += 1
                key = (cell, target_cell)
                if key in self._visibility_cache:
                    clear = self._visibility_cache[key]
                    cache_hits += 1
                else:
                    clear = self._sight_clear(point, target, boundary)
                    if len(self._visibility_cache) < 100000:
                        self._visibility_cache[key] = clear
                if clear:
                    visible.append((target, weight))
            if not visible:
                return None
            if preferred_goal is None:
                heading, gain, boundary_gain = self._view_heading(point, visible)
                if tangent is not None and (candidates is not None or self.goal_scorer is not None):
                    options = []
                    for parallel in (tangent, math.atan2(math.sin(tangent + math.pi), math.cos(tangent + math.pi))):
                        weights = [weight for target, weight in visible
                                   if _bearing_difference(math.atan2(target[1] - point[1], target[0] - point[0]), parallel)
                                   <= math.radians(self.config.sensor_fov_deg) / 2 + 1e-12]
                        if weights:
                            options.append((sum(w for w in weights if w == 2.0), sum(weights),
                                            -_bearing_difference(parallel, pose[2]), parallel))
                    if options:
                        boundary_gain, gain, _, heading = max(options)
            else:
                heading = preferred_goal[2]
                weights = [weight for target, weight in visible
                           if _bearing_difference(math.atan2(target[1] - point[1],
                                                              target[0] - point[0]), heading)
                           <= math.radians(self.config.sensor_fov_deg) / 2 + 1e-12]
                gain = sum(weights)
                boundary_gain = sum(weight for weight in weights if weight == 2.0)
                if not gain:
                    return None
            # Distances beyond sensor range share the same capped penalty.
            distance = min((math.hypot(point[0] - b[0], point[1] - b[1])
                            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                            for b in boundary_buckets.get((bucket_x + dx,
                                                           bucket_y + dy), ())),
                           default=sensor_range)
            distance = min(distance, sensor_range) if boundary else 0.0
            effective_gain = boundary_gain if boundary_gain else gain
            score = (effective_gain / (1 + path_cost
                     + self.config.boundary_distance_weight * distance)
                     - 2 * self.goal_visits.get(cell, 0))
            if ((candidates is not None or self.goal_scorer is not None)
                    and score > 0 and cell not in self._excluded_goals):
                item = GoalScoreCandidate(cell, (point[0], point[1], float(heading)),
                    tuple(map(float, pose)), float(gain), float(boundary_gain),
                    float(path_cost), float(distance), self.goal_visits.get(cell, 0),
                    float(score), float(sensor_range), tangent, direction_confidence)
                if candidates is not None:
                    path = self._build_path(pose, cell, start, previous)
                    if self._path_is_safe(path, path_cells):
                        candidates.append(item)
                if self.goal_scorer is not None:
                    score = float(self.goal_scorer(item))
                    if not math.isfinite(score) or score <= 0:
                        raise ValueError("goal_scorer must return a finite positive score")
            return ((boundary_gain > 0, score, -path_cost, cell, heading),
                    gain, boundary_gain, distance)

        pending = None
        full_search = (candidates is not None
                       or getattr(self.goal_scorer, "full_goal_search", False))
        radii = ((float("inf"),) if preferred_goal is not None or full_search
                 else (*self.config.path_radii, float("inf")))
        for radius in radii:
            stage_best = None
            stage_gain = 0.0
            while True:
                if pending is None:
                    tick = perf_counter_ns()
                    pending = next(paths, None)
                    reach_ns += perf_counter_ns() - tick
                    if pending is None:
                        break
                cell, path_cost = pending
                if path_cost > radius:
                    break
                pending = None
                reachable_cells += 1
                if preferred_goal is not None and cell != preferred_cell:
                    continue
                tick = perf_counter_ns()
                candidate = score_cell(cell, path_cost)
                scored_ns += perf_counter_ns() - tick
                if candidate is None:
                    if preferred_goal is not None:
                        break
                    continue
                rank, gain, boundary_gain, distance = candidate
                geometric_view = True
                if rank[1] <= 0 or cell in self._excluded_goals:
                    if preferred_goal is not None:
                        break
                    continue
                if best is None or rank[:4] > best[:4]:
                    best = rank
                    best_gain = gain
                    best_boundary_gain = boundary_gain
                    best_distance = distance if boundary else ""
                # A local generic frontier must not hide a boundary frontier
                # reachable at a later radius. Exhaustion still permits fallback.
                qualifying_gain = boundary_gain if boundary_frontier_count else gain
                if (not full_search and qualifying_gain >= self.config.min_visible_gain and rank[1] > 0
                        and (stage_best is None or rank[:4] > stage_best[:4])):
                    stage_best = rank
                    stage_gain = gain
                    stage_boundary_gain = boundary_gain
                    stage_distance = distance
                if preferred_goal is not None:
                    selected_radius = "held"
                    break
            if stage_best is not None:
                best = stage_best
                best_gain = stage_gain
                best_boundary_gain = stage_boundary_gain
                best_distance = stage_distance if boundary else ""
                selected_radius = ("held" if preferred_goal is not None else
                                   radius if math.isfinite(radius) else "full")
                break
            if pending is None:
                break
        if performance is not None:
            performance["goal_reach_ms"] = reach_ns / 1e6
            performance["reachable_cells"] = reachable_cells
            performance["goal_score_ms"] = scored_ns / 1e6
            performance["candidate_pairs"] = candidate_pairs
            performance["sight_checks"] = sight_checks
            performance["sight_cache_hits"] = cache_hits
            performance["goal_search_radius"] = selected_radius
            performance["goal_visible_gain"] = best_gain
            performance["goal_boundary_gain"] = best_boundary_gain
            performance["goal_boundary_distance"] = best_distance
        if best is None:
            reason = ("reachable views exhausted in current map version" if geometric_view
                      else "no reachable safe viewing pose")
            return None, np.empty((0, 2)), reason
        _, _, _, target_cell, heading = best
        target = self.center(target_cell)
        path = self._build_path(pose, target_cell, start, previous)
        if not self._path_is_safe(path, path_cells):
            return None, np.empty((0, 2)), "unsafe path from robot pose"
        return (float(target[0]), float(target[1]), heading), path, "safe frontier viewpoint"

    def _boundary_closed(self, robot_cell):
        # Test confirmed boundary topology only; unknown remains floodable so
        # unknown terrain cannot falsely close the contour.
        boundary = {cell for cell, hits in self.boundary_hits.items() if hits >= 2}
        if len(boundary) < 8 or robot_cell in boundary:
            return False
        xs = [cell[0] for cell in boundary] + [robot_cell[0]]
        ys = [cell[1] for cell in boundary] + [robot_cell[1]]
        xmin, xmax = min(xs) - 2, max(xs) + 2
        ymin, ymax = min(ys) - 2, max(ys) + 2
        if (xmax - xmin + 1) * (ymax - ymin + 1) > 250000:
            return False
        seen = {robot_cell}
        queue = deque([robot_cell])
        while queue:
            x, y = queue.popleft()
            if x in (xmin, xmax) or y in (ymin, ymax):
                return False
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if neighbor not in seen and neighbor not in boundary:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return True

    def _fuse_observation(self, grass_cells, boundary_cells, timestamp):
        # Preserve the original diagnostics: newly stored evidence cells, not
        # newly confirmed grass. Classification progress is tracked separately.
        new_grass = len(grass_cells - boundary_cells - self.grass_hits.keys()
                        - self._grass_counterevidence.keys())
        new_boundary = len(boundary_cells - self.boundary_hits.keys())
        removed_boundary = 0
        for cell in boundary_cells:
            self.grass_hits.pop(cell, None)
            self._grass_counterevidence.pop(cell, None)
            self.boundary_hits[cell] = min(255, self.boundary_hits.get(cell, 0) + 1)
            self.last_seen[cell] = timestamp
        for cell in grass_cells - boundary_cells:
            self.last_seen[cell] = timestamp
            if cell in self.boundary_hits:
                count = self._grass_counterevidence.get(cell, 0) + 1
                self._grass_counterevidence[cell] = count
                if count < self.config.boundary_clear_observations:
                    continue
                del self.boundary_hits[cell]
                del self._grass_counterevidence[cell]
                self.grass_hits[cell] = count
                removed_boundary += 1
            else:
                self.grass_hits[cell] = min(max(255, self.config.grass_observations),
                                           self.grass_hits.get(cell, 0) + 1)
        if new_boundary or removed_boundary:
            self._visibility_cache.clear()
        return new_grass, new_boundary, removed_boundary

    def _release_goal(self):
        self.active_goal = None
        self.last_goal = None
        self._goal_progress_time = None
        self._goal_progress_distance = None
        self._goal_refresh_time = None
        self._goal_refresh_pose = None

    def _goal_refresh_reason(self, pose, timestamp):
        if self._goal_refresh_time is None or self._goal_refresh_pose is None:
            return "refresh_initial"
        if timestamp - self._goal_refresh_time >= self.config.goal_refresh_interval_s:
            return "refresh_interval"
        if math.hypot(pose[0] - self._goal_refresh_pose[0],
                      pose[1] - self._goal_refresh_pose[1]) >= self.config.goal_refresh_distance:
            return "refresh_distance"
        if (_bearing_difference(pose[2], self._goal_refresh_pose[2])
                >= math.radians(self.config.goal_refresh_yaw_deg)):
            return "refresh_yaw"
        return ""

    def _exploration_decision(self, pose, timestamp, known, boundary, safe,
                              closed, map_changed, performance):
        release_reason = ""
        action = "none"
        refresh_previous = None
        if self.active_goal is not None:
            cell = self.cell(self.active_goal[:2])
            distance = math.hypot(self.active_goal[0] - pose[0],
                                  self.active_goal[1] - pose[1])
            if distance <= self.config.goal_reached_distance:
                self.goal_visits[cell] = self.goal_visits.get(cell, 0) + 1
                release_reason = "arrived"
                self._release_goal()
            elif (map_changed or self._goal_progress_time is None
                  or self._goal_progress_distance - distance
                  >= self.config.goal_progress_distance):
                self._goal_progress_time = timestamp
                self._goal_progress_distance = distance
            elif timestamp - self._goal_progress_time >= self.config.goal_progress_timeout_s:
                self._excluded_goals.add(cell)
                release_reason = "progress_timeout"
                self._release_goal()

        if self.active_goal is not None:
            refresh_reason = self._goal_refresh_reason(pose, timestamp)
            if refresh_reason:
                # Reselecting the same cell must not postpone timeout forever.
                refresh_previous = (self.last_goal, self._goal_progress_time,
                                    self._goal_progress_distance)
                release_reason = refresh_reason
                self._release_goal()

        if closed:
            if self.active_goal is not None:
                release_reason = "boundary_closed"
            self._release_goal()
            status, reason, goal, path = ("complete",
                "confirmed boundary encloses robot on grid", None, np.empty((0, 2)))
        elif self.cell(pose[:2]) not in known or self.cell(pose[:2]) in boundary:
            if self.active_goal is not None:
                release_reason = "robot_unsafe"
            self._release_goal()
            status, reason, goal, path = ("waiting_for_observation",
                "robot position lacks confirmed grass", None, np.empty((0, 2)))
        else:
            goal = None
            held_performance = {} if performance is not None else None
            if self.active_goal is not None:
                goal, path, reason = self._goal(pose, known, boundary, safe,
                    held_performance, preferred_goal=self.active_goal)
                if goal is not None:
                    action = "held"
                    reason = "holding safe frontier viewpoint"
                    if performance is not None:
                        performance.update(held_performance)
                else:
                    release_reason = "goal_invalid: " + reason
                    self._release_goal()
            if goal is None:
                goal, path, reason = self._goal(pose, known, boundary, safe, performance)
                if performance is not None and held_performance:
                    for key in ("goal_reach_ms", "goal_frontier_ms", "goal_score_ms",
                                "candidate_pairs", "sight_checks", "sight_cache_hits"):
                        performance[key] += held_performance.get(key, 0)
                if goal is not None:
                    self.active_goal = goal
                    self.last_goal = self.cell(goal[:2])
                    self._goal_progress_time = timestamp
                    self._goal_progress_distance = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
                    if refresh_previous is not None and self.last_goal == refresh_previous[0]:
                        self._goal_progress_time = refresh_previous[1]
                        self._goal_progress_distance = refresh_previous[2]
                    self._goal_refresh_time = timestamp
                    self._goal_refresh_pose = tuple(pose)
                    action = "selected"
            status = ("exploring" if goal is not None else
                      "stalled" if reason == "reachable views exhausted in current map version"
                      else "incomplete_unreachable")
        if performance is not None:
            performance["goal_action"] = action
            performance["goal_release_reason"] = release_reason
            performance["goal_visit_count"] = (self.goal_visits.get(self.last_goal, 0)
                                                if self.last_goal is not None else 0)
        return status, reason, goal, path

    def update(self, grass_points, boundary_points, robot_pose, timestamp,
               sensor_to_world=None, position_std=0.0, performance=None):
        """Fuse one observation and return the next safe exploration decision.

        ``robot_pose`` is global (x, y, yaw). Point arrays are either global
        XY or sensor XYZ with ``sensor_to_world``. Each cell receives at most
        one evidence hit from a frame, regardless of point density. Equal
        timestamps re-evaluate the pose but do not fuse evidence again;
        backwards timestamps are rejected. Classification changes reset visit
        penalties, and stalled means reachable views have been exhausted.
        """
        started = perf_counter_ns()
        stage = started
        if performance is not None:
            performance.update({name: 0 for name in (
                "goal_reach_ms", "goal_frontier_ms", "goal_score_ms", "reachable_cells",
                "frontier_cells", "candidate_pairs", "sight_checks", "sight_cache_hits",
                "goal_visible_gain", "boundary_frontier_cells", "goal_boundary_gain")})
            performance.update(goal_search_radius="", goal_boundary_distance="")
        pose = np.asarray(robot_pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("robot_pose must be finite (x, y, yaw)")
        if (not np.isfinite(timestamp) or not np.isfinite(position_std)
                or position_std < 0 or (self.last_timestamp is not None
                                         and timestamp < self.last_timestamp)):
            raise ValueError("timestamp must advance and position_std must be nonnegative")
        grass = points_to_world_xy(grass_points, sensor_to_world)
        observed_boundary = points_to_world_xy(boundary_points, sensor_to_world)
        if performance is not None:
            performance["transform_ms"] = (perf_counter_ns() - stage) / 1e6
            performance["grass_points"] = len(grass)
            performance["boundary_points"] = len(observed_boundary)
        stage = perf_counter_ns()
        observation_fused = self.last_timestamp is None or timestamp > self.last_timestamp
        grass_cells = {self.cell(point) for point in grass}
        boundary_cells = self._boundary_observation_cells(observed_boundary)
        if performance is not None:
            performance["raster_ms"] = (perf_counter_ns() - stage) / 1e6
            performance["observed_grass_cells"] = len(grass_cells)
            performance["observed_boundary_cells"] = len(boundary_cells)
        stage = perf_counter_ns()
        new_grass = new_boundary = removed_boundary = 0
        if observation_fused:
            new_grass, new_boundary, removed_boundary = self._fuse_observation(
                grass_cells, boundary_cells, timestamp)
            self.last_timestamp = float(timestamp)
        if performance is not None:
            performance["evidence_ms"] = (perf_counter_ns() - stage) / 1e6
        stage = perf_counter_ns()
        known, boundary, safe = self._safe_cells(position_std)
        signature = (frozenset(known), frozenset(boundary))
        map_changed = signature != self._map_signature
        if map_changed:
            self.map_version += 1
            self._map_signature = signature
            self.goal_visits.clear()
            self._excluded_goals.clear()
        if performance is not None:
            performance["safe_ms"] = (perf_counter_ns() - stage) / 1e6
            performance["known_grass_cells"] = len(known)
            performance["mapped_boundary_cells"] = len(boundary)
            performance["safe_cells"] = len(safe)
            performance["map_version"] = self.map_version
            performance["observation_fused"] = int(observation_fused)
            performance["removed_boundary_cells"] = removed_boundary
        stage = perf_counter_ns()
        closed = self._boundary_closed(self.cell(pose[:2]))
        if performance is not None:
            performance["closure_ms"] = (perf_counter_ns() - stage) / 1e6
        stage = perf_counter_ns()
        status, reason, goal, path = self._exploration_decision(
            pose, timestamp, known, boundary, safe, closed, map_changed, performance)
        if performance is not None:
            performance["goal_ms"] = (perf_counter_ns() - stage) / 1e6
        stage = perf_counter_ns()
        result = ExplorationResult(status, reason, goal, path, frozenset(known),
                                   frozenset(boundary), frozenset(safe), closed,
                                   new_grass, new_boundary)
        if performance is not None:
            performance["snapshot_ms"] = (perf_counter_ns() - stage) / 1e6
            performance["update_ms"] = (perf_counter_ns() - started) / 1e6
            performance["status"] = status
        return result

    def trajectory_is_safe(self, poses_xytheta, position_std=0.0):
        """Check every touched grid cell of a piecewise-linear center path.

        Curved motion between poses is the caller's responsibility; use
        command_is_safe for the unicycle arc model and its deviation bound.
        """
        poses = np.asarray(poses_xytheta, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
            raise ValueError("poses_xytheta must have shape (N, 3)")
        if not np.isfinite(position_std) or position_std < 0:
            raise ValueError("position_std must be finite and nonnegative")
        _, _, safe = self._safe_cells(position_std)
        return self._path_is_safe(poses[:, :2], safe)

    def command_is_safe(self, pose_xytheta, v, omega, horizon_s,
                        step_s=0.05, position_std=0.0):
        """Predict a unicycle command and reject any unknown or boundary sweep."""
        pose = np.asarray(pose_xytheta, dtype=float)
        if (pose.shape != (3,) or not np.isfinite(pose).all()
                or not np.isfinite(v) or not np.isfinite(omega)
                or not np.isfinite(horizon_s) or horizon_s < 0
                or not np.isfinite(step_s) or step_s <= 0
                or not np.isfinite(position_std) or position_std < 0):
            raise ValueError("command and pose must be finite with valid time limits")
        steps = max(1, int(math.ceil(horizon_s / step_s)),
                    int(math.ceil(abs(omega) * horizon_s / (math.pi / 8))),
                    int(math.ceil(abs(v) * horizon_s /
                                  (self.config.resolution / 2))))
        dt = horizon_s / steps
        angle = omega * dt
        # 1-cos(a/2) = 2*sin(a/4)^2 avoids cancellation for tiny angles.
        sagitta = (abs(v / omega) * 2 * math.sin(angle / 4) ** 2
                    if omega else 0.0)
        _, _, safe = self._safe_cells(position_std, extra_margin=sagitta)
        predicted = [pose.copy()]
        x, y, heading = pose
        for _ in range(steps):
            half_angle = angle / 2
            travel = v * dt * (math.sin(half_angle) / half_angle if half_angle else 1.0)
            x += travel * math.cos(heading + half_angle)
            y += travel * math.sin(heading + half_angle)
            heading += angle
            predicted.append((x, y, heading))
        return self._path_is_safe(np.asarray(predicted)[:, :2], safe)
