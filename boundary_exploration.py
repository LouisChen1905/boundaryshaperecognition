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

    def __post_init__(self):
        if (self.robot_radius <= 0 or self.sensor_range <= 0 or self.resolution <= 0
                or self.boundary_clearance < 0 or self.pose_margin < 0
                or self.grass_observations < 1 or self.boundary_link_gap <= 0
                or self.goal_spacing < 0
                or not 0 < self.sensor_fov_deg <= 360
                or not math.isfinite(self.min_visible_gain)
                or self.min_visible_gain <= 0):
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


class BoundaryExplorer:
    """Build a sparse evidence grid and select safe boundary-viewing poses."""

    def __init__(self, config):
        self.config = config
        self.grass_hits = {}
        self.boundary_hits = {}
        self.last_seen = {}
        self.last_timestamp = None
        self.last_goal = None
        self.goal_visits = {}
        self._visibility_cache = {}

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
        # Include any cell whose square could touch the robot's disk.
        reach = radius + self.config.resolution / math.sqrt(2)
        cells = int(math.ceil(reach / self.config.resolution))
        return [(dx, dy) for dx in range(-cells, cells + 1)
                for dy in range(-cells, cells + 1)
                if math.hypot(dx, dy) * self.config.resolution <= reach]

    def _safe_cells(self, position_std):
        known = {cell for cell, hits in self.grass_hits.items()
                 if hits >= self.config.grass_observations and cell not in self.boundary_hits}
        boundary = set(self.boundary_hits)
        footprint = self.config.robot_radius + max(self.config.pose_margin,
                                                   2 * position_std)
        body_offsets = self._disk_offsets(footprint)
        boundary_offsets = self._disk_offsets(footprint + self.config.boundary_clearance)
        safe = set()
        for x, y in known:
            if any((x + dx, y + dy) not in known for dx, dy in body_offsets):
                continue
            if any((x + dx, y + dy) in boundary for dx, dy in boundary_offsets):
                continue
            safe.add((x, y))
        return known, boundary, safe

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

    def _goal(self, pose, known, boundary, safe, performance=None):
        started = perf_counter_ns()
        start = self.cell(pose[:2])
        if start not in safe:
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
            return None, np.empty((0, 2)), "robot footprint lacks confirmed grass"
        previous = {}
        paths = self._reachable_paths(start, safe, previous)
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
        bucket_width = sensor_range
        frontier_buckets = {}
        boundary_buckets = {}
        for cell in boundary:
            point = tuple(self.center(cell))
            bucket = (math.floor(point[0] / bucket_width),
                      math.floor(point[1] / bucket_width))
            boundary_buckets.setdefault(bucket, []).append(point)
        for cell in frontiers:
            point = tuple(self.center(cell))
            bucket = (math.floor(point[0] / bucket_width),
                      math.floor(point[1] / bucket_width))
            near_boundary = any(
                math.hypot(point[0] - b[0], point[1] - b[1]) <= sensor_range
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                for b in boundary_buckets.get((bucket[0] + dx,
                                               bucket[1] + dy), ()))
            frontier_buckets.setdefault(bucket, []).append(
                (cell, point, 2.0 if near_boundary else 1.0))
        if performance is not None:
            performance["goal_frontier_ms"] = (perf_counter_ns() - started) / 1e6
        started = perf_counter_ns()
        best = None
        best_gain = 0.0
        selected_radius = "full"
        candidate_pairs = 0
        sight_checks = 0
        cache_hits = 0
        scored_ns = 0

        def score_cell(cell, path_cost):
            nonlocal candidate_pairs, sight_checks, cache_hits
            point = tuple(self.center(cell))
            if math.hypot(point[0] - pose[0], point[1] - pose[1]) < goal_spacing:
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
            heading_target = max(visible, key=lambda item: item[1])[0]
            heading = math.atan2(heading_target[1] - point[1],
                                 heading_target[0] - point[0])
            half_fov = math.radians(self.config.sensor_fov_deg) / 2
            gain = sum(weight for target, weight in visible
                       if _bearing_difference(math.atan2(target[1] - point[1],
                                                         target[0] - point[0]), heading)
                       <= half_fov)
            score = gain / (1 + path_cost) - 2 * self.goal_visits.get(cell, 0)
            return (score, -path_cost, cell, heading), gain

        pending = None
        for radius in (*self.config.path_radii, float("inf")):
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
                tick = perf_counter_ns()
                candidate = score_cell(cell, path_cost)
                scored_ns += perf_counter_ns() - tick
                if candidate is None:
                    continue
                rank, gain = candidate
                if best is None or rank[:3] > best[:3]:
                    best = rank
                    best_gain = gain
                if (gain >= self.config.min_visible_gain and rank[0] > 0
                        and (stage_best is None or rank[:3] > stage_best[:3])):
                    stage_best = rank
                    stage_gain = gain
            if stage_best is not None:
                best = stage_best
                best_gain = stage_gain
                selected_radius = radius if math.isfinite(radius) else "full"
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
        if best is None:
            return None, np.empty((0, 2)), "no reachable safe viewing pose"
        _, _, target_cell, heading = best
        target = self.center(target_cell)
        cells = [target_cell]
        while cells[-1] != start:
            cells.append(previous[cells[-1]])
        cells.reverse()
        path = np.array([self.center(cell) for cell in cells])
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

    def update(self, grass_points, boundary_points, robot_pose, timestamp,
               sensor_to_world=None, position_std=0.0, performance=None):
        """Fuse one observation and return the next safe exploration decision.

        ``robot_pose`` is global (x, y, yaw). Point arrays are either global
        XY or sensor XYZ with ``sensor_to_world``. Each cell receives at most
        one evidence hit from a frame, regardless of point density.
        """
        started = perf_counter_ns()
        stage = started
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
        self.last_timestamp = float(timestamp)
        grass_cells = {self.cell(point) for point in grass}
        boundary_cells = self._boundary_observation_cells(observed_boundary)
        if performance is not None:
            performance["raster_ms"] = (perf_counter_ns() - stage) / 1e6
            performance["observed_grass_cells"] = len(grass_cells)
            performance["observed_boundary_cells"] = len(boundary_cells)
        stage = perf_counter_ns()
        new_grass = len(grass_cells - self.grass_hits.keys())
        new_boundary = len(boundary_cells - self.boundary_hits.keys())
        if new_boundary:
            self._visibility_cache.clear()
        for cell in grass_cells:
            self.grass_hits[cell] = min(255, self.grass_hits.get(cell, 0) + 1)
            self.last_seen[cell] = timestamp
        for cell in boundary_cells:
            self.boundary_hits[cell] = min(255, self.boundary_hits.get(cell, 0) + 1)
            self.last_seen[cell] = timestamp
        if performance is not None:
            performance["evidence_ms"] = (perf_counter_ns() - stage) / 1e6
        stage = perf_counter_ns()
        known, boundary, safe = self._safe_cells(position_std)
        if performance is not None:
            performance["safe_ms"] = (perf_counter_ns() - stage) / 1e6
            performance["known_grass_cells"] = len(known)
            performance["mapped_boundary_cells"] = len(boundary)
            performance["safe_cells"] = len(safe)
        stage = perf_counter_ns()
        closed = self._boundary_closed(self.cell(pose[:2]))
        if performance is not None:
            performance["closure_ms"] = (perf_counter_ns() - stage) / 1e6
        stage = perf_counter_ns()
        if closed:
            status, reason, goal, path = ("complete", "confirmed closed contour",
                                          None, np.empty((0, 2)))
        else:
            goal, path, reason = self._goal(pose, known, boundary, safe,
                                           performance)
            if goal is not None:
                status = "exploring"
                cell = self.cell(goal[:2])
                if (self.last_goal == cell and not (new_grass or new_boundary)
                        and np.linalg.norm(np.asarray(goal[:2]) - pose[:2])
                        <= self.config.goal_spacing):
                    self.goal_visits[cell] = self.goal_visits.get(cell, 0) + 1
                self.last_goal = cell
            elif self.cell(pose[:2]) not in safe:
                status = "waiting_for_observation"
            else:
                status = "incomplete_unreachable"
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
        """Check a local planner's sampled swept path, including interpolation.

        The caller must sample its dynamic model densely enough between poses;
        this function additionally interpolates each center segment at half a
        grid cell. A false result means the command must not be executed.
        """
        poses = np.asarray(poses_xytheta, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
            raise ValueError("poses_xytheta must have shape (N, 3)")
        if not np.isfinite(position_std) or position_std < 0:
            raise ValueError("position_std must be finite and nonnegative")
        _, _, safe = self._safe_cells(position_std)
        if not len(poses):
            return False
        for a, b in zip(poses[:-1], poses[1:]):
            distance = np.linalg.norm(b[:2] - a[:2])
            steps = max(1, int(math.ceil(distance / (self.config.resolution / 2))))
            if any(self.cell(a[:2] + (b[:2] - a[:2]) * (i / steps)) not in safe
                   for i in range(steps + 1)):
                return False
        return self.cell(poses[0, :2]) in safe and self.cell(poses[-1, :2]) in safe

    def command_is_safe(self, pose_xytheta, v, omega, horizon_s,
                        step_s=0.05, position_std=0.0):
        """Predict a unicycle command and reject any unknown or boundary sweep."""
        pose = np.asarray(pose_xytheta, dtype=float)
        if (pose.shape != (3,) or not np.isfinite(pose).all()
                or not np.isfinite(v) or not np.isfinite(omega)
                or not np.isfinite(horizon_s) or horizon_s < 0
                or not np.isfinite(step_s) or step_s <= 0):
            raise ValueError("command and pose must be finite with valid time limits")
        steps = max(1, int(math.ceil(horizon_s / step_s)),
                    int(math.ceil(abs(v) * horizon_s /
                                  (self.config.resolution / 2))))
        dt = horizon_s / steps
        predicted = [pose.copy()]
        x, y, heading = pose
        for _ in range(steps):
            if abs(omega) < 1e-9:
                x += v * math.cos(heading) * dt
                y += v * math.sin(heading) * dt
            else:
                next_heading = heading + omega * dt
                x += v / omega * (math.sin(next_heading) - math.sin(heading))
                y += v / omega * (math.cos(heading) - math.cos(next_heading))
            heading += omega * dt
            predicted.append((x, y, heading))
        return self.trajectory_is_safe(np.asarray(predicted), position_std)
