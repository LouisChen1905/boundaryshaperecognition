# Boundary shape recognition

## Grass-only boundary exploration

`boundary_exploration.py` is an independent, stateful grid explorer. Create
`BoundaryExplorer(ExplorationConfig(robot_radius=..., sensor_range=...))`, then
call `update(grass_points, boundary_points, (x, y, yaw), timestamp)` for each
time-aligned frame. Two-column point arrays are global XY. Three-column camera
XYZ arrays require a full 4x4 `sensor_to_world` transform; camera Y is not
discarded. The result contains the sparse known-grass, boundary, and safe-cell
sets, a boundary-closure flag, the next observation pose and a path, or a
status explaining why no path is available. Unknown cells are implicit and
never traversable. Evidence is counted once per cell per frame; grass needs
two observations by default, while even one boundary observation blocks a
cell. Consecutive boundary points are connected only across gaps no larger
than `boundary_link_gap`.

Equal timestamps re-evaluate the robot pose without adding evidence; backwards
timestamps are rejected by `update`. A boundary observation overrides grass in
the same cell and clears its old grass evidence. Three subsequent grass-only
observations of that cell retract the boundary by default
(`boundary_clear_observations`); a new boundary observation resets this count,
while missing observations leave it unchanged. Grass is then confirmed using
only the new observations. Evidence does not expire with time. These inputs
only distinguish grass and boundary, not other terrain classes.

```python
from boundary_exploration import BoundaryExplorer, ExplorationConfig

explorer = BoundaryExplorer(ExplorationConfig(robot_radius=0.2, sensor_range=2.0))
result = explorer.update(grass_world_xy, boundary_world_xy,
                         (robot_x, robot_y, robot_yaw), timestamp_s)
if result.status == "exploring":
    send_goal_to_local_planner(result.goal_pose, result.path_xy)
```

The robot's circular footprint, pose margin, and boundary clearance are
applied before routing. A safe cell now guarantees clearance for every center
position in its square, using minimum square-to-square distance; touching
unknown terrain or the boundary clearance limit is unsafe. This deliberately
reduces usable space in narrow passages. Paths begin at the actual robot pose.
`trajectory_is_safe` checks every grid cell touched by each straight segment,
including edge and corner contacts. `command_is_safe` subdivides unicycle arcs
to at most pi/8 radians per segment and adds the arc-to-chord sagitta to the
footprint margin. A general curved local-planner trajectory needs its own
deviation bound; a list of poses alone describes straight connecting segments.
Goals are selected from reachable safe cells for
their view of grass-to-unknown frontiers. Frontiers within
`boundary_frontier_distance` (default 0.3 m, measured between cell centers) of
observed boundary are treated as potential unknown boundary continuations.
This is a proximity heuristic, not a prediction of the unseen contour.
Among positive-score viewpoints, priority frontiers rank ahead of generic views.
Range and boundary occlusion are checked before choosing the sensor heading;
an angular sweep maximizes priority-frontier gain, then total gain within the
sensor field of view. `trajectory_is_safe` and `command_is_safe` check a local planner's
proposed motion before execution. If the sensor cannot confirm grass beneath
the robot at startup, the explorer returns `waiting_for_observation`; an
externally verified initial grass disk can be supplied explicitly through
`seed_verified_grass_disk`. A pose alone does not certify grass. The algorithm
uses a circular footprint and a finite-range sensor model; callers must supply
calibrated dimensions, localization uncertainty, and camera extrinsics. A
closed confirmed boundary reports `complete`: confirmed boundary cells enclose
the robot under the grid flood-fill test. This does not establish complete
interior coverage or discovery of every hole or disconnected region. Completion
is evaluated again after boundary corrections. An open contour with no reachable
safe viewpoint reports `incomplete_unreachable`; reachable views exhausted by
visit penalties or progress timeouts report `stalled` with no goal or path.

Goal search expands a single safe-grid shortest-path search through configurable
path radii (default 1, 2, and 3 m). Each stage scores only nearby frontiers via
spatial buckets. When priority frontiers exist, only their visible gain qualifies
a stage for early stopping; a nearby generic view cannot stop the search before
a more distant boundary view is considered. Otherwise total gain is used.
It stops when this gain reaches `min_visible_gain` (default 2; each priority
frontier contributes 2, each generic frontier 1) and the score is positive.
Within each priority class the score is
`gain / (1 + path_length + boundary_distance_weight * boundary_distance) - 2 * visits`.
Here gain uses priority frontiers only if any are visible, otherwise all visible
frontiers. Boundary distance is measured from the viewpoint to observed boundary
cell centers, capped at sensor range, and omitted when no boundary is known.
The distance weight defaults to 2.0; zero disables the distance preference.
This favors closer views while retaining path cost and repeat-visit penalties.
The first qualified path radius still bounds the optimization; this is not a
global nearest-boundary search. If no stage qualifies, all reachable safe cells
are searched, with positive-score generic views as fallback when no positive
priority view is available. Nonpositive views are never assigned. Set
`path_radii` and `min_visible_gain` in `ExplorationConfig`, or use
`--goal-path-radii 1 2 3` and `--min-visible-gain 2` in the JSONL replay.
The replay also exposes `--boundary-frontier-distance` and
`--boundary-distance-weight`. Performance CSV fields `boundary_frontier_cells`,
`goal_boundary_gain`, and `goal_boundary_distance` report the priority-frontier
count, selected visible priority gain, and capped boundary distance in meters.
These defaults are starting values; calibrate them against representative logs.

An assigned goal is held at its original position and heading while it remains
reachable, safe, positive-scoring and makes progress, until the next refresh.
New candidate selection is triggered by any of: elapsed sensor time
`goal_refresh_interval_s` (2 s), displacement from the last selection pose
`goal_refresh_distance` (0.3 m), or shortest wrapped heading change
`goal_refresh_yaw_deg` (20 degrees). These thresholds are checked on each
`update`; pausing the input does not run a wall-clock timer. Re-scoring can
select the same optimal cell and recomputes its heading. Selecting the same
cell does not reset its progress timeout, and map changes do not postpone the
periodic refresh. CSV `goal_release_reason` records `refresh_interval`,
`refresh_distance`, or `refresh_yaw`; `goal_action` becomes `selected`.
Replay flags are `--goal-refresh-interval-s`, `--goal-refresh-distance`, and
`--goal-refresh-yaw-deg`.

Arrival is checked before
reselection using `goal_reached_distance` (0.15 m, position only); each assignment
increments its visit count once. New goals must be outside that arrival radius
as well as satisfying `goal_spacing`. A goal is released after
`goal_progress_timeout_s` (15 s) without a cumulative improvement of
`goal_progress_distance` (0.1 m) in distance to it, and excluded for the current
map version. Confirmed grass/boundary classification changes increment
`map_version`, clear visit penalties and exclusions, and reset the progress
timer. Extra hits on unchanged classifications do not count as progress.
`stalled` automatically re-evaluates when new classifications become available.
Replay flags are `--goal-reached-distance`, `--goal-progress-timeout-s`,
`--goal-progress-distance`, and `--boundary-clear-observations`.

Detailed fitting steps and direction conventions are documented in [拟合算法说明.md](拟合算法说明.md).

Run `python replay_boundary.py` to load the log and replay each frame simultaneously in global XY (left) and camera XZ (right) coordinates. The green dots are measured boundary points, the blue curves are independently estimated boundary trends, and the bright orange arrows and labels show the **tracking reference direction** near each purple curve's midpoint. The red marker and line show the robot position and heading in the global view. Both views use the same frame controls: Play/Pause, Previous/Next, the slider, or the left/right arrow keys. `--speed 2` plays at twice the logged rate.

Navigation data packages in JSONL format use a live exploration replay. For a bounded inspection of the supplied file, run `python replay_boundary.py navigation-data.jsonl --max-records 1000`. The reader aligns `visual_boundary_points.data.timestamp` to `location.data.timestamp`, both in milliseconds. Exact timestamp matches use that pose; otherwise it interpolates position linearly and attitude with shortest-arc quaternion SLERP between bracketing poses. Full camera XYZ boundary and grass clouds use the same aligned pose. It never uses arrival order, the latest pose, `system_ms`, or extrapolation as a substitute.

The default maximum interval between the two location samples is 100 ms,
configurable with `--pose-max-gap-ms` (or `iter_data_package(...,
pose_max_gap_s=0.1)`). The reader keeps at most 2048 pose samples and 128 pending
visual packets. A visual packet can wait for a future bracket within the input
record limit. Invalid sample timestamps, excessive gaps, packets older than
retained history, pending queue overflow, and unmatched frames at EOF or the
record limit are skipped with diagnostics. No additional lines are read beyond
`--max-records` to obtain a bracket. Optional `alignment_stats` collects counts;
the CLI headless profile prints them. CSV fields `pose_alignment`
(`exact`/`interpolated`), `pose_before_timestamp`, `pose_after_timestamp` (seconds),
and `pose_gap_ms` make the selected bracket reviewable.

The current map and 50 recent frame snapshots are retained. The left view shows confirmed grass (light green), observed boundary (red), robot-center safe cells (teal), the time-aligned robot, a selected goal and safe path when available, and the original single-frame fit (blue). The right view shows camera boundary and grass points. Play/Pause, Previous/Next and arrow keys work with the bounded history; moving backward only displays a cached result and does not add evidence again. `--speed 2` doubles playback speed.

The replay uses example values `--exploration-robot-radius 0.2` and `--exploration-sensor-range 2.0`; set these to measured values before interpreting goals or paths. `--grid-resolution` controls map detail. If the starting grass patch has been independently verified, `--verified-start-radius R` seeds that patch. Without verified grass under the robot, the replay correctly reports `waiting_for_observation` even if grass is visible farther ahead. The JSONL replay displays exploration output but does not send commands to a robot.

To locate replay stalls, add `--performance-log exploration_performance.csv`. Each visual frame records JSONL reading, coordinate transformation, rasterization, evidence fusion, safe-cell calculation, closure check, goal search, artist updates, and actual canvas rendering in milliseconds, alongside point and cell counts. Rendering is measured with a synchronous canvas draw while logging is enabled, so the log is a diagnostic run and may play more slowly. For algorithm-only timing without a plot, use `--profile-only --max-records 1000 --performance-log exploration_performance.csv`; `artist_update_ms` and `render_ms` are then blank. `callback_ms` covers one UI frame callback, while `update_ms` covers only `BoundaryExplorer.update()`.

Code using `BoundaryExplorer` directly can pass an empty dictionary as `performance=timings` to `update()`; the call fills it with the same algorithm-stage timings and cell counts without writing a file.

For a frame with N input points, G confirmed grass cells, B mapped boundary cells, S safe cells, F grass frontiers, and K footprint offsets, rasterization is approximately O(N), safe-cell checks O(GK), and closure flood-fill O(A) for the bounded map rectangle A. If the robot is on a safe cell, reachable-path search costs O(S log S); goal scoring can cost O(FB + SF L/r), where L is a sightline length and r is grid resolution. This last term can dominate as explored area grows because many candidate viewpoints repeatedly test frontier visibility. Unknown or unsafe start cells skip goal search entirely.

The visibility check stops at the first blocked cell and reuses results between frames while the mapped boundary cells are unchanged. Adding or retracting boundary cells invalidates the cache. Its size is capped at 100,000 viewpoint/frontier pairs. The CSV reports `sight_checks`, `sight_cache_hits`, `goal_search_radius`, and `goal_visible_gain` to support search cost and threshold calibration. `goal_search_radius` is `held` when validating the current goal rather than selecting a new one. Additional fields `map_version`, `observation_fused`, `removed_boundary_cells`, `goal_action` (`selected`, `held`, or `none`), `goal_release_reason`, and `goal_visit_count` describe evidence and goal lifecycle changes; the visit count is for the currently assigned goal, or zero when none is assigned. Duplicate frames remain visible and logged without fusion; backwards visual timestamps and invalid sample timestamps are skipped.

The dashed purple line is one tracking reference shown in both views. It is computed in global XY by moving each fitted boundary point 0.4 m along its local normal toward the lawn side. Its point order is then chosen so the boundary lies on the **right** of the tracking direction. On a closed outer lawn boundary, this corresponds to counterclockwise travel with lawn on the left. Grass points determine the lawn side when available; otherwise the robot position is a provisional interior-side estimate, marked as such in the plot. Curves with bends too tight for a smooth 0.4 m inner offset fall back to a simpler trend. Consecutive frames with nearly unchanged robot position and yaw are fitted together to keep the reference stable while the robot is stationary. The global reference is then mapped into camera XZ using that frame's pose and camera extrinsics, assuming camera Y=0 as for the logged points. It is not a separate 0.4 m offset of the camera XZ fit.

The figure header separately reports the recorded pose loop direction and the direction of the sampled reference midpoints. These summaries require a nearly closed route; they report "open or incomplete" when the start/end gap is too large. A positive signed area and pose yaw increase indicate CCW in global XY. This geometric summary does not replace verification against grass coverage or the robot's driving limits.

Optional grass points can be included in each log line after `boundary_xz` as `grass_xz: [(x, z); (x, z)]`, before `location info`. Alternatively pass `--grass-csv grass.csv` with either `frame,x,y` (global XY) or `frame,camera_x,camera_z` (camera XZ) columns; `frame` is 1-based. The grass cloud is shown in light green. At least five points with 70% on a consistent side are needed to override the robot-side estimate. When enough grass points are present, the plot reports the fraction lying to the left of the directed reference and the fraction of reference samples with grass within 0.25 m. Set `--grass-support-radius` to match the real point-cloud density. These are supporting observations, not proof that every reference point is inside grass. The provided sample log has no grass points, so it cannot verify that the reference is inside the lawn.

Set `--min-turn-radius R` when the robot's minimum turning radius is known. The boundary fit is then limited to curvature at most `1 / (R + 0.4)` m⁻¹, with the existing 1.5 m⁻¹ cap retained. This is a geometric turning-radius constraint; it does not model steering transients, speed, or obstacles. Without `R`, trackability cannot be guaranteed from the point cloud alone.

Set `--robot-width W` to report the nominal distance from the tracked robot's right body edge to the fitted boundary: `0.4 - W/2` meters. For an asymmetric robot, use `--right-half-width H` instead to calculate `0.4 - H`. The options are mutually exclusive. A nonpositive value is highlighted in red. This assumes the reference tracks the robot center and ignores pose error, body shape, and corner sweep during turns. The provided log does not specify robot width.

For every frame, `boundary_shape.recognize(points_xyz)` estimates a centerline from global boundary points:

1. Use principal component analysis to find the long axis of the cloud.
2. Divide that axis into bins and take the median transverse coordinate in each populated bin.
3. Reject transverse outliers and fit a cubic, quadratic, or linear trend through the bin medians, subject to a curvature limit.
4. Calculate tangent directions, length, net turning angle, and a rough shape label: straight, left bend, right bend, or S bend.

`orient_shape(shape, robot_xy)` orders the blue centerline from the endpoint closest to the robot; its left/right shape labels refer to that order. The orange tracking arrow follows a separate right-side-boundary rule and may point opposite to the blue centerline's order. Direction angles are counterclockwise from global +X in the global view, and from camera +X toward +Z in the camera view. Frames with too few points or too little extent return `None` and display “insufficient shape data”.

The analysis is a geometric approximation for a single, mostly continuous boundary. Branches, loops, widely separated boundaries, and heavy occlusion can make a single trend misleading. The labels and thresholds have not been calibrated against annotated ground truth.
