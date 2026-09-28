# Boundary shape recognition

Detailed fitting steps and direction conventions are documented in [拟合算法说明.md](拟合算法说明.md).

Run `python replay_boundary.py` to load the log and replay each frame simultaneously in global XY (left) and camera XZ (right) coordinates. The green dots are measured boundary points, the blue curves are independently estimated boundary trends, and the bright orange arrows and labels show the **tracking reference direction** near each purple curve's midpoint. The red marker and line show the robot position and heading in the global view. Both views use the same frame controls: Play/Pause, Previous/Next, the slider, or the left/right arrow keys. `--speed 2` plays at twice the logged rate.

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
