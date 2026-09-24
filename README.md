# Boundary shape recognition

Detailed fitting steps and direction conventions are documented in [拟合算法说明.md](拟合算法说明.md).

Run `python replay_boundary.py` to load the log and replay each frame simultaneously in global XY (left) and camera XZ (right) coordinates. The green dots are measured boundary points, the blue curves are independently estimated boundary trends, and the bright orange arrows and labels show the estimated boundary direction near each curve's midpoint. The red marker and line show the robot position and heading in the global view. Both views use the same frame controls: Play/Pause, Previous/Next, the slider, or the left/right arrow keys. `--speed 2` plays at twice the logged rate.

The dashed purple line is one tracking reference shown in both views. It is computed in global XY by moving each fitted boundary point 0.4 m along its local normal toward the robot. The offset side is chosen from the robot's position relative to the whole curve's end-to-end direction and kept constant along the curve. Curves with bends too tight for a smooth 0.4 m inner offset fall back to a simpler trend. Consecutive frames with nearly unchanged robot position and yaw are fitted together to keep the reference stable while the robot is stationary. The global reference is then mapped into camera XZ using that frame's pose and camera extrinsics, assuming camera Y=0 as for the logged points. It is not a separate 0.4 m offset of the camera XZ fit.

For every frame, `boundary_shape.recognize(points_xyz)` estimates a centerline from global boundary points:

1. Use principal component analysis to find the long axis of the cloud.
2. Divide that axis into bins and take the median transverse coordinate in each populated bin.
3. Reject transverse outliers and fit a cubic, quadratic, or linear trend through the bin medians, subject to a curvature limit.
4. Calculate tangent directions, length, net turning angle, and a rough shape label: straight, left bend, right bend, or S bend.

`orient_shape(shape, robot_xy)` orders the centerline from the endpoint closest to the robot. Consequently the reported direction and left/right labels are relative to that chosen path direction, not necessarily the robot's actual direction of travel. Direction angles are counterclockwise from global +X, in degrees. Frames with too few points or too little extent return `None` and display “insufficient shape data”.

The analysis is a geometric approximation for a single, mostly continuous boundary. Branches, loops, widely separated boundaries, and heavy occlusion can make a single trend misleading. The labels and thresholds have not been calibrated against annotated ground truth.
