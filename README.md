# Boundary shape recognition

Run `python replay_boundary.py` to load the log and replay each frame simultaneously in global XY (left) and camera XZ (right) coordinates. The green dots are measured boundary points, the blue curves are independently estimated boundary trends in each coordinate plane, and the red marker and line show the robot position and heading in the global view. Both views use the same frame controls: Play/Pause, Previous/Next, the slider, or the left/right arrow keys. `--speed 2` plays at twice the logged rate.

For every frame, `boundary_shape.recognize(points_xyz)` estimates a centerline from global boundary points:

1. Use principal component analysis to find the long axis of the cloud.
2. Divide that axis into bins and take the median transverse coordinate in each populated bin.
3. Reject transverse outliers and fit a cubic trend through the bin medians.
4. Calculate tangent directions, length, net turning angle, and a rough shape label: straight, left bend, right bend, or S bend.

`orient_shape(shape, robot_xy)` orders the centerline from the endpoint closest to the robot. Consequently the reported direction and left/right labels are relative to that chosen path direction, not necessarily the robot's actual direction of travel. Direction angles are counterclockwise from global +X, in degrees. Frames with too few points or too little extent return `None` and display “insufficient shape data”.

The analysis is a geometric approximation for a single, mostly continuous boundary. Branches, loops, widely separated boundaries, and heavy occlusion can make a single cubic trend misleading. The labels and thresholds have not been calibrated against annotated ground truth.
