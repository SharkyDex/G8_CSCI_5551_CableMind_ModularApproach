CSCI - 5551 Introduction to Intelligent Robotics Systems

Group 8
Aayush Anand (anand239@umn.edu) <br>
Adil Arya (arya0033@umn.edu) <br>
Kshitij Nazirkar (nazir023@umn.edu) <br>
Mohammed Jassim Jahubar Ali (jahub001@umn.edu) <br>
Prashant Rao (rao00233@umn.edu) <br>

Cable Clip Robot Automation <br>
Robotic arm automation scripts for autonomously routing and securing cables using clip hardware detected via AprilTag fiducial markers and a ZED stereo camera. Built for the xArm Lite6 robot platform.

Overview <br>
This repo contains three scripts that plan and execute physical cable management tasks - guiding a robot gripper to engage round and Y-shaped cable clips mounted on a surface. Clip positions are detected in real time using a ZED camera and AprilTag pose estimation.

Files <br>
**evaluate_round.py** <br>
Evaluates and executes a single round clip engagement maneuver.

Detects AprilTag ID 6 to locate the round clip in 3D space <br>
Transforms the tag pose from camera space into robot space <br>
Plans a full 360° circular arc trajectory around the clip at a fixed low height, mimicking the motion of threading a cable into a round retaining clip <br>
Computes an entry angle based on the robot's current position so the circle motion begins naturally <br>
Adds a tangential exit step at the end of the arc <br>
Displays a visual preview of the detected pose before executing; press k to confirm <br>

Key parameters: RADIUS = 45 mm, LOW_Z = 35 mm

**evaluate_y.py** <br>
Evaluates and executes a single Y-clip engagement maneuver. <br>

Detects AprilTag ID 8 to locate the Y-shaped clip <br>
Dynamically adjusts the Y offset (42 mm or 47 mm) based on which direction the tag marker is facing relative to the robot <br>
Plans a 7-step motion sequence: approach → align → insert → finish → open gripper → lift <br>
Supports a flip mode (mirrored approach) and automatically scores both the forward and flipped plans by distance + rotation penalty, choosing the better one via  choose_y_clip_plan() <br>
Displays a visual preview before executing; press k to confirm <br>

Key parameters: X_OFFSET = -42 mm, Z_SAFE = 60 mm

**modular_cable_mind.py** <br>
The full autonomous sequencer - handles any ordered combination of round and Y-clips in a single run.

Accepts a user-defined sequence of AprilTag IDs at runtime (e.g. 6, 8, 7, 9) <br>
Tags {7, 8} are treated as Y-clips; tags {6, 9} as round clips — dispatched automatically <br>
Implements a RobotStateMachine with states: IDLE → DETECT → PLAN → EXECUTE → DONE <br>
Uses look-ahead (next clip pose) and look-behind (previous clip pose) to dynamically determine: <br>

Which direction to approach from (flip logic based on motion vector dot product) <br>
Whether to flip the Y-axis offset (flip_y based on running motion history)


Round clip planning intelligently determines rotation direction (CW vs CCW) based on the direction toward the next clip, and finds a clean exit point that avoids re-entering the circle <br>
Maintains a motion_history buffer to smooth approach direction decisions across sequential clips


Hardware & Dependencies <br>
ComponentDetailsRobotxArm Lite6CameraZED Stereo CameraFiducialsAprilTags (IDs 6, 7, 8, 9)Python libsnumpy, opencv-python, scipy, xarm-python-sdkInternal <br><br>modulesutils.presets, utils.zed_camera, utils.vis_utils, detector.BracketDetector <br>

**Usage:** <br>
Single round clip test: <br>
bashpython evaluate_round.py <br>
Single Y-clip test: <br>
bashpython evaluate_y.py <br>
Full autonomous sequence: <br>
bashpython modular_cable_mind.py <br>
**Enter sequence when prompted, e.g.: 6 8 7 9 <br>**

In the single-clip scripts, a camera preview window will open. Press k to confirm and execute, or any other key to skip.


Coordinate Conventions

All poses are 4×4 homogeneous transformation matrices <br>
Camera-to-robot transform: T_robot_cam = inv(camera_pose) <br>
Positions are converted between meters (internal) and millimeters (robot API) <br>
MIN_Z = 5 mm floor enforced on all commanded positions to prevent ground collision <br>

