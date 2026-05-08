CSCI - 5551 Introduction to Intelligent Robotics Systems

Group 8
Aayush Anand (anand239@umn.edu)
Adil Arya (arya0033@umn.edu)
Kshitij Nazirkar (nazir023@umn.edu)
Mohammed Jassim Jahubar Ali (jahub001@umn.edu)
Prashant Rao (rao00233@umn.edu)

Cable Clip Robot Automation
Robotic arm automation scripts for autonomously routing and securing cables using clip hardware detected via AprilTag fiducial markers and a ZED stereo camera. Built for the xArm Lite6 robot platform.

Overview
This repo contains three scripts that plan and execute physical cable management tasks - guiding a robot gripper to engage round and Y-shaped cable clips mounted on a surface. Clip positions are detected in real time using a ZED camera and AprilTag pose estimation.

Files
**evaluate_round.py**
Evaluates and executes a single round clip engagement maneuver.

Detects AprilTag ID 6 to locate the round clip in 3D space
Transforms the tag pose from camera space into robot space
Plans a full 360° circular arc trajectory around the clip at a fixed low height, mimicking the motion of threading a cable into a round retaining clip
Computes an entry angle based on the robot's current position so the circle motion begins naturally
Adds a tangential exit step at the end of the arc
Displays a visual preview of the detected pose before executing; press k to confirm

Key parameters: RADIUS = 45 mm, LOW_Z = 35 mm

**evaluate_y.py**
Evaluates and executes a single Y-clip engagement maneuver.

Detects AprilTag ID 8 to locate the Y-shaped clip
Dynamically adjusts the Y offset (42 mm or 47 mm) based on which direction the tag marker is facing relative to the robot
Plans a 7-step motion sequence: approach → align → insert → finish → open gripper → lift
Supports a flip mode (mirrored approach) and automatically scores both the forward and flipped plans by distance + rotation penalty, choosing the better one via choose_y_clip_plan()
Displays a visual preview before executing; press k to confirm

Key parameters: X_OFFSET = -42 mm, Z_SAFE = 60 mm

**modular_cable_mind.py**
The full autonomous sequencer - handles any ordered combination of round and Y-clips in a single run.

Accepts a user-defined sequence of AprilTag IDs at runtime (e.g. 6, 8, 7, 9)
Tags {7, 8} are treated as Y-clips; tags {6, 9} as round clips — dispatched automatically
Implements a RobotStateMachine with states: IDLE → DETECT → PLAN → EXECUTE → DONE
Uses look-ahead (next clip pose) and look-behind (previous clip pose) to dynamically determine:

Which direction to approach from (flip logic based on motion vector dot product)
Whether to flip the Y-axis offset (flip_y based on running motion history)


Round clip planning intelligently determines rotation direction (CW vs CCW) based on the direction toward the next clip, and finds a clean exit point that avoids re-entering the circle
Maintains a motion_history buffer to smooth approach direction decisions across sequential clips


Hardware & Dependencies
ComponentDetailsRobotxArm Lite6CameraZED Stereo CameraFiducialsAprilTags (IDs 6, 7, 8, 9)Python libsnumpy, opencv-python, scipy, xarm-python-sdkInternal modulesutils.presets, utils.zed_camera, utils.vis_utils, detector.BracketDetector

Usage
Single round clip test:
bashpython evaluate_round.py
Single Y-clip test:
bashpython evaluate_y.py
Full autonomous sequence:
bashpython modular_cable_mind.py
# Enter sequence when prompted, e.g.: 6 8 7 9

In the single-clip scripts, a camera preview window will open. Press k to confirm and execute, or any other key to skip.


Coordinate Conventions

All poses are 4×4 homogeneous transformation matrices
Camera-to-robot transform: T_robot_cam = inv(camera_pose)
Positions are converted between meters (internal) and millimeters (robot API)
MIN_Z = 5 mm floor enforced on all commanded positions to prevent ground collision

