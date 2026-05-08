import numpy as np
import cv2, time
from xarm.wrapper import XArmAPI
from utils import presets
from utils.zed_camera import ZedCamera
from utils.vis_utils import draw_pose_axes
from scipy.spatial.transform import Rotation
from detector import BracketDetector

SPEED = 80
MIN_Z = 0.005


def wrap_to_nearest(angle, reference):
    diff = angle - reference
    return reference + (diff + 180) % 360 - 180



Y_CLIP_TAGS = {8, 7}
ROUND_CLIP_TAGS = {6, 9}


class RobotStateMachine:
    IDLE = "IDLE"
    DETECT = "DETECT"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    DONE = "DONE"


class ActionPlanner:
    def __init__(self, camera_pose: np.array):

        self.arm = XArmAPI(presets.ROBOT_IP)
        self.arm.connect()
        self.arm.motion_enable(enable=True)
        self.arm.set_tcp_offset([0, 0, presets.GRIPPER_LENGTH, 0, 0, 0])
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.move_gohome(speed=100, wait=True)

        self.t_robot_cam = np.linalg.inv(camera_pose)

        self.config = {
            "y_clip": {"X_OFFSET": -45.0, "Z_SAFE": 55.0},
            "round_clip": {"RADIUS": 45.0},
        }

        self.prev_clip_pose = None

        print("robot initialized")

    def shutdown(self):
        self.arm.disconnect()

    def pose_to_command(self, pose):
        rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
        x, y, z, _ = pose[:, 3].flatten()
        z = max(z, MIN_Z)

        return {
            "x": x * 1000,
            "y": y * 1000,
            "z": z * 1000,
            "roll": rpy[0],
            "pitch": rpy[1],
            "yaw": rpy[2],
            "speed": SPEED,
            "wait": True,
        }


    def y_clip_plan(self, clip_pose, next_clip_pose=None):
        cfg = self.config["y_clip"]
        X_OFFSET = cfg["X_OFFSET"]
        Z_SAFE = cfg["Z_SAFE"]

        flip = False
        y = 40

        cur = (self.t_robot_cam @ clip_pose)
        cur_xy = np.array([cur[0, 3], cur[1, 3]])

        if next_clip_pose is not None and self.prev_clip_pose is not None:
            nxt = (self.t_robot_cam @ next_clip_pose)
            nxt_xy = np.array([nxt[0, 3], nxt[1, 3]])

            approach_vec = nxt_xy - cur_xy
            clip_forward = cur[:2, 0]

            flip = np.dot(approach_vec, clip_forward) < 0

        elif next_clip_pose is not None and self.prev_clip_pose is None:
            _, robot = self.arm.get_position(is_radian=False)
            robot_xy = np.array([robot[0], robot[1]]) / 1000.0

            approach_vec = cur_xy - robot_xy
            clip_forward = cur[:2, 0]

            flip = np.dot(approach_vec, clip_forward) < 0

        flip_y = False

        if self.prev_clip_pose is None:
            _, robot = self.arm.get_position(is_radian=False)
            robot_y = robot[1] / 1000.0
            y = 40
            flip_y = robot_y < 0
        else:
            prev = (self.t_robot_cam @ self.prev_clip_pose)
            cur = (self.t_robot_cam @ clip_pose)

            prev_xy = np.array([prev[0, 3], prev[1, 3]])
            cur_xy = np.array([cur[0, 3], cur[1, 3]])

            motion_vec = cur_xy - prev_xy

            norm = np.linalg.norm(motion_vec)
            if norm > 1e-6:
                motion_vec = motion_vec / norm

            if not hasattr(self, "motion_history"):
                self.motion_history = []

            self.motion_history.append(motion_vec)
            if len(self.motion_history) > 2:
                self.motion_history.pop(0)

            avg_motion = np.mean(self.motion_history, axis=0)
            avg_norm = np.linalg.norm(avg_motion)

            if avg_norm > 1e-6:
                avg_motion = avg_motion / avg_norm

            DEADZONE = 0.02 
            flip_y = avg_motion[1] > -DEADZONE

        if next_clip_pose is None and self.prev_clip_pose is not None:
            prev = (self.t_robot_cam @ self.prev_clip_pose)
            prev_xy = np.array([prev[0, 3], prev[1, 3]])

            approach_vec = cur_xy - prev_xy 
            clip_forward = cur[:2, 0]

            flip = np.dot(approach_vec, clip_forward) < 0
            if flip:
                flip_y = not flip_y

        if flip:
            X_OFFSET = -X_OFFSET - 3
            y = -y/5

        if flip_y:
            y = y - 5
            if self.prev_clip_pose is not None:
                _, robot = self.arm.get_position(is_radian=False)
                robot_xy = np.array([robot[0], robot[1]]) / 1000.0
                prev_dist = np.linalg.norm(prev_xy - robot_xy)
                cur_dist = np.linalg.norm(cur_xy - robot_xy)
                if cur_dist < prev_dist:
                    y = -y

        def step(x, y_, z, log):
            T = np.eye(4)
            T[0, 3] = x / 1000
            T[1, 3] = y_ / 1000

            pose = self.t_robot_cam @ (clip_pose @ T)

            R = pose[:3, :3]
            mx, mz = R[:, 0], R[:, 2]

            tool_x = -mx if flip else mx
            tool_y = np.cross(mz, tool_x)
            tool_y /= np.linalg.norm(tool_y)

            pose[:3, :3] = np.column_stack((tool_x, tool_y, np.cross(tool_x, tool_y)))

            cmd = self.pose_to_command(pose)
            cmd["z"] += z
            cmd["log"] = log
            return cmd

        return [
            step(X_OFFSET, 45, Z_SAFE + 30, "[0] pre-approach"),
            step(X_OFFSET, 45, Z_SAFE, "[1] approach"),
            step(2 * X_OFFSET, 3 * y, Z_SAFE, "[2] tension"),
            step(X_OFFSET, 45, Z_SAFE, "[3] re approach"),
            step(-X_OFFSET, 45, Z_SAFE, "[4] align"),
            step(-X_OFFSET, 45, 12, "[5] insert"),
            step(-2 * X_OFFSET, 45, 12, "[6] finish"),
            step(-2 * X_OFFSET, 45, 30, "[7] lift"),
        ]


    def round_clip_plan(self, clip_pose, next_clip_pose=None, next_tag=None):
        cfg = self.config["round_clip"]
        RADIUS = cfg["RADIUS"]

        T = self.t_robot_cam @ clip_pose
        cx, cy = T[0, 3], T[1, 3]

        r = RADIUS / 1000.0
        LOW_Z = 0.035

        _, robot = self.arm.get_position(is_radian=False)
        rx, ry = robot[0] / 1000, robot[1] / 1000
        base_rpy = robot[3:6]

        entry_theta = np.arctan2(ry - cy, rx - cx)
        center = np.array([cx, cy])

        def segment_intersects_circle(p1, p2, center, radius):
            d = p2 - p1
            f = p1 - center

            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c = np.dot(f, f) - radius**2

            disc = b * b - 4 * a * c
            if disc < 0:
                return False

            disc = np.sqrt(disc)
            t1 = (-b - disc) / (2 * a)
            t2 = (-b + disc) / (2 * a)

            return (0 <= t1 <= 1) or (0 <= t2 <= 1)

        plan = []

        theta = entry_theta
        dtheta = 0.05
        max_steps = 500

        best_exit = None
        best_score = 1e9

        locked_yaw = base_rpy[2]

        next_is_y_clip = next_tag in Y_CLIP_TAGS if next_tag is not None else False

        if next_clip_pose is not None:
            nxt = self.t_robot_cam @ next_clip_pose
            goal = np.array([nxt[0, 3], nxt[1, 3]])


            robot_xy = np.array([rx, ry])

            v1 = np.array([cx, cy]) - robot_xy
            v2 = goal - np.array([cx, cy])

            cross = v1[0] * v2[1] - v1[1] * v2[0]

            direction = 1 if cross > 0 else -1

        else:
            goal = None
            direction = 1

        start_theta = theta
        total_angle = 0.0

        for i in range(max_steps):

            px = cx + r * np.cos(theta)
            py = cy + r * np.sin(theta)
            p = np.array([px, py])

            plan.append({
                "x": px * 1000,
                "y": py * 1000,
                "z": LOW_Z * 1000,
                "roll": base_rpy[0],
                "pitch": base_rpy[1],
                "yaw": locked_yaw,
                "speed": SPEED,
                "wait": True,
            })

            if goal is None:
                total_angle += abs(dtheta)

                if total_angle >= 1.5 * np.pi:
                    radial = np.array([px - cx, py - cy])
                    radial /= np.linalg.norm(radial)

                    ex = px + radial[0] * 0.005
                    ey = py + radial[1] * 0.005

                    plan.append({
                        "x": ex * 1000,
                        "y": ey * 1000,
                        "z": LOW_Z * 1000,
                        "roll": base_rpy[0],
                        "pitch": base_rpy[1],
                        "yaw": locked_yaw,
                        "speed": SPEED,
                        "wait": True,
                    })
                    return plan

                theta += direction * dtheta
                continue

            if total_angle >= np.pi:
                if not segment_intersects_circle(p, goal, center, r * 0.85):

                    if next_is_y_clip:

                        radial = p - center
                        radial = radial / np.linalg.norm(radial)
                        ex = cx + radial[0] * (r + 0.015)
                        ey = cy + radial[1] * (r + 0.015)
                    else:
                        ex, ey = goal

                    plan.append({
                        "x": ex * 1000,
                        "y": ey * 1000,
                        "z": LOW_Z * 1000,
                        "roll": base_rpy[0],
                        "pitch": base_rpy[1],
                        "yaw": locked_yaw,
                        "speed": SPEED,
                        "wait": True,
                    })
                    return plan

            score = np.linalg.norm(goal - p)
            if score < best_score:
                best_score = score
                best_exit = p

            theta += direction * dtheta
            total_angle += abs(dtheta)

        px, py = best_exit if best_exit is not None else (cx, cy)

        radial = np.array([px - cx, py - cy])
        radial /= np.linalg.norm(radial)

        ex = px + (radial[0] * 0.015) - 100
        ey = py + (radial[1] * 0.015) - 100

        plan.append({
            "x": ex * 1000,
            "y": ey * 1000,
            "z": LOW_Z * 1000,
            "roll": base_rpy[0],
            "pitch": base_rpy[1],
            "yaw": locked_yaw,
            "speed": SPEED,
            "wait": True,
        })

        return plan

    def execute_plan(self, plan):
        for s in plan:
            if "pause" in s:
                time.sleep(s["pause"])
                continue
            if "gripper" in s:
                continue

            print(s)
            if self.arm.set_position(**s) != 0:
                raise RuntimeError("motion failed")


def main():

    zed = ZedCamera()
    det = BracketDetector(zed.image, zed.camera_intrinsic)
    planner = ActionPlanner(det.camera_pose)

    results = det.identify_april_tag_ids()
    tag_map = {tid: pose for tid, pose in results}

    SEQUENCE = [int(x) for x in input("Enter sequence: ").replace(",", " ").split()]

    state = RobotStateMachine.IDLE
    i = 0
    tag = None
    pose = None
    next_pose = None

    while state != RobotStateMachine.DONE:

        if state == RobotStateMachine.IDLE:
            state = RobotStateMachine.DETECT

        elif state == RobotStateMachine.DETECT:

            if i >= len(SEQUENCE):
                state = RobotStateMachine.DONE
                continue

            tag = SEQUENCE[i]

            if tag not in tag_map:
                print(f"[WARN] missing tag {tag}")
                i += 1
                continue

            pose = tag_map[tag]
            next_pose = tag_map.get(SEQUENCE[i + 1]) if i + 1 < len(SEQUENCE) else None

            vis = det.color_image.copy()
            draw_pose_axes(vis, zed.camera_intrinsic, pose)

            cv2.imshow(f"tag {tag}", vis)
            key = cv2.waitKey(0)

            if key != ord('k'):
                continue

            state = RobotStateMachine.PLAN

        elif state == RobotStateMachine.PLAN:

            next_tag = SEQUENCE[i + 1] if i + 1 < len(SEQUENCE) else None

            if tag in Y_CLIP_TAGS:
                plan = planner.y_clip_plan(pose, next_pose)
            else:
                plan = planner.round_clip_plan(pose, next_pose, next_tag)

            state = RobotStateMachine.EXECUTE

        elif state == RobotStateMachine.EXECUTE:

            planner.execute_plan(plan)
            planner.prev_clip_pose = pose

            i += 1
            state = RobotStateMachine.DETECT

    print("[FINAL] Open gripper")
    planner.arm.open_lite6_gripper(sync=True)

    print("[FINAL] Go home")
    planner.arm.move_gohome(speed=100, wait=True)

    print("[FINAL] Close gripper")
    planner.arm.close_lite6_gripper(sync=True)

    print("[FINAL] Stop robot")
    planner.arm.set_state(4)
    planner.arm.stop_lite6_gripper(sync=True)

    planner.shutdown()
    zed.close()


if __name__ == "__main__":
    main()