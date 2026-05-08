import numpy as np
import cv2, time
from xarm.wrapper import XArmAPI
from utils import presets
from utils.zed_camera import ZedCamera
from utils.vis_utils import draw_pose_axes
from scipy.spatial.transform import Rotation
from detector import BracketDetector
from typing import List

SPEED = 80
MIN_Z = 0.005  


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
            "y_clip": {"X_OFFSET": -42.0, "Z_SAFE": 60.0},
        }

        print("robot initialized")

    def shutdown(self):
        self.arm.open_lite6_gripper(sync=True)
        self.arm.set_pause_time(sltime=0.2, wait=True)

        _, safe = self.arm.get_position(is_radian=False)
        self.arm.set_position(
            x=safe[0], y=safe[1], z=safe[2] + 20,
            roll=safe[3], pitch=safe[4], yaw=safe[5],
            is_radian=False, speed=100, wait=True
        )

        self.arm.move_gohome(speed=100, wait=True)
        self.arm.close_lite6_gripper(sync=True)
        self.arm.stop_lite6_gripper(sync=True)

        time.sleep(1)
        self.arm.disconnect()

    def pose_to_command(self, pose: np.array) -> dict:
        rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
        x_m, y_m, z_m, _ = pose[:, 3].flatten()

        z_m = max(z_m, MIN_Z)

        return {
            "x": x_m * 1000,
            "y": y_m * 1000,
            "z": z_m * 1000,
            "roll": rpy[0],
            "pitch": rpy[1],
            "yaw": rpy[2],
            "is_radian": False,
            "speed": SPEED,
            "wait": True,
        }


    def execute_plan(self, plan: List[dict]) -> None:
        for step in plan:

            if step.get("gripper") == "open":
                print("[Gripper] Opening")
                self.arm.open_lite6_gripper(sync=True)
                continue

            if step.get("pause"):
                time.sleep(step["pause"])
                continue

            if "log" in step:
                print(step["log"])
                step.pop("log")

            print(step)
            ret = self.arm.set_position(**step)

            if ret != 0:
                raise RuntimeError("Motion failed")

        self.arm.set_pause_time(sltime=0.5, wait=True)
        print("Motion Completed")

    def y_clip_plan(self, clip_pose: np.ndarray, flip: bool = False) -> List[dict]:

        T = self.t_robot_cam @ clip_pose
        R0 = T[:3, :3]

        mx = R0[:, 0]
        tag_pos = T[:3, 3]

        to_tag = tag_pos / np.linalg.norm(tag_pos)
        alignment = np.dot(mx, to_tag)

        X_OFFSET = self.config["y_clip"]["X_OFFSET"]
        Z_SAFE = self.config["y_clip"]["Z_SAFE"]

        # ✅ dynamic Y offset (your rule)
        if alignment > 0:
            Y_OFFSET = 42.0
            print("[Y-clip] marker facing away → Y_OFFSET = 42")
        else:
            Y_OFFSET = 47.0
            print("[Y-clip] marker facing robot → Y_OFFSET = 47")

        if flip:
            X_OFFSET = -X_OFFSET

        def step(x, y, z, log):
            t = np.eye(4)
            t[0, 3] = x / 1000
            t[1, 3] = y / 1000

            pose = self.t_robot_cam @ clip_pose @ t

            R = pose[:3, :3]
            mx = R[:, 0]
            mz = R[:, 2]

            tool_x = mx if not flip else -mx

            tool_y = np.cross(mz, tool_x)
            tool_y /= np.linalg.norm(tool_y)

            tool_z = np.cross(tool_x, tool_y)

            pose[:3, :3] = np.column_stack((tool_x, tool_y, tool_z))

            cmd = self.pose_to_command(pose)
            cmd["z"] += z
            cmd["log"] = log
            return cmd

        return [
            step(X_OFFSET, Y_OFFSET, Z_SAFE, "[1/5] Approach"),
            step(-X_OFFSET, Y_OFFSET, Z_SAFE, "[2/5] Align"),
            step(-X_OFFSET, Y_OFFSET, 12.5, "[3/5] Insert"),
            step(-2 * X_OFFSET, Y_OFFSET, 12.5, "[4/5] Finish"),

            {"gripper": "open"},
            {"pause": 0.3},

            step(-2 * X_OFFSET, Y_OFFSET, 30, "[5/5] Lift"),
        ]

    def score_plan(self, plan: List[dict]) -> float:
        _, current = self.arm.get_position(is_radian=False)

        cx, cy, cz = current[0], current[1], current[2]
        first = plan[0]

        dx = first["x"] - cx
        dy = first["y"] - cy
        dz = first["z"] - cz

        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        rot_penalty = abs(first["roll"]) + abs(first["pitch"])

        return dist + 0.5 * rot_penalty

    def choose_y_clip_plan(self, clip_pose: np.ndarray) -> List[dict]:
        plan_f = self.y_clip_plan(clip_pose, flip=False)
        plan_b = self.y_clip_plan(clip_pose, flip=True)

        score_f = self.score_plan(plan_f)
        score_b = self.score_plan(plan_b)

        print(f"[Planner] Forward: {score_f:.2f}")
        print(f"[Planner] Backward: {score_b:.2f}")

        return plan_f if score_f <= score_b else plan_b


def main():
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    detector = BracketDetector(
        observation=zed.image,
        intrinsic=camera_intrinsic,
    )

    planner = ActionPlanner(camera_pose=detector.camera_pose)

    try:
        results = detector.identify_april_tag_ids()
        if not results:
            raise Exception("No objects found")

        tag_dict = {tid: pose for tid, pose in results}

        TAG_ID = 8
        if TAG_ID not in tag_dict:
            raise Exception("Y-clip tag not found")

        pose = tag_dict[TAG_ID]
        print("Executing Y-clip")

        vis = detector.color_image.copy()
        draw_pose_axes(vis, camera_intrinsic, pose)

        cv2.imshow("Y-clip", vis)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key != ord("k"):
            print("Skipped")
            return

        motion_plan = planner.choose_y_clip_plan(pose)

        planner.arm.close_lite6_gripper(sync=True)
        planner.execute_plan(motion_plan)

    finally:
        planner.shutdown()
        zed.close()


if __name__ == "__main__":
    start = time.time()
    main()
    print("Total time:", np.round(time.time() - start, 2), "s")