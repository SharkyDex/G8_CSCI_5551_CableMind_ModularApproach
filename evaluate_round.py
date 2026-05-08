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
            "round_clip": {"RADIUS": 45.0, "Z_SAFE": 60.0},
        }

        print("robot initialized")

    def shutdown(self):
        self.arm.open_lite6_gripper(sync=True)
        self.arm.set_pause_time(sltime=0.2, wait=True)

        self.arm.move_gohome(speed=100, wait=True)

        self.arm.close_lite6_gripper(sync=True)
        self.arm.stop_lite6_gripper(sync=True)

        time.sleep(1)
        self.arm.disconnect()

    def pose_to_command(self, pose: np.ndarray) -> dict:
        rpy = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
        x_m, y_m, z_m, _ = pose[:, 3].flatten()

        z_m = max(z_m, MIN_Z)
        
        return {
            "x": x_m * 1000,
            "y": y_m * 1000,
            "z": z_m * 1000,
            "roll": rpy[0],
            "pitch": rpy[1],
            "yaw": 0,
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

    def round_clip_plan(self, clip_pose: np.ndarray) -> List[dict]:

        cfg = self.config["round_clip"]
        RADIUS = cfg["RADIUS"]

        T = self.t_robot_cam @ clip_pose
        cx, cy = T[0, 3], T[1, 3]

        r = RADIUS / 1000.0
        LOW_Z = 0.035


        _, robot = self.arm.get_position(is_radian=False)
        rx, ry = robot[0] / 1000.0, robot[1] / 1000.0

        entry_theta = np.arctan2(ry - cy, rx - cx)

        angles = np.linspace(entry_theta, entry_theta + 2 * np.pi, 120)

        motion_plan = []

        rot = Rotation.from_matrix(T[:3, :3])
        quat = rot.as_quat()
        if quat[3] < 0:
            quat = -quat

        rot = Rotation.from_quat(quat)
        fixed_r, fixed_p, fixed_y = rot.as_euler("xyz", degrees=True)


        x0 = cx + r * np.cos(angles[0])
        y0 = cy + r * np.sin(angles[0])

        motion_plan.append({
            "x": x0 * 1000,
            "y": y0 * 1000,
            "z": LOW_Z * 1000,
            "roll": fixed_r,
            "pitch": fixed_p,
            "yaw": fixed_y,
            "speed": SPEED,
            "wait": True,
            "log": "[1] Start circle"
        })


        for th in angles:
            x = cx + r * np.cos(th)
            y = cy + r * np.sin(th)

            motion_plan.append({
                "x": x * 1000,
                "y": y * 1000,
                "z": LOW_Z * 1000,
                "roll": fixed_r,
                "pitch": fixed_p,
                "yaw": fixed_y,
                "speed": SPEED,
                "wait": True,
            })

        x1, y1 = cx + r * np.cos(angles[-2]), cy + r * np.sin(angles[-2])
        x2, y2 = cx + r * np.cos(angles[-1]), cy + r * np.sin(angles[-1])

        dx, dy = x2 - x1, y2 - y1
        norm = np.sqrt(dx**2 + dy**2)
        dx, dy = dx / norm, dy / norm

        ex = x2 + dx * 0.03
        ey = y2 + dy * 0.03

        motion_plan.append({
            "x": ex * 1000,
            "y": ey * 1000,
            "z": LOW_Z * 1000,
            "roll": fixed_r,
            "pitch": fixed_p,
            "yaw": fixed_y,
            "speed": SPEED,
            "wait": True,
            "log": "[END] Exit tangent"
        })

        return motion_plan


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

        TAG_ID = 6  
        if TAG_ID not in tag_dict:
            raise Exception("Round clip not found")

        pose = tag_dict[TAG_ID]

        vis = detector.color_image.copy()
        draw_pose_axes(vis, camera_intrinsic, pose)

        cv2.imshow("Round Clip", vis)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key != ord("k"):
            print("Skipped")
            return

        print("Executing FULL 360° round clip")

        motion_plan = planner.round_clip_plan(pose)

        planner.arm.close_lite6_gripper(sync=True)
        planner.execute_plan(motion_plan)

    finally:
        planner.shutdown()
        zed.close()


if __name__ == "__main__":
    start = time.time()
    main()
    print("Total time:", np.round(time.time() - start, 2), "s")