import .lab2

import numpy as np
import urx
import matplotlib.pyplot as plt
import cv2
import cv2.aruco as aruco

from enum import IntEnum


def make_T(pos, orient):
    r = orient[0]
    p = orient[1]
    y = orient[2]

    # Rotation matrices
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos

    return T


class PadTag(IntEnum):
    START_PAD = 0
    MIDDLE_PAD = 1
    END_PAD = 2


class BoxTag(IntEnum):
    SMALL_BOX = 3
    MEDIUM_BOX = 4
    LARGE_BOX = 5


box_widths = {
    BoxTag.SMALL_BOX: 0.06,
    BoxTag.MEDIUM_BOX: 0.1,
    BoxTag.LARGE_BOX: 0.12,
}

# T5 to camera transform
T5c = make_T([0, -0.1016, 0.0848], [0, 0, 0])


def move_to_pose(
    T_target,
    *,
    T6t=None,
    thetaSeed0=None,
    fk_func=None,
    safety_check_func=None,
    external_safety_filter=None,
    joint_limits_rad=None,
    joint_limit_margin_rad=0.0,
    simulate=True,
    robot=None,
    movej_fn=None,
    movej_kwargs=None,
    translate_fn=None # Required if mode="linear"
):
    """
    Moves the robot to a specific 4x4 target matrix.
    Reuses the logic from draw_string/draw_character.
    """
    if fk_func is None:
        raise ValueError("move_to_pose: pass fk_func=fk")
    if movej_kwargs is None:
        movej_kwargs = {}
    if T6t is None:
        T6t = np.eye(4)
    
    # Initialize seed
    thetaSeed = np.zeros(6) if thetaSeed0 is None else np.asarray(thetaSeed0)

    # 1. Compute Inverse Kinematics
    theta_mod, _, info = lab2.ik(
        T_target,
        T6t=T6t,
        thetaSeed=thetaSeed,
        fk_func=fk_func,
        safety_check_func=safety_check_func,
        joint_limits_rad=joint_limits_rad,
        joint_limit_margin_rad=joint_limit_margin_rad,
    )

    if not info.get("success", False):
        print(f"Movement failed: {info.get('message')}")
        return False, thetaSeed

    # 2. Convert to Classical for robot/safety filter
    q_class = lab2.dh_modified_to_classical(theta_mod)

    # 3. Safety Check
    if external_safety_filter is not None:
        if not external_safety_filter(q_class):
            print("Movement rejected by external safety filter.")
            return False, thetaSeed

    # 4. Execution
    if not simulate:
        if robot is None:
            raise ValueError("Robot object required for non-simulated move.")
        
        if movej_fn is not None:
            movej_fn(q_class, **movej_kwargs)
        else:
            robot.movej(tuple(q_class.tolist()), **movej_kwargs)
        
    return True, theta_mod

class ArucoDetector:
    def __init__(self):
        # Intrinsics provided in Lab 3 manual [cite: 191, 192]
        self.K = np.array([[1698.75, 0, 1115.55],
                           [0, 1695.98, 751.98],
                           [0, 0, 1]])
        self.d = np.array([-0.00670872, -0.1481124, -0.00250596, 0.00299921, -1.68711031])

        self.tag_width = 0.02  # 2cm [cite: 190]
        self.block_thickness = 0.05  # 5cm
        self.pad_size = 0.2  # 20cm

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL) # [cite: 178]
        self.parameters = aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.tag_points = np.array([
            [-self.tag_width / 2, +self.tag_width / 2, 0],
            [+self.tag_width / 2, +self.tag_width / 2, 0],
            [+self.tag_width / 2, -self.tag_width / 2, 0],
            [-self.tag_width / 2, -self.tag_width / 2, 0],
        ])

    def find_tags(self, frame=None, draw=False, cam=None):
        """Detects tags and returns list of (ID, T_cam_marker) [cite: 184, 186]"""
        if frame is None:
            if cam is None:
                raise ValueError("Either frame or cam must be provided.")
            ret, frame = cam.read()
            if not ret:
                raise ValueError("Failed to capture image from camera.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)

        if len(ids) == 0:
            print("No tags detected")
            return []

        results = []

        # Solve PnP for each detected marker
        for i in range(len(ids)):
            tag_id = ids[i][0].item()

            _, rvec, tvec = cv2.solvePnP(self.tag_points, corners[i],
                                         self.K, self.d)

            # Convert rvec, tvec to 4x4 matrix
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec[:, 0]
            results.append((tag_id, T))

            print(f"\n--- Tag ID: {tag_id} ---")
            print(f"Translation (m): {tvec[:, 0]}")
            print("Rotation Matrix:")
            print(T[:3, :3])

            if draw:
                # Optional: Draw detection on the image for visual verification
                aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, self.K, self.d, rvec, tvec, 0.01)

        cv2.imshow("Detection Test", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(results)

        if cam is not None:
            cam.release()

        return results

    def make_grasp_dict(self, waypoints: dict[BoxTag, np.ndarray]):
        grasp_dict = {}
        for tag, T_base_waypoint in waypoints.items():
            T_base_grasp = self.get_grasp_pose(T_base_waypoint, box_widths[tag])
            grasp_dict[tag] = T_base_grasp
        return grasp_dict

    def make_place_dict(self, waypoints: dict[PadTag, np.ndarray]):
        place_dict = {}
        for tag, T_base_waypoint in waypoints.items():
            T_base_place = self.get_place_pose(T_base_waypoint, pad_size)
            place_dict[tag] = T_base_place
        return place_dict

    def test_image(self, image_path):
        """Loads a PNG and prints detected tag info"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image at {image_path}")
            return

        results = self.find_tags(frame=frame, draw=True)
        return results


    def get_grasp_pose(self, T_base_aruco, block_width):
        """
        Calculates T_bg (base to gripper center) [cite: 205, 209]
        T_base_aruco: transform of ArUco tag in base frame
        block_width: width of the rectangular prism (d)
        """
        T_aruco_block = np.eye(4)
        T_aruco_block[0, 3] = block_width / 2
        T_aruco_block[1, 3] = self.tag_width / 2 
        T_aruco_block[2, 3] = self.block_thickness / 2

        T_base_block = T_base_aruco @ T_aruco_block

        # Orientation for gripper: Approach from above (-Z base) 
        # and align jaws to grab along block's x-axis
        # T_block_grasp = np.eye(4)
        # T_block_grasp[:3, :3] = np.array([[1, 0, 0],
        #                                 [0, -1, 0],
        #                                 [0, 0, -1]]) 
        # T_block_grasp[2, 3] = 0.2 # gripper center offset
        #
        # T_base_grasp = T_base_block @ T_block_grasp
        #return T_base_grasp
        return T_base_block

    def get_place_pose(self, T_base_aruco):
        # Offset from the ArUco tag to the center of the pad
        offset = self.pad_size / 2
        T_base_place = T_base_aruco @ make_T([offset, offset, 0], [0, 0, 0])
        return T_base_place


def perform_scan(
    robot: urx.Robot,
    camera: cv2.VideoCapture,
    detector: ArucoDetector,
    scan_height: float = 0.4,
) -> dict[HanoiTag, np.ndarray]:
    # Box bounds in base frame
    # box_bounds = np.array([[-0.7, -1.0, -0.03], [0.7, 0.0, 0.03]])

    # Points in base frame for camera to scan at
    scan_points = np.array([
        [-0.7/2, -1/3, scan_height],
        [0, -1/3, scan_height],
        [0.7/2, -1/3, scan_height],
        [-0.7/2, 2/3, scan_height],
        [0, 2/3, scan_height],
        [0.7/2, 2/3, scan_height],
    ])

    # Mapping from tag ID to T_base_aruco
    box_detections = {}
    pad_detections = {}

    # Solve IK for the camera at each scan point
    for scan_point in scan_points:
        # Create transform from base to camera at scan point
        Tbc = make_T(scan_point, [np.pi, 0, 0])
        # Move the camera to the scan point
        move_to_pose(
            Tbc,
            T6t=T5c,
            robot=robot,
            simulate=False,
            movej_fn=lab2.movej_fn,
            movej_kwargs={"wait": True},
            fk_func=lab2.fk,
            safety_check_func=lab2.safety_check
        )
        # Find the ArUco tags in the camera image
        matches = detector.find_tags(cam=camera)

        for tag_id, T_base_aruco in matches:
            # Check if it is a valid Hanoi tag
            if tag_id in BoxTag:
                box_detections[BoxTag(tag_id)] = T_base_aruco
            elif tag_id in PadTag:
                pad_detections[PadTag(tag_id)] = T_base_aruco
            else:
                print(f"Detected non-Hanoi tag ID: {tag_id}")

    # Check that each tag was detected
    for tag in BoxTag | PadTag:
        if tag not in box_detections and tag not in pad_detections:
            raise ValueError(f"Tag {tag} not detected")

    return box_detections, pad_detections


def move_to(
    robot: urx.Robot,
    waypoints: dict[BoxTag | PadTag, np.ndarray],
    tag: BoxTag | PadTag,
    z_offset: float = 0.0,
):
    move_to_pose(
        waypoints[tag] @ make_T([0, 0, z_offset], [0, 0, 0]),
        T6t=T6t,
        fk_func=lab2.fk,
        safety_check_func=lab2.safety_check,
        external_safety_filter=None,
        simulate=False,
        robot=robot,
        movej_fn=lab2.movej_fn,
        movej_kwargs={"wait": True},
    )


def pick_up(robot: urx.Robot, distance: float = 0.05):
    robot.gripper.close()
    lab2.translate_fn(robot, (0, 0, distance), acc=al, vel=vl, wait=True)


def place_down(robot: urx.Robot, distance: float = 0.05):
    robot.gripper.open()
    lab2.translate_fn(robot, (0, 0, distance), acc=al, vel=vl, wait=True)


def home(robot: urx.Robot):
    lab2.set_initial_pose(
        Tbt_init,
        T6t=T6t,
        thetaSeed=None,
        fk_func=lab2.fk,
        safety_check_func=lab2.safety_check,
        external_safety_filter=None,
        simulate=False,
        dry_run=False,                  # switch to False for REAL RUN on robot
        robot=robot,
        movej_fn=lab2.movej_fn,
        verify_fk=True,
        on_fail="raise",
    )

def run():
    # Connect to robot
    UR_IP = "192.168.0.2"
    robot = urx.Robot(UR_IP)

    # specify max acc and vel
    aj, vj = 1.0, 0.5   # rad/s^2, rad/s (from your Lab 1 file)
    al, vl = 0.1, 0.03  # m/s^2, m/s

    # IMPORTANT: decide tool-offset convention:
    robot.set_tcp((0, 0, 0, 0, 0, 0))  # may need to set this
    # robot = 1

    # set initial pose to be able to see all tags from above
    # should be relative to base of robot
    # specify pure translation of tool frame (IN FRAME 6 COORDINATES)
    T6_trans = [0, 0, 0]  # meters
    T6t = make_T(T6_trans, [0, 0, 0])  # no rotation

    # # or specify full transformation matrix (IN FRAME 6 COORDINATES)
    # T6t = np.eye(4, dtype=float)
    # T6t[:3, 3] = [0, 0, 0] # can add rotation 

    # User-specified initial pose in base frame (IN WORLD/BASE COORDINATES)
    Tbt_init = np.eye(4)
    Tbt_init[:3, :3] = lab2._tool_orientation_not_axis_aligned()
    Tbt_init[:3, 3] = [0.40, -0.10, 0.10]  # ADJUST z0 HERE

    # Move to initial pose (IK + FK verification, but no robot)
    home(robot)

    # Setup the camera and detector
    camera = cv2.VideoCapture(0)
    detector = ArucoDetector()
    box_detections, pad_detections = perform_scan(robot, camera, detector)

    # The points to execute grasps at
    box_waypoints = detector.make_grasp_dict(box_detections)
    pad_waypoints = detector.make_place_dict(pad_detections)

    # Move the small box to the middle pad
    move_to(robot, box_waypoints, BoxTag.SMALL_BOX)
    pick_up(robot)
    home(robot)
    move_to(robot, pad_waypoints, PadTag.MIDDLE_PAD)
    place_down(robot)
    home(robot)

    # Move the medium box to the final pad
    move_to(robot, box_waypoints, BoxTag.MEDIUM_BOX)
    pick_up(robot)
    home(robot)
    move_to(robot, pad_waypoints, PadTag.END_PAD)
    place_down(robot)
    home(robot)
