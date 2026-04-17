import numpy as np
import matplotlib.pyplot as plt
import lab2_NikosMynhierClareWu as lab2
import cv2
import cv2.aruco as aruco

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
    print(f'make t: {T}')
    return T

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
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL) # [cite: 178]
        self.parameters = aruco.DetectorParameters()

    def find_tags(self, frame=None, draw=False, cam=None):
        """Detects tags and returns list of (ID, T_cam_marker) [cite: 184, 186]"""
        if frame is None:
            if cam is None:
                raise ValueError("Either frame or cam must be provided.")
            ret, frame = cam.read()
            if not ret:
                raise ValueError("Failed to capture image from camera.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        results = []
        if ids is not None:
            # Solve PnP for each detected marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.tag_width, self.K, self.d)
            for i in range(len(ids)):
                # Convert rvec, tvec to 4x4 matrix
                R, _ = cv2.Rodrigues(rvecs[i][0])
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvecs[i][0]
                results.append((ids[i][0], T))

                print(f"\n--- Tag ID: {ids[i][0]} ---")
                print(f"Translation (m): {tvecs[i][0]}")
                print("Rotation Matrix:")
                print(T[:3, :3])
                
                if draw:
                    # Optional: Draw detection on the image for visual verification
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(frame, self.K, self.d, rvecs[i], tvecs[i], 0.01)
            cv2.imshow("Detection Test", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No tags detected.")
        print(results)

        if cam is not None:
            cam.release()
        return results
    
    def make_grasp_dict(self, mats, T_base_cam):
        grasp_dict = {}
        for mat in mats:
            print(f'tag id: {mat[0]}')
            T_base_aruco = T_base_cam @ mat[1]
            print(f'T_base_aruco:\n{T_base_aruco}')
            T_base_grasp = self.get_grasp_pose(T_base_aruco, 0.08) # Example block width of 4cm
            print("\nGrasp Pose (T_bg):")
            print(T_base_grasp)
            grasp_dict[mat[0]] = T_base_grasp
        return grasp_dict

    def test_image(self, image_path):
        """Loads a PNG and prints detected tag info"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image at {image_path}")
            return

        results = self.find_tags(frame=frame, draw=True)
        return results


    def get_grasp_pose(self, T_base_aruco, block_width, block_length=0.0):
        """
        Calculates T_bg (base to gripper center) [cite: 205, 209]
        T_base_aruco: transform of ArUco tag in base frame
        block_width: width of the rectangular prism (d)
        """
        print(type(block_width))
        T_aruco_block = np.eye(4)
        T_aruco_block[0, 3] = block_width / 2.0 if block_length == 0.0 else block_length / 2.0
        T_aruco_block[1, 3] = block_width / 2.0 
        T_aruco_block[2, 3] = 0.025 # half of 5cm thickness
        
        T_base_block = T_base_aruco @ T_aruco_block
        
        # Orientation for gripper: Approach from above (-Z base) 
        # and align jaws to grab along block's x-axis
        T_block_grasp = np.eye(4)
        T_block_grasp[:3, :3] = np.array([[1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, -1]]) 
        T_block_grasp[2, 3] = 0.2 # gripper center offset
        
        T_base_grasp = T_base_block @ T_block_grasp
        return T_base_grasp

tester = ArucoDetector()
mats = tester.test_image('aruco_detection_test_practice.png')
# cam mat made based off tester image
cam_pos = [0.0, -0.4, 0.4]
cam_orient = [np.pi, 0.0, np.pi]
T_base_cam = make_T(cam_pos, cam_orient)
tester.make_grasp_dict(mats, T_base_cam)


# ACTUAL ROBOT STUFF
# connect to UR10e
# UR_IP = "192.168.0.2"
# robot = urx.Robot(UR_IP)

# specify max acc and vel
aj, vj = 1.0, 0.5   # rad/s^2, rad/s (from your Lab 1 file)
al, vl = 0.1, 0.03  # m/s^2, m/s

# IMPORTANT: decide tool-offset convention:
# robot.set_tcp((0,0,0,0,0,0)) # may need to set this
robot = 1

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
Tbt_init[:3, 3]  = [0.40, -0.10, 0.10]  # ADJUST z0 HERE

# Move to initial pose (IK + FK verification, but no robot)
init_out = lab2.set_initial_pose(
    Tbt_init,
    T6t=T6t,
    thetaSeed=None,
    fk_func=lab2.fk,
    safety_check_func=lab2.safety_check,
    external_safety_filter=None,
    simulate=False,                
    dry_run=True,                  # switch to False for REAL RUN on robot
    robot=robot,
    movej_fn=lab2.movej_fn,
    verify_fk=True,
    on_fail="raise",
)

print("\n=== set_initial_pose dry_run ===")
print("success:", init_out["success"])
print("executed:", init_out.get("executed"))
print("q_class:", np.array2string(init_out["q_class"], precision=4))
print("verify:", init_out.get("verify", {}))

# setting up camera?
cam = cv2.VideoCapture(0)

detector = ArucoDetector()
mats = detector.find_tags(cam=cam, draw=True)
grasp_dict = detector.make_grasp_dict(mats, T_base_cam)

# move to grasp pose
move_to_pose(grasp_dict[0], 
             T6t=T6t, 
             fk_func=lab2.fk, 
             safety_check_func=lab2.safety_check, 
             external_safety_filter=None, 
             simulate=False, robot=robot, 
             movej_fn=lab2.movej_fn, 
             movej_kwargs={"wait": True})

robot.gripper.close()
lab2.translate_fn(robot, (0, 0, 0.1), acc=al, vel=vl, wait=True)

#return to initial pose to look for next aruco tag?
init_out = lab2.set_initial_pose(
    Tbt_init,
    T6t=T6t,
    thetaSeed=None,
    fk_func=lab2.fk,
    safety_check_func=lab2.safety_check,
    external_safety_filter=None,
    simulate=False,                
    dry_run=True,                  # switch to False for REAL RUN on robot
    robot=robot,
    movej_fn=lab2.movej_fn,
    verify_fk=True,
    on_fail="raise",
)
