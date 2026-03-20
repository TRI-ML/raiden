"""Orchestrates the full camera calibration workflow"""

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Import kinematics from i2rt
from i2rt.robots.kinematics import Kinematics

from raiden._config import CALIBRATION_FILE, CALIBRATION_POSES_FILE, CAMERA_CONFIG
from raiden._xml_paths import get_yam_4310_linear_xml_path

from raiden.calibration.core import (
    CameraCalibrator,
    ChArUcoBoardConfig,
    load_calibration_poses,
    save_calibration_results,
)
from raiden.camera_config import CameraConfig
from raiden.cameras.base import Camera
from raiden.robot.controller import RobotController, smooth_move_joints

# Global kinematics instance (cached for efficiency)
_kinematics_cache: Optional[Kinematics] = None


def compute_bimanual_base_transform_from_calibration(
    left_arm_poses: List[np.ndarray],
    right_arm_poses: List[np.ndarray],
    left_board_poses: List[Tuple[np.ndarray, np.ndarray]],
    right_board_poses: List[Tuple[np.ndarray, np.ndarray]],
    left_hand_eye_result: Dict,
    right_hand_eye_result: Dict,
) -> np.ndarray:
    """Compute transform from right arm base to left arm base after hand-eye calibrations

    Uses hand-eye calibration results for both arms:
    1. For each pose, compute board position from left arm: left_base -> left_ee -> left_cam -> board
    2. For each pose, compute board position from right arm: right_base -> right_ee -> right_cam -> board
    3. Since board is fixed, both should point to same location in world
    4. Compute offset between bases and average across poses

    Args:
        left_arm_poses: List of T_left_base_to_left_ee transforms
        right_arm_poses: List of T_right_base_to_right_ee transforms (in right base frame)
        left_board_poses: List of (rvec, tvec) board observations from left camera
        right_board_poses: List of (rvec, tvec) board observations from right camera
        left_hand_eye_result: Hand-eye calibration result for left wrist (T_left_cam_to_left_ee)
        right_hand_eye_result: Hand-eye calibration result for right wrist (T_right_cam_to_right_ee)

    Returns:
        T_right_base_to_left_base: 4x4 transformation matrix
    """
    if len(left_arm_poses) < 1 or len(right_arm_poses) < 1:
        raise ValueError("Need at least 1 pose from both arms")

    if len(left_board_poses) < 1 or len(right_board_poses) < 1:
        raise ValueError("Need at least 1 board observation from both cameras")

    from scipy.spatial.transform import Rotation

    num_poses = min(
        len(left_arm_poses),
        len(right_arm_poses),
        len(left_board_poses),
        len(right_board_poses),
    )
    print(
        f"    Using hand-eye calibrations to compute bimanual base transform ({num_poses} poses)..."
    )

    # Extract hand-eye transforms (camera to end-effector)
    T_left_cam_to_left_ee = np.eye(4)
    T_left_cam_to_left_ee[:3, :3] = np.array(left_hand_eye_result["rotation_matrix"])
    T_left_cam_to_left_ee[:3, 3] = np.array(left_hand_eye_result["translation_vector"])

    T_right_cam_to_right_ee = np.eye(4)
    T_right_cam_to_right_ee[:3, :3] = np.array(right_hand_eye_result["rotation_matrix"])
    T_right_cam_to_right_ee[:3, 3] = np.array(
        right_hand_eye_result["translation_vector"]
    )

    # Verify rotation matrices are valid
    det_left = np.linalg.det(T_left_cam_to_left_ee[:3, :3])
    det_right = np.linalg.det(T_right_cam_to_right_ee[:3, :3])

    if abs(det_left - 1.0) > 0.01 or abs(det_right - 1.0) > 0.01:
        print(
            f"    Warning: Rotation matrices are not orthogonal (det_left={det_left:.4f}, det_right={det_right:.4f})"
        )

    # For each pose, compute board position from each base frame
    transforms = []

    for i in range(num_poses):
        # Get robot FK poses
        T_left_base_to_left_ee = left_arm_poses[i]
        T_right_base_to_right_ee = right_arm_poses[i]

        # Get board observations (board-to-camera)
        rvec_l, tvec_l = left_board_poses[i]
        R_l, _ = cv2.Rodrigues(rvec_l)
        T_board_to_left_cam = np.eye(4)
        T_board_to_left_cam[:3, :3] = R_l
        T_board_to_left_cam[:3, 3] = tvec_l.flatten()

        rvec_r, tvec_r = right_board_poses[i]
        R_r, _ = cv2.Rodrigues(rvec_r)
        T_board_to_right_cam = np.eye(4)
        T_board_to_right_cam[:3, :3] = R_r
        T_board_to_right_cam[:3, 3] = tvec_r.flatten()

        # Correct chain: T^base_board = T^base_ee @ T^ee_cam @ T^cam_board
        # = FK @ hand_eye_result @ ChArUco_result
        # No inversions needed:
        #   - T_left_base_to_left_ee  = FK               = T^base_ee  (transforms ee->base)
        #   - T_left_cam_to_left_ee   = hand-eye output  = T^ee_cam   (transforms cam->ee)
        #   - T_board_to_left_cam     = ChArUco output   = T^cam_board (transforms board->cam)

        T_left_base_to_board = (
            T_left_base_to_left_ee @ T_left_cam_to_left_ee @ T_board_to_left_cam
        )

        T_right_base_to_board = (
            T_right_base_to_right_ee @ T_right_cam_to_right_ee @ T_board_to_right_cam
        )

        # Since board is fixed: T_left_base_to_board = T_left_base_to_right_base @ T_right_base_to_board
        # Therefore: T_left_base_to_right_base = T_left_base_to_board @ inv(T_right_base_to_board)
        T_left_base_to_right_base = T_left_base_to_board @ np.linalg.inv(
            T_right_base_to_board
        )

        # We want T_right_base_to_left_base = inv(T_left_base_to_right_base)
        T_right_base_to_left_base = np.linalg.inv(T_left_base_to_right_base)

        transforms.append(T_right_base_to_left_base)

    # Average the transforms across all poses
    rotations = [Rotation.from_matrix(T[:3, :3]) for T in transforms]
    translations = [T[:3, 3] for T in transforms]

    avg_rotation = Rotation.concatenate(rotations).mean()
    avg_translation = np.mean(translations, axis=0)

    T_right_base_to_left_base = np.eye(4)
    T_right_base_to_left_base[:3, :3] = avg_rotation.as_matrix()
    T_right_base_to_left_base[:3, 3] = avg_translation

    print(
        f"    Bimanual transform (T_right_base_to_left_base) translation: {T_right_base_to_left_base[:3, 3]}"
    )

    return T_right_base_to_left_base


# Bimanual base transform (computed during calibration)
T_RIGHT_BASE_TO_LEFT_BASE: Optional[np.ndarray] = None


def compute_forward_kinematics(
    joint_positions: np.ndarray, arm: str = "left"
) -> np.ndarray:
    """Compute forward kinematics for YAM robot arm using i2rt library

    Args:
        joint_positions: 6 joint positions (without gripper)
        arm: Which arm ("left" or "right")

    Returns:
        4x4 transformation matrix from left arm base to end-effector (grasp_site)
        All transforms are expressed relative to the left arm base frame.
    """
    global _kinematics_cache

    # Initialize kinematics on first call (cached for subsequent calls)
    if _kinematics_cache is None:
        _kinematics_cache = Kinematics(
            get_yam_4310_linear_xml_path(), site_name="grasp_site"
        )

    # Ensure we have 6 DoF (without gripper)
    assert len(joint_positions) == 6, (
        f"Expected 6 joint positions, got {len(joint_positions)}"
    )

    # Compute forward kinematics (returns T_arm_base_to_ee)
    T_arm_base_to_ee = _kinematics_cache.fk(joint_positions)

    # Transform to left arm base frame
    if arm == "right":
        if T_RIGHT_BASE_TO_LEFT_BASE is None:
            raise ValueError(
                "Bimanual transform not available. "
                "Calibrate both left and right wrist cameras together to compute it."
            )

        # For right arm: T_left_base_to_ee = T_left_base_to_right_base @ T_right_base_to_ee
        T_left_base_to_right_base = np.linalg.inv(T_RIGHT_BASE_TO_LEFT_BASE)
        return T_left_base_to_right_base @ T_arm_base_to_ee

    # For left arm, already in the correct frame
    return T_arm_base_to_ee


class CalibrationRunner:
    """Runs the complete calibration workflow"""

    def __init__(
        self,
        camera_config_file: str = CAMERA_CONFIG,
        poses_file: str = CALIBRATION_POSES_FILE,
        output_file: str = CALIBRATION_FILE,
        charuco_config: Optional[ChArUcoBoardConfig] = None,
    ):
        self.camera_config_file = camera_config_file
        self.poses_file = poses_file
        self.output_file = output_file
        self._charuco_config_override = charuco_config

        self.camera_config: Optional[CameraConfig] = None
        self.poses_data: Optional[Dict] = None
        self.board_config: Optional[ChArUcoBoardConfig] = None
        self.calibrator: Optional[CameraCalibrator] = None

        self.robot_controller: Optional[RobotController] = None
        self.cameras: Dict[str, Camera] = {}

    def load_configuration(self):
        """Load camera config and calibration poses"""
        print("\nLoading configuration...")

        # Load camera config
        self.camera_config = CameraConfig(self.camera_config_file)
        num_cameras = len(self.camera_config.list_cameras())
        print(f"✓ Camera config loaded: {num_cameras} camera(s) configured")

        # Load poses
        self.poses_data = load_calibration_poses(self.poses_file)
        num_poses = len(self.poses_data["poses"])
        print(f"✓ Calibration poses loaded: {num_poses} pose(s)")

        # Load ChArUco config (CLI override takes precedence over poses file)
        if self._charuco_config_override is not None:
            self.board_config = self._charuco_config_override
        else:
            self.board_config = ChArUcoBoardConfig.from_dict(
                self.poses_data["charuco_config"]
            )
        print("✓ ChArUco board config loaded")

        # Create calibrator
        self.calibrator = CameraCalibrator(self.board_config)

    def initialize_robots(self, left_wrist: bool, right_wrist: bool):
        """Initialize robots based on which cameras to calibrate"""
        # Create robot controller (only followers, no leaders needed for calibration)
        self.robot_controller = RobotController(
            use_right_leader=False,
            use_left_leader=False,
            use_right_follower=right_wrist,
            use_left_follower=left_wrist,
        )

        # Check CAN interfaces
        print("\nChecking CAN interfaces...")
        self.robot_controller.check_can_interfaces()

        # Initialize robots
        print("\nInitializing robots...")
        self.robot_controller.initialize_robots(gravity_comp_mode=False)

        # Move to home positions
        self.robot_controller.move_to_home_positions(simultaneous=True)

    def initialize_cameras(self, camera_names: List[str], warmup_frames: int = 10):
        """Initialize cameras by name using the camera factory."""
        print("\nInitializing cameras...")

        for name in camera_names:
            cam_type = self.camera_config.get_camera_type(name)
            serial = self.camera_config.get_serial_by_name(name)
            if serial is None:
                raise ValueError(f"Camera '{name}' not found in config")

            print(f"  - Initializing {name} ({cam_type}, serial: {serial})...")
            camera = self.camera_config.create_camera(name)
            camera.open()
            self.cameras[name] = camera

        # Warm up all cameras together after all are open, so each one has had
        # time to stabilize regardless of how many cameras are connected.
        print(f"  Warming up cameras ({warmup_frames} frames)...")
        for _ in range(warmup_frames):
            for camera in self.cameras.values():
                camera.grab()

        print("✓ Cameras initialized")

    def _compute_and_apply_bimanual_transform(
        self, camera_data: Dict, left_camera_id: str, right_camera_id: str
    ) -> bool:
        """Compute bimanual transform from hand-eye calibration results

        Strategy:
        1. Compute left hand-eye calibration (in left arm base frame)
        2. Compute right hand-eye calibration (in right arm base frame)
        3. Use both hand-eye results to compute bimanual base transform
        4. Update right arm poses to left base frame

        Args:
            camera_data: Dictionary with calibration data for all cameras
            left_camera_id: Name of left wrist camera
            right_camera_id: Name of right wrist camera

        Returns:
            True if transform was successfully computed, False otherwise
        """
        global T_RIGHT_BASE_TO_LEFT_BASE

        print("\nComputing bimanual base transform from hand-eye calibrations...")

        left_data = camera_data.get(left_camera_id)
        right_data = camera_data.get(right_camera_id)

        # Validate we have data from both cameras
        if not left_data or not right_data:
            print("  ✗ Missing camera data")
            return False

        if not left_data["corners"] or not right_data["corners"]:
            print("  ✗ No board detections found")
            return False

        # Step 1: Get camera intrinsics from SDK (factory calibrated)
        print("  Step 1: Getting camera intrinsics from SDK...")
        left_camera = self.cameras.get(left_camera_id)
        right_camera = self.cameras.get(right_camera_id)

        if not left_camera or not right_camera:
            print("  ✗ Camera objects not found")
            return False

        left_cam_matrix, left_dist_coeffs, left_img_size = left_camera.get_intrinsics()
        right_cam_matrix, right_dist_coeffs, right_img_size = (
            right_camera.get_intrinsics()
        )

        print(
            f"    ✓ Left camera: fx={left_cam_matrix[0, 0]:.1f}, fy={left_cam_matrix[1, 1]:.1f}"
        )
        print(
            f"    ✓ Right camera: fx={right_cam_matrix[0, 0]:.1f}, fy={right_cam_matrix[1, 1]:.1f}"
        )

        # Step 2: Estimate board poses from both cameras
        left_board_poses = self._estimate_board_poses(
            left_data,
            left_cam_matrix.tolist(),
            left_dist_coeffs.tolist(),
        )
        right_board_poses = self._estimate_board_poses(
            right_data,
            right_cam_matrix.tolist(),
            right_dist_coeffs.tolist(),
        )

        # Match valid poses
        valid_poses = self._match_valid_poses(
            left_data["robot_poses"],
            right_data["robot_poses"],
            left_board_poses,
            right_board_poses,
        )

        if not valid_poses:
            print("  ✗ No valid pose pairs found")
            return False

        print(
            f"  Step 2: Computing hand-eye calibrations ({len(valid_poses['left_robot'])} pose pairs)..."
        )

        # Step 3: Compute left hand-eye calibration (in left arm base frame)
        print("    Computing left wrist hand-eye calibration...")
        left_hand_eye_result = self.calibrator.calibrate_hand_eye(
            valid_poses["left_robot"], valid_poses["left_board"]
        )

        if not left_hand_eye_result["success"]:
            print(
                f"      ✗ Left hand-eye calibration failed: {left_hand_eye_result.get('error', 'Unknown error')}"
            )
            return False

        print(
            f"      ✓ Left hand-eye calibration complete (method: {left_hand_eye_result['method']})"
        )

        # Step 4: Compute right hand-eye calibration (in right arm base frame)
        print("    Computing right wrist hand-eye calibration...")
        right_hand_eye_result = self.calibrator.calibrate_hand_eye(
            valid_poses["right_robot"], valid_poses["right_board"]
        )

        if not right_hand_eye_result["success"]:
            print(
                f"      ✗ Right hand-eye calibration failed: {right_hand_eye_result.get('error', 'Unknown error')}"
            )
            return False

        print(
            f"      ✓ Right hand-eye calibration complete (method: {right_hand_eye_result['method']})"
        )

        # Step 5: Compute bimanual base transform using both hand-eye results
        print("  Step 3: Computing bimanual base transform from hand-eye results...")

        T_RIGHT_BASE_TO_LEFT_BASE = compute_bimanual_base_transform_from_calibration(
            valid_poses["left_robot"],
            valid_poses["right_robot"],
            valid_poses["left_board"],
            valid_poses["right_board"],
            left_hand_eye_result,
            right_hand_eye_result,
        )

        print(
            f"    ✓ Bimanual transform computed from {len(valid_poses['left_robot'])} pose pair(s)"
        )

        # Step 6: Transform right arm poses to left base frame
        T_left_base_to_right_base = np.linalg.inv(T_RIGHT_BASE_TO_LEFT_BASE)
        right_data["robot_poses"] = [
            T_left_base_to_right_base @ pose if pose is not None else None
            for pose in right_data["robot_poses"]
        ]
        print("  ✓ Right arm poses transformed to left base frame")

        return True

    def _estimate_board_poses(
        self, camera_data: Dict, camera_matrix: list, distortion_coeffs: list
    ) -> List:
        """Estimate board poses from corner detections"""
        camera_matrix_np = np.array(camera_matrix)
        dist_coeffs_np = np.array(distortion_coeffs)
        board_poses = []

        for corners, ids in zip(camera_data["corners"], camera_data["ids"]):
            rvec, tvec = self.calibrator.detector.estimate_pose(
                corners, ids, camera_matrix_np, dist_coeffs_np
            )
            if rvec is not None:
                board_poses.append((rvec, tvec))

        return board_poses

    def _match_valid_poses(
        self,
        left_robot_poses: List,
        right_robot_poses: List,
        left_board_poses: List,
        right_board_poses: List,
    ) -> Optional[Dict]:
        """Match valid pose pairs from both arms"""
        valid_left_robot = []
        valid_right_robot = []
        valid_left_board = []
        valid_right_board = []

        min_len = min(
            len(left_robot_poses),
            len(right_robot_poses),
            len(left_board_poses),
            len(right_board_poses),
        )

        for i in range(min_len):
            if left_robot_poses[i] is not None and right_robot_poses[i] is not None:
                valid_left_robot.append(left_robot_poses[i])
                valid_right_robot.append(right_robot_poses[i])
                valid_left_board.append(left_board_poses[i])
                valid_right_board.append(right_board_poses[i])

        if not valid_left_robot:
            return None

        return {
            "left_robot": valid_left_robot,
            "right_robot": valid_right_robot,
            "left_board": valid_left_board,
            "right_board": valid_right_board,
        }

    def collect_calibration_data(
        self,
    ) -> Dict[str, Dict[str, List]]:
        """Move through poses and collect calibration data.

        Returns:
            Dictionary mapping camera names to their calibration data.
            Each value is a dict with keys ``images``, ``corners``, ``ids``,
            and ``robot_poses`` (for hand-eye calibration).
        """
        print("\nStarting calibration sequence...")
        print("=" * 70)
        print(
            "Coordinate frame: All transforms will be expressed relative to left arm base"
        )
        print()

        # Initialize data storage
        camera_data = {
            name: {
                "images": [],
                "corners": [],
                "ids": [],
                "robot_poses": [],
                "joint_positions": [],  # Store joint positions for bimanual transform computation
                "arm_type": "left"
                if "left" in name
                else ("right" if "right" in name else None),
            }
            for name in self.cameras.keys()
        }

        poses = self.poses_data["poses"]

        for i, pose in enumerate(poses):
            print(f"\nPose {i + 1}/{len(poses)}: {pose['name']}")

            # Move robots to pose (only if we have robots)
            if self.robot_controller:
                threads = []
                if self.robot_controller.follower_r and "follower_r" in pose:
                    target_r = np.array(pose["follower_r"])
                    threads.append(
                        threading.Thread(
                            target=smooth_move_joints,
                            args=(self.robot_controller.follower_r, target_r),
                            daemon=True,
                        )
                    )
                if self.robot_controller.follower_l and "follower_l" in pose:
                    target_l = np.array(pose["follower_l"])
                    threads.append(
                        threading.Thread(
                            target=smooth_move_joints,
                            args=(self.robot_controller.follower_l, target_l),
                            daemon=True,
                        )
                    )

                if threads:
                    print(
                        f"  Moving {'both arms' if len(threads) > 1 else 'arm'} simultaneously..."
                    )
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()

                # Wait for stabilization
                print("  Waiting for stabilization...")
                time.sleep(1.0)
            else:
                # No robots - just a short delay for scene camera
                time.sleep(0.5)

            # Capture images and detect board
            print("  Capturing images...")
            for camera_name, camera in self.cameras.items():
                try:
                    # Capture frame
                    if not camera.grab():
                        raise RuntimeError(f"grab() failed for {camera_name}")
                    image = camera.get_frame().color

                    # Ensure image is contiguous and has correct dtype
                    image = np.ascontiguousarray(image, dtype=np.uint8)

                    camera_data[camera_name]["images"].append(image)

                    # Detect ChArUco board
                    corners, ids = self.calibrator.detector.detect(image)

                    if corners is not None and len(corners) > 0:
                        camera_data[camera_name]["corners"].append(corners)
                        camera_data[camera_name]["ids"].append(ids)
                        print(
                            f"    ✓ {camera_name}: ChArUco board detected ({len(corners)} corners)"
                        )

                        # For wrist cameras, store robot pose for hand-eye calibration
                        if "wrist" in camera_name and self.robot_controller:
                            # Get corresponding robot and actual joint positions
                            robot = None
                            if (
                                "right" in camera_name
                                and self.robot_controller.follower_r
                            ):
                                robot = self.robot_controller.follower_r
                            elif (
                                "left" in camera_name
                                and self.robot_controller.follower_l
                            ):
                                robot = self.robot_controller.follower_l
                            else:
                                continue

                            # Read actual joint positions from the robot
                            try:
                                actual_joint_pos = robot.get_joint_pos()

                                # Store actual joint positions for later use
                                camera_data[camera_name]["joint_positions"].append(
                                    actual_joint_pos
                                )

                                # Compute forward kinematics from actual joint positions
                                # For initial data collection, compute FK in local frames
                                # (we'll recompute after getting bimanual transform)
                                global _kinematics_cache
                                if _kinematics_cache is None:
                                    _kinematics_cache = Kinematics(
                                        get_yam_4310_linear_xml_path(),
                                        site_name="grasp_site",
                                    )

                                robot_pose = _kinematics_cache.fk(actual_joint_pos[:6])
                                camera_data[camera_name]["robot_poses"].append(
                                    robot_pose
                                )
                            except Exception as e:
                                print(
                                    f"    Warning: Failed to get actual robot pose: {e}"
                                )
                                # Store None as placeholder
                                camera_data[camera_name]["robot_poses"].append(None)
                    else:
                        print(f"    ✗ {camera_name}: Board not detected")

                except Exception as e:
                    print(f"    ✗ {camera_name}: Error - {e}")

        print("\n" + "=" * 70)
        print("Data collection complete!")
        return camera_data

    def run_calibration(
        self,
        left_wrist_camera_id: Optional[str] = None,
        right_wrist_camera_id: Optional[str] = None,
        scene_camera_ids: Optional[List[str]] = None,
    ) -> Dict:
        """Run the full calibration workflow

        Args:
            left_wrist_camera_id: Name of left wrist camera (e.g., "left_wrist")
            right_wrist_camera_id: Name of right wrist camera (e.g., "right_wrist")
            scene_camera_ids: Names of all scene cameras (multiple allowed)

        Returns:
            Calibration results dictionary
        """
        try:
            # Load configuration
            self.load_configuration()

            # Determine which cameras to calibrate
            target_cameras = []
            calibrate_left = left_wrist_camera_id is not None
            calibrate_right = right_wrist_camera_id is not None
            calibrate_scene = bool(scene_camera_ids)

            if calibrate_right:
                target_cameras.append(right_wrist_camera_id)
            if calibrate_left:
                target_cameras.append(left_wrist_camera_id)
            if calibrate_scene:
                target_cameras.extend(scene_camera_ids)

            # Check bimanual transform requirements
            global T_RIGHT_BASE_TO_LEFT_BASE
            if (
                calibrate_right
                and not calibrate_left
                and T_RIGHT_BASE_TO_LEFT_BASE is None
            ):
                raise ValueError(
                    "Cannot calibrate right wrist camera without bimanual transform.\n"
                    "Please calibrate both left and right wrist cameras together to compute the transform."
                )

            print("\nCalibration targets:")
            for camera in target_cameras:
                camera_type = "hand-eye" if "wrist" in camera else "scene"
                print(f"  - {camera} ({camera_type})")

            # Initialize hardware
            # Initialize cameras first (quick check before robot initialization)
            self.initialize_cameras(target_cameras)

            # Only initialize robots if we're calibrating wrist cameras
            if calibrate_left or calibrate_right:
                self.initialize_robots(calibrate_left, calibrate_right)
            else:
                print("\nNo wrist cameras selected - skipping robot initialization")

            # Collect data
            camera_data = self.collect_calibration_data()

            # Compute bimanual transform if both arms are calibrated
            bimanual_transform_computed = False
            if calibrate_left and calibrate_right and T_RIGHT_BASE_TO_LEFT_BASE is None:
                bimanual_transform_computed = (
                    self._compute_and_apply_bimanual_transform(
                        camera_data, left_wrist_camera_id, right_wrist_camera_id
                    )
                )

            # Run calibration for each camera
            print("\nComputing calibrations...")
            print("=" * 70)

            results = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "coordinate_frame": "left_arm_base",  # All transforms relative to left arm base
                "charuco_config": self.board_config.__dict__,
                "cameras": {},
                "quality_metrics": {},
            }

            # Add bimanual transform if computed
            if bimanual_transform_computed and T_RIGHT_BASE_TO_LEFT_BASE is not None:
                results["bimanual_transform"] = {
                    "right_base_to_left_base": T_RIGHT_BASE_TO_LEFT_BASE.tolist(),
                    "computed_from_calibration": True,
                    "description": "Transform from right arm base frame to left arm base frame",
                }
            elif T_RIGHT_BASE_TO_LEFT_BASE is not None:
                results["bimanual_transform"] = {
                    "right_base_to_left_base": T_RIGHT_BASE_TO_LEFT_BASE.tolist(),
                    "computed_from_calibration": False,
                    "description": "Transform from right arm base frame to left arm base frame (from config)",
                }

            # Process cameras in order: left wrist first (needed for scene camera conversion), then others
            # This ensures left wrist hand-eye calibration is available for scene camera conversion
            cameras_to_process = []
            if left_wrist_camera_id and left_wrist_camera_id in target_cameras:
                cameras_to_process.append(left_wrist_camera_id)
            for camera_name in target_cameras:
                if camera_name != left_wrist_camera_id:
                    cameras_to_process.append(camera_name)

            for camera_name in cameras_to_process:
                print(f"\n{camera_name}:")
                data = camera_data[camera_name]

                if len(data["corners"]) < 3:
                    print(
                        f"  ✗ Insufficient data ({len(data['corners'])} views, need >= 3)"
                    )
                    continue

                # Get factory-calibrated intrinsics from the camera SDK
                cam_type = self.camera_config.get_camera_type(camera_name) or "unknown"
                print(
                    f"  Getting intrinsics from {cam_type} SDK (factory calibration)..."
                )
                camera = self.cameras.get(camera_name)
                if not camera:
                    print("  ✗ Camera object not found")
                    continue

                cam_matrix, dist_coeffs, image_size = camera.get_intrinsics()
                print(
                    f"    ✓ fx={cam_matrix[0, 0]:.1f}, fy={cam_matrix[1, 1]:.1f}, cx={cam_matrix[0, 2]:.1f}, cy={cam_matrix[1, 2]:.1f}"
                )

                intrinsics_result = {
                    "success": True,
                    "camera_matrix": cam_matrix.tolist(),
                    "distortion_coeffs": dist_coeffs.tolist(),
                }

                # Store intrinsics
                camera_result = {
                    "type": "hand_eye" if "wrist" in camera_name else "scene",
                    "serial_number": self.camera_config.get_serial_by_name(camera_name),
                    "intrinsics": {
                        "camera_matrix": intrinsics_result["camera_matrix"],
                        "distortion_coeffs": intrinsics_result["distortion_coeffs"],
                        "image_size": list(image_size),
                        "source": f"{cam_type}_sdk_factory_calibration",
                    },
                    "num_poses_used": len(data["corners"]),
                }

                # Perform extrinsics calibration
                if "wrist" in camera_name:
                    # Hand-eye calibration
                    print("  Computing hand-eye calibration...")

                    # Check if we have robot poses
                    if not data["robot_poses"] or data["robot_poses"][0] is None:
                        print(
                            "  ✗ Forward kinematics not implemented - skipping hand-eye calibration"
                        )
                    else:
                        # Estimate camera poses (board-to-camera)
                        camera_matrix = np.array(intrinsics_result["camera_matrix"])
                        dist_coeffs = np.array(intrinsics_result["distortion_coeffs"])
                        camera_poses = []

                        for corners, ids in zip(data["corners"], data["ids"]):
                            rvec, tvec = self.calibrator.detector.estimate_pose(
                                corners, ids, camera_matrix, dist_coeffs
                            )
                            if rvec is not None:
                                camera_poses.append((rvec, tvec))

                        # Run hand-eye calibration
                        hand_eye_result = self.calibrator.calibrate_hand_eye(
                            data["robot_poses"], camera_poses
                        )

                        if hand_eye_result["success"]:
                            camera_result["hand_eye_calibration"] = hand_eye_result
                            print(
                                f"    ✓ Hand-eye calibration complete (method: {hand_eye_result['method']})"
                            )
                        else:
                            print(
                                f"  ✗ Hand-eye calibration failed: {hand_eye_result.get('error', 'Unknown error')}"
                            )

                else:
                    # Scene camera calibration
                    print("  Computing scene camera extrinsics...")

                    # Estimate camera poses (T_board_to_scene_cam)
                    camera_matrix = np.array(intrinsics_result["camera_matrix"])
                    dist_coeffs = np.array(intrinsics_result["distortion_coeffs"])
                    camera_poses = []

                    for corners, ids in zip(data["corners"], data["ids"]):
                        rvec, tvec = self.calibrator.detector.estimate_pose(
                            corners, ids, camera_matrix, dist_coeffs
                        )
                        if rvec is not None:
                            camera_poses.append((rvec, tvec))

                    # Calibrate scene camera relative to board frame first
                    scene_result = self.calibrator.calibrate_scene_camera(camera_poses)

                    if scene_result["success"]:
                        # Now convert to left arm base frame if we have left wrist camera data
                        # T_scene_cam_to_left_base = T_scene_cam_to_board @ T_board_to_left_base
                        # We can get T_board_to_left_base from the left wrist camera observations

                        if calibrate_left and left_wrist_camera_id:
                            left_data = camera_data.get(left_wrist_camera_id)
                            if left_data and "hand_eye_calibration" in results[
                                "cameras"
                            ].get(left_wrist_camera_id, {}):
                                print(
                                    "    Converting scene camera pose to left arm base frame..."
                                )

                                # Get left hand-eye calibration (T_left_cam_to_left_ee)
                                left_hand_eye = results["cameras"][
                                    left_wrist_camera_id
                                ]["hand_eye_calibration"]
                                T_left_cam_to_left_ee = np.eye(4)
                                T_left_cam_to_left_ee[:3, :3] = np.array(
                                    left_hand_eye["rotation_matrix"]
                                )
                                T_left_cam_to_left_ee[:3, 3] = np.array(
                                    left_hand_eye["translation_vector"]
                                )

                                # Get left intrinsics
                                left_camera = self.cameras.get(left_wrist_camera_id)
                                left_cam_matrix, left_dist_coeffs, _ = (
                                    left_camera.get_intrinsics()
                                )

                                # Estimate board poses from left camera
                                left_board_poses = []
                                for corners, ids in zip(
                                    left_data["corners"], left_data["ids"]
                                ):
                                    rvec, tvec = self.calibrator.detector.estimate_pose(
                                        corners, ids, left_cam_matrix, left_dist_coeffs
                                    )
                                    if rvec is not None:
                                        left_board_poses.append((rvec, tvec))

                                if left_board_poses and left_data["robot_poses"]:
                                    # Compute average T_board_to_left_base from left wrist observations
                                    print(
                                        f"      Computing T_board_to_left_base from {len(left_board_poses)} left wrist observations..."
                                    )
                                    board_to_left_base_transforms = []
                                    for i, (rvec, tvec) in enumerate(left_board_poses):
                                        if (
                                            i >= len(left_data["robot_poses"])
                                            or left_data["robot_poses"][i] is None
                                        ):
                                            continue

                                        # T_board_to_left_cam
                                        R, _ = cv2.Rodrigues(rvec)
                                        T_board_to_left_cam = np.eye(4)
                                        T_board_to_left_cam[:3, :3] = R
                                        T_board_to_left_cam[:3, 3] = tvec.flatten()

                                        # Correct chain: T^base_board = FK @ hand_eye @ ChArUco
                                        T_left_base_to_left_ee = left_data[
                                            "robot_poses"
                                        ][i]

                                        T_board_to_left_base = (
                                            T_left_base_to_left_ee
                                            @ T_left_cam_to_left_ee
                                            @ T_board_to_left_cam
                                        )

                                        board_to_left_base_transforms.append(
                                            T_board_to_left_base
                                        )

                                    if board_to_left_base_transforms:
                                        # Average the transforms
                                        from scipy.spatial.transform import Rotation

                                        rotations = [
                                            Rotation.from_matrix(T[:3, :3])
                                            for T in board_to_left_base_transforms
                                        ]
                                        translations = [
                                            T[:3, 3]
                                            for T in board_to_left_base_transforms
                                        ]

                                        avg_rotation = Rotation.concatenate(
                                            rotations
                                        ).mean()
                                        avg_translation = np.mean(translations, axis=0)

                                        T_board_to_left_base = np.eye(4)
                                        T_board_to_left_base[:3, :3] = (
                                            avg_rotation.as_matrix()
                                        )
                                        T_board_to_left_base[:3, 3] = avg_translation

                                        # Now compute T_scene_cam_to_left_base
                                        T_scene_cam_to_board = np.eye(4)
                                        T_scene_cam_to_board[:3, :3] = np.array(
                                            scene_result["rotation_matrix"]
                                        )
                                        T_scene_cam_to_board[:3, 3] = np.array(
                                            scene_result["translation_vector"]
                                        )

                                        # T_board_to_left_base is T^base_board (transforms board->base)
                                        # T_scene_cam_to_board is T^board_cam (transforms cam->board, from calibrate_scene_camera)
                                        # T^base_cam = T^base_board @ T^board_cam
                                        T_left_base_to_scene_cam = (
                                            T_board_to_left_base @ T_scene_cam_to_board
                                        )

                                        scene_cam_pos_in_left_base = (
                                            T_left_base_to_scene_cam[:3, 3]
                                        )
                                        print(
                                            f"      Scene camera position in left base: {scene_cam_pos_in_left_base}"
                                        )

                                        # Store T^base_cam: scene camera pose in left base frame
                                        # translation_vector = scene camera position in left base frame
                                        scene_result["rotation_matrix"] = (
                                            T_left_base_to_scene_cam[:3, :3].tolist()
                                        )
                                        scene_result["translation_vector"] = (
                                            T_left_base_to_scene_cam[:3, 3].tolist()
                                        )
                                        rvec_new, _ = cv2.Rodrigues(
                                            T_left_base_to_scene_cam[:3, :3]
                                        )
                                        scene_result["rotation_vector"] = (
                                            rvec_new.flatten().tolist()
                                        )
                                        scene_result["reference_frame"] = (
                                            "left_arm_base"
                                        )

                                        print(
                                            "      ✓ Scene camera pose converted to left arm base frame"
                                        )
                                    else:
                                        print(
                                            "      ! Warning: Could not convert to left arm base frame (no valid board observations)"
                                        )
                            else:
                                print(
                                    "      ! Warning: Cannot convert to left arm base frame (left wrist not calibrated)"
                                )
                        else:
                            print(
                                "      ! Warning: Scene camera extrinsics are in board frame (left wrist not calibrated)"
                            )

                        camera_result["extrinsics"] = scene_result
                        print("    ✓ Scene camera calibration complete")
                    else:
                        print("  ✗ Scene camera calibration failed")

                results["cameras"][camera_name] = camera_result

            # Compute quality metrics
            total_poses = len(self.poses_data["poses"])
            poses_with_detections = max(
                len(data["corners"]) for data in camera_data.values()
            )

            results["quality_metrics"] = {
                "overall_success": len(results["cameras"]) > 0,
                "poses_collected": total_poses,
                "poses_with_detections": poses_with_detections,
                "detection_rate": poses_with_detections / total_poses
                if total_poses > 0
                else 0,
            }

            print("\n" + "=" * 70)
            print("Calibration complete!")
            print("\nSummary:")
            print(f"  - Cameras calibrated: {len(results['cameras'])}")
            print(f"  - Poses used: {poses_with_detections}/{total_poses}")
            print(
                f"  - Detection rate: {results['quality_metrics']['detection_rate'] * 100:.1f}%"
            )
            if bimanual_transform_computed:
                print("  - Bimanual transform: Computed from calibration data ✓")
            elif T_RIGHT_BASE_TO_LEFT_BASE is not None:
                print("  - Bimanual transform: Loaded from configuration file")

            # Save results
            save_calibration_results(results, self.output_file)
            print(f"\n✓ Results saved to: {self.output_file}")

            # Store calibration result in DB
            try:
                from raiden.db.database import get_db

                get_db().add_calibration_result(results, self.output_file)
                print("✓ Calibration result recorded in DB")
            except Exception as _db_err:
                print(f"  Warning: could not record calibration in DB: {_db_err}")

            # Cleanup
            print("\nCleaning up...")

            if self.robot_controller:
                # Move the arms to the home positions.
                self.robot_controller.move_to_home_positions(simultaneous=True)

                # Stop control threads without moving to home
                # (Robots will stay in their current position - user can move them manually)
                print("  Stopping robot control threads...")
                self.robot_controller.stop_teleoperation()

                # Give threads time to fully terminate
                time.sleep(0.5)

                # Cleanup robot resources
                self.robot_controller.cleanup()

                # Give robot cleanup time to complete before exit
                time.sleep(0.5)
                print("  ✓ Robot controller cleaned up")
                print(
                    "  Note: Arms remain in current position - manually move to home if needed"
                )

            # Close cameras first (quick and safe)
            for camera in self.cameras.values():
                camera.close()
            print("  ✓ Cameras closed")

            print("\n✓ Cleanup complete")

            return results

        except KeyboardInterrupt:
            print("\n\nCalibration interrupted by user")
            if self.robot_controller and self.robot_controller.has_robots():
                self.robot_controller.emergency_stop()
            raise

        except Exception as e:
            print(f"\nError during calibration: {e}")
            import traceback

            traceback.print_exc()

            # Perform emergency stop on error
            if self.robot_controller and self.robot_controller.has_robots():
                self.robot_controller.emergency_stop()
            raise
