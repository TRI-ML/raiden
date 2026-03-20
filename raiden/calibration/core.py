"""Camera calibration engine using ChArUco boards"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class ChArUcoBoardConfig:
    """Configuration for ChArUco calibration board"""

    squares_x: int = 9
    squares_y: int = 9
    square_length: float = 0.03  # meters (checker size: 30mm)
    marker_length: float = 0.023  # meters (marker size: 23mm)
    dictionary: str = "DICT_6X6_250"  # ArUco dictionary

    @staticmethod
    def from_dict(data: dict) -> "ChArUcoBoardConfig":
        return ChArUcoBoardConfig(
            squares_x=data["squares_x"],
            squares_y=data["squares_y"],
            square_length=data["square_length"],
            marker_length=data["marker_length"],
            dictionary=data["dictionary"],
        )


class ChArUcoDetector:
    """Detects ChArUco boards in images"""

    def __init__(self, board_config: ChArUcoBoardConfig):
        self.board_config = board_config

        # Get ArUco dictionary
        dict_name = board_config.dictionary

        if hasattr(cv2.aruco, dict_name):
            aruco_dict = getattr(cv2.aruco, dict_name)
            self.dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        else:
            raise ValueError(f"Unknown ArUco dictionary: {dict_name}")

        # Create ChArUco board
        self.board = cv2.aruco.CharucoBoard(
            size=(board_config.squares_x, board_config.squares_y),
            squareLength=board_config.square_length,
            markerLength=board_config.marker_length,
            dictionary=self.dictionary,
        )

        # Detector parameters
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

    def detect(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect ChArUco board in image

        Args:
            image: Input image (grayscale or color)

        Returns:
            Tuple of (corners, ids) or (None, None) if detection failed
            - corners: Nx2 array of corner positions
            - ids: Nx1 array of corner IDs
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect ArUco markers
        marker_corners, marker_ids, _ = self.detector.detectMarkers(gray)

        if marker_ids is None or len(marker_ids) == 0:
            return None, None

        # Interpolate ChArUco corners
        num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=marker_corners,
            markerIds=marker_ids,
            image=gray,
            board=self.board,
        )

        if num_corners == 0:
            return None, None

        return charuco_corners, charuco_ids

    def detect_with_markers(
        self, image: np.ndarray
    ) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[list], Optional[np.ndarray]
    ]:
        """Detect ChArUco board and return ArUco marker info for debugging

        Args:
            image: Input image (grayscale or color)

        Returns:
            Tuple of (charuco_corners, charuco_ids, marker_corners, marker_ids)
            - charuco_corners: Nx2 array of ChArUco corner positions (or None)
            - charuco_ids: Nx1 array of ChArUco corner IDs (or None)
            - marker_corners: List of detected ArUco marker corners (or None)
            - marker_ids: Array of detected ArUco marker IDs (or None)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect ArUco markers
        marker_corners, marker_ids, _ = self.detector.detectMarkers(gray)

        if marker_ids is None or len(marker_ids) == 0:
            return None, None, None, None

        # Interpolate ChArUco corners
        num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=marker_corners,
            markerIds=marker_ids,
            image=gray,
            board=self.board,
        )

        if num_corners == 0:
            # Return marker info even if ChArUco detection failed
            return None, None, marker_corners, marker_ids

        return charuco_corners, charuco_ids, marker_corners, marker_ids

    def estimate_pose(
        self,
        corners: np.ndarray,
        ids: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate board pose from detected corners

        Args:
            corners: Detected corner positions
            ids: Detected corner IDs
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients

        Returns:
            Tuple of (rvec, tvec) or (None, None) if estimation failed
            - rvec: Rotation vector (3x1)
            - tvec: Translation vector (3x1)
        """
        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charucoCorners=corners,
            charucoIds=ids,
            board=self.board,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
            rvec=None,
            tvec=None,
        )

        if not success:
            return None, None

        return rvec, tvec


class CameraCalibrator:
    """Calibrates cameras using ChArUco boards"""

    def __init__(self, board_config: ChArUcoBoardConfig):
        self.board_config = board_config
        self.detector = ChArUcoDetector(board_config)

    def calibrate_intrinsics(
        self,
        all_corners: List[np.ndarray],
        all_ids: List[np.ndarray],
        image_size: Tuple[int, int],
    ) -> Dict:
        """Calibrate camera intrinsics from multiple views

        Args:
            all_corners: List of detected corners from each view
            all_ids: List of detected IDs from each view
            image_size: Image size (width, height)

        Returns:
            Dictionary with calibration results:
            - camera_matrix: 3x3 intrinsic matrix
            - distortion_coeffs: Distortion coefficients
            - reprojection_error: RMS reprojection error
            - success: Whether calibration succeeded
        """
        if len(all_corners) < 3:
            return {"success": False, "error": "Need at least 3 views for calibration"}

        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = (
            cv2.aruco.calibrateCameraCharuco(
                charucoCorners=all_corners,
                charucoIds=all_ids,
                board=self.detector.board,
                imageSize=image_size,
                cameraMatrix=None,
                distCoeffs=None,
            )
        )

        return {
            "success": True,
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coeffs": dist_coeffs.flatten().tolist(),
            "reprojection_error": ret,
            "rvecs": [r.tolist() for r in rvecs],
            "tvecs": [t.tolist() for t in tvecs],
        }

    def calibrate_hand_eye(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[Tuple[np.ndarray, np.ndarray]],
        method: int = cv2.CALIB_HAND_EYE_TSAI,
    ) -> Dict:
        """Perform hand-eye calibration (eye-in-hand configuration)

        Solves the AX=XB problem where:
        - A: Transformation between robot gripper poses
        - B: Transformation between camera poses
        - X: Transformation from camera to gripper (what we're solving for)

        Args:
            robot_poses: List of 4x4 robot gripper-to-base transformation matrices
            camera_poses: List of (rvec, tvec) tuples for board-to-camera transforms
            method: OpenCV calibration method (default: Tsai)

        Returns:
            Dictionary with calibration results:
            - rotation_matrix: 3x3 rotation matrix (camera to gripper)
            - translation_vector: 3x1 translation vector (camera to gripper)
            - rotation_vector: 3x1 rotation vector (axis-angle)
            - success: Whether calibration succeeded
        """
        if len(robot_poses) < 3 or len(camera_poses) < 3:
            return {"success": False, "error": "Need at least 3 pose pairs"}

        if len(robot_poses) != len(camera_poses):
            return {
                "success": False,
                "error": "Robot poses and camera poses must have same length",
            }

        # robot_poses are FK results = T^base_ee (transforms gripper frame to base frame)
        # OpenCV calibrateHandEye expects R_gripper2base = T^base_ee, which is exactly what FK gives
        R_gripper2base = []
        t_gripper2base = []
        for pose in robot_poses:
            R_gripper2base.append(pose[:3, :3])
            t_gripper2base.append(pose[:3, 3:4])

        # Convert camera poses (board-to-camera) to target-to-camera
        # ChArUco detection gives us T_board_to_camera, which is what OpenCV expects
        R_target2cam = []
        t_target2cam = []
        for rvec, tvec in camera_poses:
            R, _ = cv2.Rodrigues(rvec)
            R_target2cam.append(R)
            t_target2cam.append(tvec)

        # Perform hand-eye calibration
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            method=method,
        )

        # Convert rotation matrix to rotation vector
        rvec_cam2gripper, _ = cv2.Rodrigues(R_cam2gripper)

        return {
            "success": True,
            "rotation_matrix": R_cam2gripper.tolist(),
            "translation_vector": t_cam2gripper.flatten().tolist(),
            "rotation_vector": rvec_cam2gripper.flatten().tolist(),
            "method": self._get_method_name(method),
        }

    def calibrate_scene_camera(
        self, camera_poses: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict:
        """Calibrate a fixed scene camera

        For a fixed scene camera viewing a fixed calibration board,
        we compute the camera's pose relative to the board frame (world frame).
        We average across multiple observations for robustness.

        Args:
            camera_poses: List of (rvec, tvec) tuples for board-to-camera transforms

        Returns:
            Dictionary with calibration results:
            - rotation_matrix: 3x3 rotation matrix (camera to board/world frame)
            - translation_vector: 3x1 translation vector (camera position in board/world frame)
            - rotation_vector: 3x1 rotation vector
            - reference_frame: "board" (the board frame is used as world frame)
            - success: Whether calibration succeeded
        """
        if len(camera_poses) < 1:
            return {"success": False, "error": "Need at least 1 pose"}

        # Average rotation and translation across all poses
        # For rotation, we convert to quaternions, average, then convert back
        rotations = []
        translations = []

        for rvec, tvec in camera_poses:
            R, _ = cv2.Rodrigues(rvec)
            rotations.append(Rotation.from_matrix(R))
            translations.append(tvec.flatten())

        # Average rotations using scipy
        if len(rotations) > 1:
            avg_rotation = Rotation.concatenate(rotations).mean()
        else:
            avg_rotation = rotations[0]

        avg_R_board_to_cam = avg_rotation.as_matrix()
        avg_t_board_to_cam = np.mean(translations, axis=0)

        # Invert to get camera-to-board transform
        # This is what we want: the camera's pose relative to the board (world frame)
        avg_R_cam_to_board = avg_R_board_to_cam.T
        avg_t_cam_to_board = -avg_R_cam_to_board @ avg_t_board_to_cam

        # Convert to rotation vector
        avg_rvec, _ = cv2.Rodrigues(avg_R_cam_to_board)

        return {
            "success": True,
            "rotation_matrix": avg_R_cam_to_board.tolist(),
            "translation_vector": avg_t_cam_to_board.tolist(),
            "rotation_vector": avg_rvec.flatten().tolist(),
            "reference_frame": "board",
        }

    @staticmethod
    def _get_method_name(method: int) -> str:
        """Get human-readable name for calibration method"""
        method_names = {
            cv2.CALIB_HAND_EYE_TSAI: "CALIB_HAND_EYE_TSAI",
            cv2.CALIB_HAND_EYE_PARK: "CALIB_HAND_EYE_PARK",
            cv2.CALIB_HAND_EYE_HORAUD: "CALIB_HAND_EYE_HORAUD",
            cv2.CALIB_HAND_EYE_ANDREFF: "CALIB_HAND_EYE_ANDREFF",
            cv2.CALIB_HAND_EYE_DANIILIDIS: "CALIB_HAND_EYE_DANIILIDIS",
        }
        return method_names.get(method, f"UNKNOWN_{method}")

    def compute_reprojection_error(
        self,
        corners: np.ndarray,
        ids: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> float:
        """Compute reprojection error for a single view

        Args:
            corners: Detected corner positions
            ids: Detected corner IDs
            rvec: Rotation vector
            tvec: Translation vector
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients

        Returns:
            RMS reprojection error in pixels
        """
        # Get 3D object points for detected corners
        obj_points = self.detector.board.getChessboardCorners()
        obj_points_detected = obj_points[ids.flatten()]

        # Project 3D points to 2D
        projected, _ = cv2.projectPoints(
            obj_points_detected, rvec, tvec, camera_matrix, dist_coeffs
        )

        # Compute error
        projected = projected.reshape(-1, 2)
        corners = corners.reshape(-1, 2)
        errors = np.linalg.norm(projected - corners, axis=1)

        return np.sqrt(np.mean(errors**2))


def load_calibration_poses(poses_file: str) -> Dict:
    """Load calibration poses from JSON file

    Args:
        poses_file: Path to calibration poses JSON file

    Returns:
        Dictionary with poses data
    """
    with open(poses_file, "r") as f:
        return json.load(f)


def save_calibration_results(results: Dict, output_file: str):
    """Save calibration results to JSON file

    Args:
        results: Calibration results dictionary
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
