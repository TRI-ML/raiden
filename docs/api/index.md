# API Reference

## Architecture overview

Raiden is structured as a pipeline of loosely coupled modules. Data flows from
hardware through recording and conversion into a flat, policy-ready dataset.

```
Hardware (ZED / RealSense cameras, YAM arms, SpaceMouse)
    │
    ▼
CameraConfig ──► Camera (ZedCamera / RealSenseCamera)
RobotController ──► i2rt MotorChainRobot / IK via PyRoki + J-Parse
    │
    ▼
Recorder  ──► data/raw/<task>/<episode>/
    │              metadata.json
    │              robot_data.npz
    │              cameras/<name>.svo2 or .bag
    ▼
Converter ──► data/processed/<task>/<episode>/
    │              rgb/<camera>/<frame>.jpg
    │              depth/<camera>/<frame>.npz   (uint16 mm)
    │              lowdim/<camera>/<frame>.npz  (intrinsics, extrinsics, action)
    ▼
Visualizer  (Rerun)
```

---

## Modules

### [`CameraConfig`](camera_config.md)
Loads `~/.config/raiden/camera.json` and maps semantic camera names (e.g.
`left_wrist`) to hardware serial numbers, camera types (`zed` / `realsense`),
and roles (`scene` / `left_wrist` / `right_wrist`). Both ZED and RealSense
cameras can be freely mixed in a single session.

### [`Camera`](cameras.md)
Thin wrappers around the ZED SDK (`ZedCamera`) and the Intel RealSense SDK
(`RealSenseCamera`). Both expose a uniform interface: `open()`, `grab()` →
`(rgb, depth)`, `intrinsics`, `close()`. Depth is returned as `uint16`
millimetres. ZED cameras additionally support SVO2 recording and playback.

### [`RobotController`](../api/cli.md)
Manages YAM follower and leader arms over CAN via i2rt. In SpaceMouse mode it
runs a real-time IK loop using **PyRoki** and **J-Parse** to convert
end-effector velocity commands into joint targets, with manipulability-aware
damping near singularities. Handles bimanual and single-arm configurations,
e-stop integration, and optional foot-pedal control.

### [`Recorder`](recorder.md)
Orchestrates a full recording session: opens cameras once and keeps them alive
across episodes, spawns per-episode threads for teleoperation, camera capture
(`30 Hz`), and robot joint logging (`~100 Hz`). Writes SVO2/bag files and
`robot_data.npz` to `data/raw/`.

### [`Converter`](converter.md)
Offline post-processing step (`rd convert`). Reads raw SVO2/bag recordings,
synchronises multi-camera streams by timestamp, extracts JPEG frames and depth
maps, interpolates joint poses onto the camera timeline, and writes the
per-frame `lowdim.npz` files that bundle intrinsics, per-frame extrinsics,
the action vector, and the language instruction. Supports three depth backends:
RealSense IR, ZED SDK NEURAL_LIGHT, and Fast Foundation Stereo (with optional
TensorRT acceleration).

### [`Visualizer`](visualizer.md)
Loads a converted recording and streams it into a [Rerun](https://rerun.io)
viewer: RGB and depth images on a timeline, 3-D point clouds, robot joint
overlays, and the action trajectory.

### [`Calibration`](calibration.md)
Hand-eye calibration (`rd calibrate`) for wrist cameras and static extrinsic
estimation for scene cameras. Writes `~/.config/raiden/calibration_results.json` with
`T_cam2ee` per wrist camera and `T_base2cam` for scene cameras, plus a
`bimanual_transform` mapping the right-arm base into the left-arm base frame.

### `FFSDepthPredictor` / `FFSTrtDepthPredictor`
Depth estimation backends wrapping
[Fast Foundation Stereo](https://github.com/NVlabs/Fast-FoundationStereo).
`FFSDepthPredictor` runs inference in PyTorch; `FFSTrtDepthPredictor` uses
compiled TensorRT FP16 engines for faster throughput. Both are used
transparently by the Converter — TRT engines are preferred when present.

---

## Data conventions

| Item | Convention |
|---|---|
| World frame | Left-arm base frame |
| Extrinsics | `T_world_cam` (4×4 float64, row-major in npz) |
| Depth | `uint16`, millimetres |
| Joint layout | `[r_joint×6, r_grip×1, l_joint×6, l_grip×1]` |
| Action | `pos(3) + rot_mat_flat(9) + gripper(1)` per arm |
| Right wrist camera | Mounted upside-down; images rotated 180° at capture time |
