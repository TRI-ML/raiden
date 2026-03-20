# Bill of Materials

Raiden is designed for the **YAM bimanual robot system**. Exact quantities depend on whether you are running a unimanual or bimanual setup.

**PC**

| Component | Notes |
|---|---|
| PC | A GPU is required when using ZED cameras; RealSense cameras produce depth on-device |

!!! note "PC specifications"
    An NVIDIA GPU is required when using ZED cameras (ZED SDK or Fast Foundation Stereo
    depth). RealSense cameras produce depth on-device so a GPU is not needed for
    RealSense-only setups. Required GPU memory and CPU performance depend on your
    specific configuration; check the requirements before purchasing.

**Robot Arms and Controllers**

| Component | Qty | Notes | Link |
|---|---|---|---|
| YAM 6-DoF Follower Arm (standard, Pro, or Ultra) | 1–2 | One per side for bimanual | [i2rt.com](https://i2rt.com/products/yam-6-dof-arm) |
| YAM Leader Arm | 0–2 | Optional; required for leader-follower control only | [i2rt.com](https://i2rt.com/products/yam-leader) |
| 3Dconnexion SpaceMouse Compact | 0–2 | Optional; alternative to leader arms for teleoperation | [3dconnexion.com](https://3dconnexion.com/uk/product/spacemouse-compact/) |
| PCsensor USB Foot Switch | 0–1 | Optional soft e-stop | [Amazon](https://a.co/d/04osof8S) |

For bimanual teleoperation you need two follower arms and two leader arms (one pair per side). For unimanual, one of each is sufficient. Leader arms are not required if you use SpaceMouse control instead — one SpaceMouse per arm replaces the leader arms entirely.

**Cameras**

| Component | Notes | Link |
|---|---|---|
| ZED Mini | Wrist camera; one per follower arm | [stereolabs.com](https://www.stereolabs.com/store/products/zed-mini) |
| ZED 2i | Fixed scene camera | [stereolabs.com](https://www.stereolabs.com/store/products/zed-2i) |
| Intel RealSense D400 series (e.g. D405) | Alternative wrist or scene camera | [realsenseai.com](https://www.realsenseai.com/stereo-depth-cameras/) |
| USB Type-C Cable (4 m) | One per ZED Mini; shorter cables will not reach at full arm extension | [stereolabs.com](https://www.stereolabs.com/store/products/usb-type-c-cable) |
| 3D-printable wrist camera mounts | Required to attach ZED Mini or RealSense to follower wrist link | [Left](https://tri-ml.github.io/raiden/assets/zed_mounter_left.STL) / [Right](https://tri-ml.github.io/raiden/assets/zed_mounter_right.STL) |
| ChArUco board | Required for camera calibration; print and mount rigidly on a flat surface | - |

!!! warning "ZED Mini cable length"
    Use a **4 m USB Type-C cable** for each ZED Mini. Shorter cables will not
    reach from the wrist to the PC once the arm is fully extended.
