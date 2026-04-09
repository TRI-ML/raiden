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

For bimanual teleoperation you need two follower arms and two leader arms (one pair per side). For unimanual, one of each is sufficient.

!!! note "Teleop controller"
    You need either **YAM Leader Arms** or **SpaceMouse** devices for teleoperation — one per arm. Leader arms are not required if you use SpaceMouse control, and vice versa.

**Cameras (recommended setup)**

The recommended camera setup is **2 × ZED Mini** (one per wrist) + **1 × ZED 2i** (scene). ZED cameras share a common clock, making multi-camera synchronization straightforward.

| Component | Qty | Notes | Link |
|---|---|---|---|
| ZED Mini | 1–2 | Wrist camera; one per follower arm | [stereolabs.com](https://www.stereolabs.com/store/products/zed-mini) |
| ZED 2i | 1 | Fixed scene camera | [stereolabs.com](https://www.stereolabs.com/store/products/zed-2i) |
| USB Type-C Cable (4 m) | 1–2 | One per ZED Mini; shorter cables will not reach at full arm extension | [stereolabs.com](https://www.stereolabs.com/store/products/usb-type-c-cable) |
| 3D-printable wrist camera mounts | 1–2 | Required to attach ZED Mini to follower wrist link | [Left](https://tri-ml.github.io/raiden/assets/zed_mounter_left.STL) / [Right](https://tri-ml.github.io/raiden/assets/zed_mounter_right.STL) |
| ChArUco board | 1 | Required for camera calibration; print and mount rigidly on a flat surface | - |

!!! warning "ZED Mini cable length"
    Use a **4 m USB Type-C cable** for each ZED Mini. Shorter cables will not
    reach from the wrist to the PC once the arm is fully extended.

**Cameras (optional — mix and match)**

Raiden also supports Intel RealSense D400-series cameras as wrist or scene cameras, and ZED and RealSense cameras can be mixed freely within the same session.

!!! warning "Prefer ZED cameras for multi-camera setups"
    ZED cameras share a common wall-clock timestamp, making synchronization across cameras straightforward. RealSense cameras have known synchronization limitations — see [Hardware Setup](hardware.md#camera-configuration) for details. If you must mix camera types, use ZED cameras for wrist roles where timestamp alignment is most critical.

| Component | Notes | Link |
|---|---|---|
| Intel RealSense D400 series (e.g. D405) | Alternative wrist or scene camera; no GPU required for depth | [realsenseai.com](https://www.realsenseai.com/stereo-depth-cameras/) |

**Fin-ray gripper (optional)**

Fin-ray compliant grippers conform to object shapes for robust and gentle grasping. See [Hardware Setup](hardware.md#fin-ray-gripper) for assembly instructions.

| Component | Qty | Notes | Link |
|---|---|---|---|
| 3D-printable fin-ray adapter | 1–2 | PA6-CF; one per arm | [finray_adapter.STL](https://tri-ml.github.io/raiden/assets/finray_adapter.STL) |
| 3D-printable short fin-ray finger | 2–4 | TPU 95A HF; two per gripper; **tested and recommended** | [finray_short.STL](https://tri-ml.github.io/raiden/assets/finray_short.STL) |
| 3D-printable long fin-ray finger | 2–4 | TPU 95A HF; two per gripper; not fully tested | [finray_long.STL](https://tri-ml.github.io/raiden/assets/finray_long.STL) |
| M3×0.8mm socket head screws | 1 pack of 100 | 20 per arm | [McMaster-Carr 91292A112](https://www.mcmaster.com/91292A112/) |
| Female hex standoffs | 4–8 | One per arm: 4 (single), 8 (bimanual) | [McMaster-Carr 94868A713](https://www.mcmaster.com/94868A713/) |
| Friction sheet | — | Applied to finger contact surface for better grip | [Amazon](https://a.co/d/0iTBIF98) |
