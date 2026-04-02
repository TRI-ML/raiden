# User Guide

## Workflow

1. **[Purchase hardware](../guide/bom.md)** - procure robot arms, cameras, and accessories. See the [Bill of Materials](../guide/bom.md) for the recommended setup.
2. **[Hardware setup](hardware.md)** - assemble robot arms, cameras, and optional foot switch.
3. **[Install software](installation.md)** - install Raiden and hardware SDKs.
4. **[Calibrate cameras](calibration.md)** - hand-eye calibration for wrist cameras and static extrinsics for the scene camera.
5. **[Record demonstrations](recording.md)** - capture teleoperation episodes with synchronized cameras and robot joint data.
6. **[Convert to dataset](conversion.md)** - extract frames, synchronize multi-camera streams, and interpolate joint poses into a structured dataset.
7. **[Shardify](shardify.md)** - export converted episodes to WebDataset sharded `.tar` files for policy training.
8. **[Evaluation](serve.md)** - run the live policy inference server (chiral protocol).
9. **[Replay](replay.md)** - replay recorded follower arm motion on the physical hardware to verify a recording.
10. **[Visualize](visualization.md)** - inspect converted recordings interactively in Rerun.

## Commands

| Command | Description |
|---|---|
| `rd list_devices` | List all connected cameras, arms, and SpaceMouse devices |
| `rd calibrate` | Calibrate cameras (hand-eye + scene extrinsics) |
| `rd teleop` | Teleoperate arms without recording |
| `rd record` | Record teleoperation demonstrations |
| `rd convert` | Convert raw recordings to a structured dataset |
| `rd shardify` | Export converted episodes to WebDataset shards |
| `rd serve` | Start the chiral policy server for live inference |
| `rd replay` | Replay recorded follower arm motion |
| `rd visualize` | Visualize a converted recording with Rerun |

Run `rd <command> --help` for options.
