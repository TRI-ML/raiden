# Recovery data collection (HG-DAgger)

High-precision tasks often fail because clean success demonstrations never show a
near-miss, so a policy that drifts off-course has no learned way back.
**HG-DAgger** collects the missing data: you deploy the policy, take over with the
leader arms at the exact moment it starts to fail, correct it, and hand back — all
recorded as one episode with a per-frame flag marking who was in control. Feed
those corrections back into training (the ABC recipe) and the policy learns to
recover.

This is an interactive-correction loop layered on top of the local inference
command `rd infer`. It follows the ABC paper (§5.3): record the **whole rollout**
and upweight the human corrections, rather than training on interventions alone.

!!! note "HG-DAgger needs a policy; it runs on the robot"
    `rd infer` loads a model **in-process** on the robot host through a
    `ModelBridge` (see [Writing a bridge](#writing-a-modelbridge)). This is a
    separate path from [`rd serve`](serve.md), which streams observations to a
    remote policy over the chiral protocol. HG-DAgger lives in the local loop
    because that is the only place the followers are commanded and the leaders can
    be read for takeover.

## Prerequisites

- A trained checkpoint and a `ModelBridge` that loads it.
- Leader **and** follower arms connected with CAN up, plus the ZED cameras — the
  same rig as [recording](recording.md). (Ordinary `rd serve` leaves the leaders
  off; `--intervene` powers them so you can take over.)
- The model and its framework importable on the robot host.

!!! warning "Keep your hands off the leaders during start-up"
    In intervene mode the leaders are powered during homing and — with shadow
    tracking (the default) — keep moving under power during autonomy to mirror the
    followers. Do not rest your hands on them; press a trigger only when you want
    to take over.

## The operator loop

```
  ┌─ policy runs autonomously (you watch) ──┐
  │                                         │
  │   you see it start to fail              │
  ▼                                         │
  press a leader trigger ──▶ you take over INSTANTLY (the leader already mirrors
                             the follower; it goes compliant — move it to correct)
                                            │
  press the trigger again ──▶ hand back     │
                                            │
  the policy resumes from the EXACT pose    │
  you left, re-observes, re-predicts ───────┘   (repeat as many times as needed)

  Ctrl+C ──▶ arms return home, THEN the whole rollout is saved as ONE episode
```

## Running it

```bash
rd infer \
    --bridge my_model.bridge:MyBridge \
    --ckpt-path /path/to/checkpoint \
    --intervene \
    --dagger-task box_fold_dagger_r1 \
    --dagger-instruction "fold the box flap"
```

- `--intervene` turns on the takeover state machine and **auto-enables recording**.
- `--dagger-task` is the directory the rollout is saved under (also the round
  label — see [Organizing rounds](#organizing-rounds-for-the-801010-mix)).
- `--dagger-instruction` is the language prompt stored with the episode.
- `--leader-track` (default **on**) keeps the leaders mirroring the followers
  during autonomy, so takeover is instant and seam-free. Pass `--no-leader-track`
  to keep them still until you take over — they then sync to the follower over
  ~1.5 s when you first press a trigger (hands off during that sync).

Without `--intervene`, `rd infer` runs the policy autonomously; add `--record`
alone to capture a pure-policy rollout (`control_source` all 0).

### Taking over and handing back

- **Press a leader trigger** (either arm) to toggle `policy → teleop`. Takeover is
  instant — the leader already tracks the follower, so it goes compliant exactly
  where the arm is and you move it from there (joint-space delta, no jump at the
  seam). The other arm holds its position.
- **Press again** to hand back. The policy resumes from the exact pose you left,
  takes a fresh observation, re-predicts, and continues — no blending, no reset to
  home.
- Toggle as many times as you like in one rollout; every segment is flagged.
- The terminal shows the live mode (`[HUMAN]` while you drive, `[policy]`
  otherwise) and the loop rate.

!!! note "The human is the safety authority during takeover"
    `rd infer` enforces a per-step joint-delta limit (`--max-joint-delta`) while
    the **policy** drives, aborting on a runaway command. That check is skipped
    while **you** are driving — a deliberate correction is not a fault.

### Stopping

`Ctrl+C` — exactly like a normal deploy: the arms **return home first** (while CAN
is healthy), then the rollout is saved. The arms do not go limp. The heavy save
runs only after the arms are home and de-energised.

```
data/raw/box_fold_dagger_r1/0000/
    cameras/{scene_camera,left_wrist_camera,right_wrist_camera}.svo2
    robot_data.npz     # standard keys + control_source (0 = policy, 1 = you)
    metadata.json      # episode_kind=dagger, intervention_ratio
```

!!! note "One episode, both pools"
    Your interventions and the policy's own frames live in the **same** episode,
    distinguished by `control_source` — there is no separate intervention file.
    The camera is shared with inference (one ZED handle records SVO2 while still
    feeding the policy), not a second process.

## Convert — tag and optionally upweight

```bash
rd convert                        # tags only (ABC mixing recipe)
rd convert --w-intervention 8     # also upweight human-takeover frames ×8
```

`rd convert` writes a per-frame `control_source`, `is_intervention`, and
`sample_weight` into every lowdim file, and carries `episode_kind` and
`intervention_ratio` forward into the converted episode metadata. `control_source`
is resampled onto the camera grid by **nearest timestamp** (never interpolated —
it is a discrete flag). Plain teleop demos convert exactly as before, with all
weights `1.0`.

`--w-intervention` is optional. ABC uses dataset **mixing**, not per-frame
weights — if you build the 80:10:10 mix at train time you can leave it at `1.0`
and rely on the tags.

## Shardify — optionally subset

```bash
rd shardify                                          # all frames (default)
rd shardify --keep interventions                     # only your corrections
rd shardify --keep interventions_context --keep-context 15   # corrections + 15 frames of lead-in
```

You recorded the whole rollout once; choose the training subset here. Each shard
sample's `metadata.json` carries `sample_weight`, `is_intervention`, and
`episode_kind`, so a merged shard set stays splittable.

!!! warning "`--keep interventions*` empties a non-dagger dataset"
    The keep filter selects on `control_source`, which is 0 for every ordinary
    teleop frame. Running `--keep interventions` on data without human takeovers
    yields **zero** samples. Use `--keep all` (the default) on non-dagger data.

## Organizing rounds for the 80:10:10 mix

A "round" is just a directory. Give each round its own `--dagger-task`:

```
data/raw/box_fold/            # base teleop demos           ── 80 % pool (prior rounds)
data/raw/box_fold_dagger_r1/  # round-1 HG-DAgger rollouts  ── after r1, joins the prior pool
data/raw/box_fold_dagger_r2/  # round-2 HG-DAgger rollouts  ── 20 % current (split 10/10 by control_source)
```

Convert and shardify each round into its own directory, then mix the shard
directories **80 : 10 : 10** at train time:

| Pool | Where |
|---|---|
| 80 % | base + earlier-round shard dirs |
| 10 % | current round, samples with `is_intervention == True` |
| 10 % | current round, samples with `is_intervention == False` |

This implicitly overweights interventions (a minority of frames given an equal
10 % share) while keeping whole rollouts — the ABC recipe.

## Trainer-side consumption

Each shard sample's `metadata.json` carries the provenance. Either weight the loss:

```python
meta = json.loads(sample["metadata.json"])
loss = per_sample_loss * meta["sample_weight"]     # 1.0 for normal frames
```

…or route samples on `episode_kind` / `is_intervention` into the three pools and
sample them at 0.8 / 0.1 / 0.1. Raiden emits the tags; this routing is the only
change needed in the training repo.

## Writing a `ModelBridge`

`rd infer` is model-agnostic. Implement `ModelBridge` from `raiden.inference` and
point `--bridge` at it:

```python
from raiden.inference import ModelBridge
import numpy as np

class MyBridge(ModelBridge):
    def load(self, ckpt_path: str, **kwargs) -> None:
        self.model = load_my_model(ckpt_path)

    def reset(self) -> None:
        """Discard any cached action chunk — called on every handback so the
        policy re-infers on the current observation."""
        ...

    def predict(self, obs) -> np.ndarray:
        """Return a (14,) float32 joint command:
        [left_arm(6), left_grip(1), right_arm(6), right_grip(1)]."""
        ...
```

!!! warning "Action ordering is left-then-right"
    The `(14,)` joint action is `[left_arm(6), left_grip(1), right_arm(6),
    right_grip(1)]` — the same left-then-right convention Raiden uses internally
    for `--action-type ee_pose` and in [`rd serve`](serve.md). Getting the order
    wrong silently mirrors the robot.

## Options

| Flag | Default | Description |
|---|---|---|
| `--bridge` | *required* | `module.path:ClassName` implementing `ModelBridge`. |
| `--ckpt-path` | *required* | Checkpoint passed to `bridge.load()`. |
| `--intervene` | `False` | Enable the HG-DAgger takeover loop (implies `--record`). |
| `--record` | `False` | Record a pure-policy rollout (no leaders). |
| `--dagger-task` | *interactive* | Directory / round name the rollout is saved under. |
| `--dagger-instruction` | — | Language prompt stored with the episode. |
| `--session` | `False` | Looped collection — record many rollouts in one run (BOTTOM button ends an episode, `s`/`d` saves or discards) instead of one Ctrl+C-terminated rollout. |
| `--leader-track` / `--no-leader-track` | on | Shadow the leaders onto the followers during autonomy for instant takeover. |
| `--action-hz` | `30.0` | Control-loop rate; match the training data rate. |
| `--action-type` | `joint` | `joint` (14-D) or `ee_pose` (20-D EE pose solved with IK). |
| `--reset-pose` | — | Move the followers to the first frame of a `robot_data.npz` before the rollout. |
| `--max-joint-delta` | `0.8` | Per-step joint-delta safety limit (policy mode only). |

Run `rd infer --help` for the full list.

## Related

- [Evaluation](serve.md) — remote policy serving with `rd serve`.
- [Recording](recording.md) — base teleoperation demonstrations.
- [Conversion](conversion.md) and [Shardify](shardify.md) — the pipeline the tags flow through.
- [Data quality inspection](inspection.md) — check the rollout SVO2 before converting.
