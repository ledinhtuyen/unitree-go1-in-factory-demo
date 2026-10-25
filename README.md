# Go1 factory demo — obstacle map with doors

A Unitree Go1 quadruped (MuJoCo / MJX, mujoco_playground, Brax PPO)
walks a demo factory hall, passes through doors, stops at three
pumps, and reads their analog gauges by camera.

This is a **simulation demo**, not a real factory layout. The hall
is generic: 12 m x 8 m, pumps along the north wall. Swap the pump
positions / wall segments in `scenes/factory_obstacle.xml` if you
have a real layout.

## What is in here

- `scenes/factory_obstacle.xml` — the obstacle hall. Two partition
  walls: a south partition at y = -1.0 with 1.0 m doors centred at
  x = -2 and x = +2, and a north partition at y = +1.5 with 1.0 m
  doors centred at x = -3, 0, +3 (one per pump). Plus 5 crates and
  3 pillars placed **off** the door-centre route. Pumps at
  (-3, 3.2), (0, 3.2), (3, 3.2), analog gauges 0-10 bar.
- `scenes/factory_nav.xml` — the original open hall (no partitions),
  for comparison.
- `scenes/factory.xml` — the original hall with needle *joints*
  (used by `gauge_test.py`); the nav scenes use fixed needle geoms
  so the Go1 qpos stays at 19 and the policy observation stays
  48 / 123.
- `scenes/go1_factory.xml` — Go1 model copy with mesh paths fixed
  to the mujoco_menagerie assets directory.
- `scripts/train_obstacle.py` — registers a custom playground env
  (`Go1JoystickObstacle`) that loads the obstacle XML, then reuses
  the stock `train_jax_ppo` loop. No playground package file is
  edited: registration and the `locomotion_params.brax_ppo_config`
  whitelist workaround are done at runtime in this script.
  It restores from the 50M flat-terrain checkpoint (see below).
- `scripts/nav_obstacle.py` — waypoint controller through door
  centres to each pump. Route: south door (-2, -1.0) -> north
  door -> pump -> back through the same door -> south of that
  door -> middle zone -> next door -> next pump. Going south
  through a door before any diagonal move matters: a diagonal
  segment straight from a pump back to the middle zone clips the
  partition wall (this was found the hard way — see Results).
- `scripts/nav.py` — the original open-hall waypoint nav.
- `scripts/gauge_test.py` / `scripts/gauge_obstacle.py` — render
  each gauge with EGL and read the red needle angle -> bar value.
- `scripts/render_obstacle.py` — render a nav trajectory to video
  (tracking camera + gauge close-ups).
- `results/` — JSON summaries (waypoints, distances, gauge
  true vs read) for the final obstacle run and the 50M baseline.
- `factory_obstacle_demo.mp4` — the demo video (640x480, 25 fps,
  25.7 s, 6.9 MB). This is the **only** video in this clean copy.

## Results (verified on the instance, not just claimed)

Obstacle training (see Training below):
- Eval reward in the obstacle env: 7.752 -> 19.216
  (starting point is the restored 50M policy evaluated *in the
  obstacle env*, so it is not comparable to the flat-terrain
  26.406 — in the obstacle env many random commands drive into
  walls and lose tracking reward).
- Final obstacle checkpoint:
  `go1-obstacle/Go1JoystickObstacle-20261025-014038/checkpoints/000058982400`
  (not included here — see Excluded files).

Door navigation, obstacle-trained checkpoint, fixed route above:
- Doors (door-centre waypoints): 6/6 reached, 0 falls
- Pumps: 3/3 reached within 0.35 m
  - pump1: 0.341 m, 83 steps; pump2: 0.340 m, 87 steps;
    pump3: 0.343 m, 92 steps
- Total: 1966 steps, path 17.27 m
- Distinct door gaps traversed: south (-2) + north (-3, 0, 3) = 4

Baseline for comparison (same route, 50M flat checkpoint, no
obstacle training): 1893 steps, path 17.50 m, also 3/3 pumps,
0 falls. **Honest conclusion: obstacle training did not
measurably speed up this waypoint route**, because the route
already works once waypoints go through door centres. What
obstacle training did change is the policy's behaviour in the
obstacle env itself (reward 7.75 -> 19.22 under random commands
with wall contact). A policy that *perceives* doors on its own
would need door/waypoint observations added to the env — that
is not in this repo, because the waypoint controller already
supplies the door routing.

Gauge reading in the obstacle scene (EGL render, needle angle):
- pump1: true 2.5 bar, read 2.5003, error 0.0003
- pump2: true 5.8 bar, read 5.8010, error 0.0010
- pump3: true 8.1 bar, read 8.1005, error 0.0005

## How to train

Prerequisites (as used on the instance): Python 3.12, a venv with
`mujoco_playground` (editable, v0.2.0), `brax 0.14.2`, `jax 0.10.1`
with `jax[cuda12]`, `mujoco 3.17.0`, `mujoco-mjx 3.17.0`,
`warp-lang 1.18.0`. Note: brax 0.14.2 calls
`jax.device_put_replicated`, removed in newer jax — that is why
jax is pinned to 0.10.1 here, plus the single-GPU patch to
`brax/training/agents/ppo/train.py` and `brax/training/pmap.py`
(replace `device_put_replicated` with `device_put` + a leading
device axis of size 1; keep the `.bak` backups).

```bash
# 1. Flat-terrain base policy (50M steps, ~12 min on an L4)
train-jax-ppo --env_name Go1JoystickFlatTerrain \
  --num_envs 1024 --num_timesteps 50000000 --seed 1 \
  --logdir /path/to/go1-50m

# 2. Obstacle training, restored from the flat checkpoint
#    (JIT ~169 s + train ~1360 s on an L4, ~25 min total)
python scripts/train_obstacle.py
#    --logdir /path/to/go1-obstacle, --num_timesteps 50000000,
#    --load_checkpoint_path <go1-50m>/.../checkpoints  (parent dir:
#    train_jax_ppo picks the latest numeric subdir)
```

Cost note: the instance used was a Brev L4 (24 GB, GCP) at
$0.90/hr running ($0.85 compute + $0.05 storage), $0.05/hr when
stopped. Training + nav + rendering for this obstacle stage
took under an hour of GPU time.

## How to run nav / gauge / video

```bash
# Waypoint nav through doors (needs a checkpoint; see Excluded files)
python scripts/nav_obstacle.py \
  --ckpt <checkpoint_dir> --out_prefix obstacle
#   -> nav_summary_obstacle.json, nav_trajectory_obstacle.npz

# Gauge reading in the obstacle scene
MUJOCO_GL=egl python scripts/gauge_obstacle.py

# Render the video from a nav trajectory
MUJOCO_GL=egl python scripts/render_obstacle.py
```

## Path assumptions (read before running elsewhere)

The scripts and XMLs in this clean copy are faithful copies of
what was actually run, so they contain the original absolute
paths. Before running on another machine, replace:

- `/home/ubuntu/factory` -> the directory you put this repo in
  (scripts expect the factory XMLs, gauge PNGs and outputs there
  unless you edit the paths at the top of each script).
- `/home/ubuntu/mujoco_playground` -> your mujoco_playground
  checkout (the Go1 mesh directory and the playground XMLs).
- `/home/ubuntu/mjx-env/bin/python` -> your venv python.

Headless rendering needs EGL (`MUJOCO_GL=egl`); the classic
MuJoCo renderer fails headless with
`OpenGL platform library has not been loaded`. The EGL
destructor warnings at render exit are harmless. The
post-training inference render in `train_jax_ppo` fails the same
way headless — training and checkpoints are unaffected.

## Excluded files (deliberately NOT in this clean copy)

> Update (autonomous nav): the exclusions below were written for the
> waypoint-mode clean copy. They are superseded in three places by the
> Autonomous navigation section below: (1) the FINAL navigation
> checkpoint checkpoints/go1-nav/000058982400 IS included (the only
> checkpoint exception in .gitignore); other checkpoints (go1-50m,
> go1-obstacle) remain excluded. (2) Training and evaluation logs
> ARE kept in logs/ (see the .gitignore logs exception). (3) TWO
> videos are kept: factory_obstacle_demo.mp4 (waypoint mode) and
> factory_nav_auto_demo.mp4 (autonomous mode).
> Everything else below still applies.

- **Checkpoints** (`go1-50m/...`, `go1-obstacle/...`, 15 MB each
  logdir, 1.6 MB per checkpoint): excluded by `.gitignore`.
  To reproduce the video without the included mp4, retrain with
  the commands above; the obstacle checkpoint path is recorded
  in Results. Originals are kept on the training instance and
  were **not** deleted for this cleanup.
- **Logs** (`*.log`): training / nav / render logs, excluded.
- **Trajectories** (`*.npz`, 0.3-1.1 MB each): intermediate
  nav trajectories, excluded; rerun `nav_obstacle.py` to
  regenerate them.
- **Other videos** (`factory_demo.mp4`, `factory_demo_50m.mp4`):
  earlier open-hall demos, excluded — only the obstacle video
  is kept, per the "at most one video" rule.
- **`__pycache__/`**, baseline scripts (`diag.py`, `test_orig.py`)
  and intermediate render scripts: excluded as scratch.
- **Original `/home/ubuntu/factory/` directory**: untouched by
  this cleanup; this `factory-github/` copy is a separate,
  clean copy, not a move.

## Autonomous navigation (no waypoint controller)

The waypoint controller in `scripts/nav_obstacle.py` is **not** used here.
`scripts/nav_env.py` defines a new MJX env, `Go1Nav`, that loads the same
`scenes/factory_obstacle.xml` and gives the policy map-derived navigation
features as part of its observation. At evaluation
(`scripts/eval_nav_auto.py`) the only target information given is the
target pump index, set once at reset; the policy then outputs joint
actions directly until it reaches the pump, falls, or hits the 1500-step
limit. No waypoint list, no door-centre steering, and no joystick
commands are used at evaluation time.

Observation, policy `state` (70 dims), in exact order:
- proprioception (45): local linvel 3, gyro 3, gravity 3, joint angles
  minus default pose 12, joint velocities 12, last action 12
- navigation (25): target local dx, dy / 6 (2), target distance / 6 (1),
  sin/cos heading error to target (2), current-door local dx, dy / 6 (2),
  current-door distance / 6 (1), sin/cos heading error to current door (2),
  distances / 6 to all 5 doors (5), zone one-hot south/middle/north (3),
  analytic clearance / 3 in body fwd/left/right/back directions (4),
  target one-hot (3)
- value `privileged_state` (145 dims) = state (70) + the same 75 extra
  features as the joystick env (clean gyro/accel/gravity/linvel/angvel,
  joint angles/velocities, actuator forces, foot contacts, foot velocities,
  foot air times, torso external force, perturbation flag)

Navigation features are computed from the robot pose and the known map
(pump positions, door positions, wall segments). The clearance is an
analytic raycast against known axis-aligned wall segments and circles
(crates/pillars + 0.25 m robot radius), capped at 3 m. This is
map-derived sensing, not a camera or lidar. The "current door" feature
is: the south door nearest the target while in the south zone, the north
door for the target pump while in the middle zone, the target itself in
the north zone. It is an observation feature, not a controller: the
policy still has to walk, turn, and pass the door by itself.

Reward (see `scripts/nav_env.py` for exact weights): the joystick
stability terms (same scales as the flat env) plus navigation terms
that dominate door/target behaviour — progress toward the target
`(prev_dist - dist) / dt` at scale 1.0, +50 per upward zone change
(south->middle or middle->north, which can only happen through a door
gap), +100 on reaching within 0.35 m (reaching also ends the episode),
-1.0 per step while stalled (speed < 0.15 m/s with the waypoint > 0.5 m
away), and -0.1 per step time penalty. The reward also tracks a
map-derived desired velocity toward the current door/target (training
shaping only; it is never fed to the policy and never used at
evaluation). Start positions are randomised per episode: 80% south zone,
20% middle zone; target pump uniform among the 3.

Training (`scripts/train_nav.py`): Brax PPO, 50,000,000 requested steps,
num_envs 1024, seed 1, env `Go1Nav`, logdir `go1-nav`, experiment
`Go1Nav-20261025-023911`. Because the nav observation is larger than
the joystick observation (70 vs 48 / 145 vs 123), the obstacle
checkpoint could not be restored verbatim. Weight surgery (documented
in `train_nav.py`): policy/value first-layer kernels copy the 45
proprioceptive rows verbatim, nav rows are N(0, 0.01), value privileged
extras copy old rows 48:123 to new rows 70:145, deeper layers and biases
copy verbatim, normalizer proprio means/stds copy and nav means are 0 /
stds 1; the optimizer state is fresh. A short script-verification run
(logdir `go1-nav-smoke`, final step 14,745,600, reward 0.016 -> 12.633)
already reached all 3 pumps autonomously, before the full run below.

Full-run eval reward (from `logs/go1-nav.log`): 0.016 -> 7.235 ->
11.366 -> 12.059 -> 12.719 -> 11.957 -> 12.797 -> 11.725 -> 11.834 ->
11.311 at final step 58,982,400 (JIT ~63 s, train ~470 s). These eval
rewards are in the nav env and are not comparable to flat/obstacle
joystick eval rewards.

Autonomous evaluation, final checkpoint, from a fixed south-zone start
(-4.5, -2.5, facing north), 1500-step limit, no waypoint controller
(`results/nav_summary_auto.json`):
- pump1: REACHED, 250 steps, 2 doors crossed, final distance 0.303 m,
  path 6.11 m, no fall
- pump2: REACHED, 268 steps, 2 doors crossed, final distance 0.331 m,
  path 6.75 m, no fall
- pump3: REACHED, 429 steps, 2 doors crossed, final distance 0.336 m,
  path 9.97 m, no fall
- Total: 3/3 pumps, 947 steps, 0 falls

Baseline ablation, same test with the old obstacle policy and the
waypoint controller disabled (fixed forward command [1, 0, 0], no
steering; `results/nav_summary_baseline_auto.json`):
- pump1/2/3: NOT reached, 1500 steps each, 0 doors crossed, stuck at
  (-4.52, -1.24) against the south partition, path ~1.45 m each, no fall
- Total: 0/3 pumps, 4500 steps. This is the honest contrast: the old
  policy cannot find doors on its own; the nav policy can.

Gauge reading after reaching (re-run in the obstacle scene,
`results/gauge_obstacle_results.json`): pump1 true 2.5 -> read 2.5003
(err 0.0003), pump2 5.8 -> 5.8010 (err 0.0010), pump3 8.1 -> 8.1005
(err 0.0005) bar.

Video: `factory_nav_auto_demo.mp4` — the actual autonomous trajectory
(640x480, 25 fps, 15.5 s, 387 frames, 4.0 MB), tracking camera, plus
gauge close-ups. `factory_obstacle_demo.mp4` is kept for comparison:
it shows the earlier waypoint-controller mode.

## Model architecture (final navigation checkpoint)

Checkpoint included here: `checkpoints/go1-nav/000058982400/` (1.7 MB,
14 files, `commit_success.txt` present). All values below are read
from that checkpoint's `ppo_network_config.json` and from
`scripts/nav_env.py` / `scripts/train_nav.py` — not guessed.

- Network type: Brax PPO, two MLPs (policy + value), no vision, no
  distributional critic (`use_distributional_critic: false`).
- Policy MLP hidden sizes: [512, 256, 128]; value MLP hidden sizes:
  [512, 256, 128].
- Activation: `silu` (swish). Kernel initializer: `lecun_uniform` for
  both networks. Mean kernel init: none (default).
- Distribution: `tanh_normal` (bounded actions), `init_noise_std: 1.0`,
  `noise_std_type: scalar`, `state_dependent_std: false`,
  `num_quantiles: 32` (unused when distributional critic is off).
- Observation: policy obs key `state`, shape [70], dtype float32;
  value obs key `privileged_state`, shape [145], dtype float32.
  Breakdown: see the observation list in the Autonomous navigation
  section above (45 proprioceptive + 25 navigation = 70; 70 + 75
  privileged extras = 145).
- Normalizer: `normalize_observations: true` (running-statistics
  normalizer over `state` and `privileged_state`, stored in the
  checkpoint; nav dims were initialised mean 0 / std 1 in surgery
  and then updated during training).
- Action: `action_size: 12` — one motor target per Go1 joint
  (4 legs x abduction, hip, knee). The env applies
  `motor_targets = default_pose + action * 0.5` (action_scale 0.5)
  with the Go1 PD gains Kp 35.0 / Kd 0.5 from the joystick config.
- Training config (from `scripts/train_nav.py`, which takes the tuned
  Go1 PPO config): num_envs 1024, requested num_timesteps 50,000,000
  (Brax final step 58,982,400), seed 1, episode_length 1000,
  discounting 0.97, learning_rate 3e-4, entropy_cost 1e-2,
  unroll_length 20, num_minibatches 32, num_updates_per_batch 4,
  batch_size 256, max_grad_norm 1.0, reward_scaling 1.0,
  num_evals 10, num_eval_envs 128.
- Other checkpoints are NOT included (see Excluded files): only the v1
  final navigation checkpoint above and the v2 sequential checkpoint
  (`checkpoints/go1-nav-seq/000058982400/`) are excepted from the
  checkpoint exclusion in `.gitignore`. Retrain with `scripts/train_nav.py`
  to reproduce the obstacle/flat checkpoints, which are deliberately excluded.

## Sequential test (one continuous run, no reset between pumps)

How it differs from the per-pump reset test: `scripts/eval_nav_auto.py`
resets the robot to the south-zone start (-4.5, -2.5) before EACH pump.
`scripts/eval_nav_seq.py` resets ONCE with target pump1, then, on reaching
a pump (within 0.35 m), switches the target in `state.info`
(`target_idx`, `target_xy`, `prev_dist`, `prev_zone` to the current zone,
`doors_crossed` reset to 0 for the new segment, `reached` reset to 0,
`command` zero), recomputes the observation with `env._get_obs`, clears
`state.done`, and continues WITHOUT resetting qpos/qvel/ctrl, in the
order pump1 -> pump2 -> pump3. Total step limit 2500. Doors crossed in
`state.info` are reset per segment, so per-segment counts are clean;
the total is the sum of the segment counts. After pump1, pump2/pump3 stop
points are all in the north zone (y=2.6), so later segments should need
no extra door crossings -- only the first segment crosses 2 doors
(south->middle, middle->north).

Result, final nav checkpoint, verified from `results/nav_summary_seq.json`
(this run did NOT succeed end-to-end -- do not claim otherwise):
- Segment 1, pump1: REACHED, 250 steps, 2 doors crossed, final distance
  0.303 m, path 6.11 m, gauge 2.5 -> 2.5003 bar (verified obstacle-scene
  reading, reused as in the per-pump test).
- Segment 2, pump2: FAILED -- the robot FELL after 39 steps, ending at
  (-2.987, 3.057), 3.022 m from the pump2 stop point, 0 doors crossed,
  path 0.78 m. Pump3 was never attempted.
- Total: 289 steps, path 6.94 m, doors crossed 2, pumps reached 1/3,
  fell=True, failed segment pump2.

Why it failed (diagnosis from `nav_trajectory_seq.npz`, not a script
error): segment 1 is byte-identical in outcome to the per-pump pump1 run
(250 steps, same final position), so the target switch itself worked --
after the switch the robot kept moving (it did not end immediately as
"reached", because the pump2 distance was 3.02 m). From the pump1 stop
point it kept walking north instead of turning east toward pump2, its
up-vector fell from 1.00 to below 0 at the end, and it toppled near the
pump1 structure (y ~ 3.06, pump at y=3.2). Training never started an
episode in the north zone (80% south, 20% middle starts) and episodes
always terminated on reach, so "continue in the north zone with a new
target after a reach" is out-of-distribution for this policy. Fixing it
properly means retraining with north-zone starts and/or multi-target
episodes that do not terminate on the first reach -- that retraining was
not done for this test.

Video: `factory_nav_seq_demo.mp4` -- the actual sequential trajectory
(640x480, 25 fps, 4.9 s, 123 frames, 1.3 MB), with per-segment labels;
the pump1 gauge close-up is inserted at the segment-1 boundary only,
because pump2's gauge was not read in this run (its segment failed).
Script: `scripts/eval_nav_seq.py`, render: `scripts/render_nav_seq.py`,
trajectory: `nav_trajectory_seq.npz` (kept on the training instance;
segment boundaries are stored in the npz as `segment_boundaries`).

## Sequential test v2 (retrain for multi-target continuation)

Why v2 exists: the v1 sequential test (see "Sequential test" above) reached
pump1 in 250 steps, then fell after 39 steps on the pump2 segment, walking
north instead of east, and never attempted pump3 (total 289 steps, 1/3).
Diagnosis: v1 training never started in the north zone (80% south / 20%
middle) and episodes terminated on the first reach, so continuing in the
north zone with a new target was out-of-distribution.

Env changes (`scripts/nav_env_v2.py`, class `Go1NavSeq`; v1 is kept as
`scripts/nav_env.py` and backed up on the instance as `nav_env_v1.py`):
- Start distribution: 1/3 south (y -3.2..-1.6), 1/3 middle (y -0.4..1.0),
  1/3 north (y 1.9..2.4, x -4.5..4.5, pushed out of north crates).
- Episodes do NOT terminate on reach: done is fall only. On a reach step
  (dist < 0.35 m, not fall) the +100 reach bonus is given for that step,
  then the env samples a new target pump different from the current one
  (JAX-traceable `jp.where`, not a Python if), updates target_idx /
  target_xy / prev_dist in info, increments targets_reached, and recomputes
  obs with the new target. The episode continues to the 1000-step
  truncation or a fall.
- Observation (70/145), network ([512, 256, 128]), and reward scales are
  UNCHANGED from v1, so the v1 nav checkpoint restores verbatim.

Retraining (`scripts/train_nav_seq.py`, log `logs/go1-nav-seq.log`):
- Restored directly from `go1-nav/Go1Nav-20261025-023911/checkpoints/000058982400`
  (no weight surgery needed; the surgery fallback from `train_nav.py` is kept
  in the script but was not used). 50,000,000 steps, num_envs 1024, seed 1,
  logdir `go1-nav-seq`, experiment `Go1NavSeq-20261025-032046`.
- Eval reward (v2 episodes accumulate multiple reach bonuses over 1000
  steps, so these numbers are NOT comparable to v1's 11.311):
  29.698 (step 0, restored policy) -> 45.185 -> 44.454 -> 47.361 -> 47.537 ->
  50.104 -> 49.970 -> 48.125 -> 46.752 -> 46.839 at final step 58,982,400
  (JIT 68.6 s, train ~488 s). Final checkpoint
  `go1-nav-seq/Go1NavSeq-20261025-032046/checkpoints/000058982400`
  (`commit_success.txt` present). Checkpoint included here:
  `checkpoints/go1-nav-seq/000058982400/` (1.7 MB, 14 files,
  `commit_success.txt` present). Architecture is identical to v1 (see Model
  architecture): policy/value MLPs [512, 256, 128], silu, tanh_normal,
  state 70 / privileged_state 145, action 12, normalize_observations true,
  verified from this checkpoint's `ppo_network_config.json`.

Sequential re-test, eval script UNCHANGED except the checkpoint path
(`scripts/eval_nav_seq.py` still imports the v1 env, whose manual target
switch on reach is exactly the continuation v2 trains for):
`results/nav_summary_seq_v2.json`, log `logs/eval_seq_v2.log`,
video `factory_nav_seq_demo_v2.mp4` (640x480, 25 fps, 243 frames, 9.72 s,
2,011,257 bytes; the v1 video `factory_nav_seq_demo.mp4` is kept unchanged):
- pump1: REACHED, 218 steps, 2 doors, final distance 0.318 m, path 6.15 m,
  gauge 2.5 -> 2.5003 bar
- pump2: REACHED, 101 steps, 0 doors, final distance 0.342 m, path 3.11 m,
  gauge 5.8 -> 5.8010 bar (this segment fell in v1 after 39 steps)
- pump3: REACHED, 53 steps, 0 doors, final distance 0.348 m, path 2.95 m,
  gauge 8.1 -> 8.1005 bar
- Total: 3/3 pumps, 372 steps, path 12.32 m, doors 2, 0 falls, success true.
  Direct comparison to v1: pump1 250 -> 218 steps, pump2 fell at 39 steps
  -> reached in 101 steps, pump3 never attempted -> reached in 53 steps,
  total 289 steps / 1/3 -> 372 steps / 3/3.
Gauge values are the verified obstacle-scene readings reused as in the v1
sequential test; no new camera gauge run is claimed in the sequential script.
