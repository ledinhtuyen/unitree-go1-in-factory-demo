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
