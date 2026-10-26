# Unitree Go1 in a factory demo

A Unitree Go1 quadruped in MuJoCo / MJX walks a demo factory hall,
passes through doors, stops at pumps, and reads their analog gauges.
Two lines of work live in this repo:

1. **Brev (trained policies)** — Brax PPO policies trained on a
   12 m x 8 m factory hall, from waypoint control to autonomous
   navigation and multi-target continuation.
2. **RunPod (Gemini Robotics ER 2 planning)** — ER 2 plans waypoints
   on a grid map, and the same Go1 controller executes them,
   including a version where the map is not known in advance.

This is a **simulation demo, not a real factory layout and not
hardware**. Walls in the grid sims are logical cells, gauge values
in the known-map runs are sim ground truth for the planner, and
low-level pose comes from sim ground truth, not SLAM with odometry
drift. Where a value was read visually from a rendered dial, that
is stated explicitly.

## Contents

- `scenes/` — factory XMLs: open hall, obstacle hall with doors,
  crates and pillars.
- `scripts/` — training, waypoint nav, autonomous nav envs,
  evaluation, rendering and gauge-reading scripts.
- `checkpoints/` — only the two final navigation checkpoints are
  included: `go1-nav/` (autonomous, v1) and `go1-nav-seq/`
  (multi-target continuation, v2). Other checkpoints (50M flat,
  obstacle) are excluded by `.gitignore`; retrain to reproduce them.
- `logs/` — training and evaluation logs are kept.
- `results/` — JSON summaries for the runs below.
- `factory_obstacle_demo.mp4`, `factory_nav_auto_demo.mp4`,
  `factory_nav_seq_demo.mp4`, `factory_nav_seq_demo_v2.mp4` —
  Brev videos (waypoint, autonomous, sequential v1, sequential v2).
- `runpod-er2/` — the RunPod ER 2 work: planners, scenes, sim
  scripts, logs, videos and close-up gauge images. See its own
  `README.md` for the full ER 2 report.

## Brev results (verified on the instance)

### Waypoint navigation, obstacle hall

Route through door centres to 3 pumps, obstacle-trained checkpoint:

- Doors: 6/6 reached, pumps: 3/3 within 0.35 m, 0 falls
- Total: 1966 steps, path 17.27 m
- Gauges (rendered needle angle): 2.5 -> 2.5003, 5.8 -> 5.8010,
  8.1 -> 8.1005 bar

Honest comparison: the 50M flat checkpoint on the same route also
reached 3/3 pumps in 1893 steps, 0 falls. Obstacle training did not
measurably speed up this waypoint route, because the route already
works once waypoints go through door centres. It did raise the
obstacle-env eval reward (7.752 -> 19.216) under random commands.

### Autonomous navigation (no waypoint controller)

Policy `Go1Nav` (see `scripts/nav_env.py`): at evaluation the only
target information is the pump index, set once at reset. The policy
outputs joint actions directly. Observation is 70 dims (45
proprioceptive + 25 map-derived navigation features); value
privileged state is 145 dims. This is map-derived sensing, not a
camera or lidar.

From a fixed south-zone start (-4.5, -2.5, facing north),
1500-step limit, final checkpoint:

- Pump 1: reached, 250 steps, 2 doors, 0.303 m
- Pump 2: reached, 268 steps, 2 doors, 0.331 m
- Pump 3: reached, 429 steps, 2 doors, 0.336 m
- Total: 3/3, 947 steps, 0 falls

Baseline (old obstacle policy, waypoint controller disabled, fixed
forward command): 0/3, 4500 steps, 0 doors crossed, stuck against
the south partition.

### Sequential test, one continuous run

`eval_nav_seq.py` resets once, then switches target on each reach
without resetting qpos/qvel, pump 1 -> 2 -> 3, 2500-step limit.

- v1 (final nav checkpoint): 1/3. Pump 1 reached in 250 steps,
  then the robot fell after 39 steps on pump 2, pump 3 never
  attempted. Diagnosis: training never started in the north zone
  and episodes terminated on the first reach, so continuing in the
  north zone with a new target was out-of-distribution.
- v2 (retrained for multi-target continuation, `nav_env_v2.py`:
  1/3 south / 1/3 middle / 1/3 north starts, episodes do not
  terminate on reach): 3/3, 372 steps, 12.32 m, 2 doors, 0 falls.
  Pump 1: 218 steps, pump 2: 101 steps, pump 3: 53 steps.

### Model architecture (both navigation checkpoints)

Brax PPO, policy/value MLPs [512, 256, 128], silu, lecun_uniform,
tanh_normal, init_noise_std 1.0, state 70 / privileged_state 145,
action 12, normalize_observations true. Read from each
checkpoint's `ppo_network_config.json`, not guessed.

## RunPod ER 2 results (simulation, 2026-10-26)

Planner: `gemini-robotics-er-2-preview` via the Gemini API
Interactions API (not a drop-in model-name change from the
original Gemini 2.0 Thinking planner; see `runpod-er2/planners/`).
The Vertex variant is mock-tested only: live Vertex tests returned
model-not-found / Interactions-unsupported, which is a private
preview restriction, not an auth or billing failure.

- Simple maze: Success, ER 2 waypoints [(0,0),(4,0),(4,4)],
  498 steps.
- Complex factory 12x8, 5 gauges: Success, step 2570/6000, plan
  matched cell-for-cell. G4 3.7 step 268, G2 5.8 step 787,
  G5 6.4 step 1104, G3 8.1 step 1498, G1 2.5 step 2540.
- Maze2 16x10, 6 real analog dial gauges: Success, step 4797.
  ER 2 v1 entered wall (1,7); v2 after feedback succeeded, order
  G4 -> G2 -> G5 -> G6 -> G3 -> G1. Visual needle reading from
  close-up renders was within 0.15 bar for all 6 gauges.
- Unknown-map exploration (planner never sees the true grid,
  72-ray `mj_ray` lidar builds the map, ER 2 chooses frontiers):
  phase 1 ended NoFrontier at 7340 steps with 3/6 gauges
  (G5, G6, G3), because unlimited-range lidar through doors
  revealed far rooms without visiting them, so the map was
  perfect (160/160 cells) while 54 free cells were never visited.
  A sweep phase (second run from (0,0), BFS nearest unvisited
  cell) then found the remaining G4, G1, G2 in 4425 steps, for
  6/6 across the two runs.
- Unified full run (exploration + sweep in one continuous run,
  no restart): in progress at the time of this README cleanup;
  its video and script are added to `runpod-er2/unknown_map/`
  when it finishes, and this section is updated with its numbers.

## How to train and run (Brev work)

Prerequisites as used on the instance: Python 3.12, a venv with
`mujoco_playground`, `brax 0.14.2`, `jax 0.10.1` with
`jax[cuda12]`, `mujoco 3.17.0`. Brax 0.14.2 calls
`jax.device_put_replicated`, removed in newer jax — hence the
jax pin and the single-GPU patch in the training scripts.

```bash
# Flat-terrain base policy (50M steps, ~12 min on an L4)
train-jax-ppo --env_name Go1JoystickFlatTerrain \
  --num_envs 1024 --num_timesteps 50000000 --seed 1 \
  --logdir /path/to/go1-50m

# Obstacle training, restored from the flat checkpoint
python scripts/train_obstacle.py

# Autonomous nav training (see scripts/train_nav.py)
python scripts/train_nav.py

# Sequential v2 training (see scripts/train_nav_seq.py)
python scripts/train_nav_seq.py
```

Headless rendering needs EGL (`MUJOCO_GL=egl`). Scripts in this
repo contain the absolute paths they were run with
(`/home/ubuntu/...`); replace those with your own checkout and
venv paths before running elsewhere.

## Excluded files

- Checkpoints other than `go1-nav/` and `go1-nav-seq/`
  (see above).
- Intermediate trajectories (`*.npz`, kept on the training
  instance; rerun the eval scripts to regenerate).
- `__pycache__/` and scratch scripts.
- Originals on the training instances were not deleted for this
  cleanup; this repo is a clean copy, not a move.
