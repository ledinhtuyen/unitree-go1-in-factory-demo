# Go1 + Gemini Robotics ER 2 — factory gauge missions (2026-10-26)

ER 2 (`gemini-robotics-er-2-preview`, Gemini API Interactions) plans waypoints;
a Unitree Go1 in MuJoCo/MJX executes them. Original playground repo:
https://github.com/shaoanlu/llm_mjx_go1_playground (upstream, not included here).
Only the new planners, scenes, sim scripts, logs and videos are in this bundle.

## Results (all in simulation, not hardware)

- Simple maze: Trial 1 Success, ER 2 waypoints [(0,0),(4,0),(4,4)], 498 steps.
  Video: `mission/er2_nav_full.mp4`
- Complex factory 12x8, 5 gauges: Success step 2570/6000, plan matched
  cell-for-cell. Gauges: G4 3.7 step 268, G2 5.8 step 787, G5 6.4 step 1104,
  G3 8.1 step 1498, G1 2.5 step 2540. Video: `factory/factory_er2_full.mp4`
- Maze2 16x10, 6 real analog dial gauges: Success step 4797, ER 2 v2
  (v1 entered wall (1,7)). Gauges: G4 3.7 step 437, G2 5.8 step 1038,
  G5 6.4 step 1736, G6 4.2 step 2208, G3 8.1 step 2721, G1 2.5 step 4766.
  Visual needle reading from close-up renders was within 0.15 bar for all 6.
  Video: `maze2/maze2_full.mp4`, close-ups: `maze2/closeups/`
- Unknown-map phase 1 (planner never sees the true grid, 72-ray mj_ray lidar
  builds the map, frontier exploration, ER 2 chooses frontiers): status
  NoFrontier, 7340 steps. Found 3/6 gauges (G5, G6, G3). Built map was
  perfect (160/160 cells, 0 wrong) because unlimited-range lidar through
  doors revealed far rooms without visiting them, so frontier exploration
  stopped while 54 free cells were never visited.
  Video: `unknown_map/unknown_full.mp4`
- Sweep phase (second run from (0,0), reusing phase-1 built map and found
  gauges, BFS nearest unvisited cell, no ER 2): Success, 4425 steps.
  Found remaining gauges: G4 3.7 visual 3.75 step 228, G1 2.5 visual 2.62
  step 1851, G2 5.8 visual 5.74 step 4382. Together with phase 1: 6/6 gauges.
  Video: `unknown_map/sweep_full.mp4`

## Honest limits

- Walls in these sims are logical grid cells; wall geoms are contype=0,
  so there is no physical contact. Low-level pose comes from sim ground
  truth, not SLAM with odometry drift. Gauge values in the known-map runs
  were sim ground truth for the planner; only maze2 / unknown-map /
  sweep read the dial visually from renders.
- This does not prove real-world practicality: MuJoCo, not hardware.

## Files

- `planners/er2_navigator.py` — Gemini API ER 2 adapter (Interactions API,
  stateless per call, same parsing as the original planner).
- `planners/er2_vertex_navigator.py` — Vertex variant (private preview;
  live test returned model-not-found / Interactions-unsupported, not
  an auth/billing failure). Mock-tested only.
- `factory/`, `maze2/`, `unknown_map/` — scene XML, grid, waypoints,
  sim scripts, logs, videos. `tools/watcher.py` (repo root) ran on
  the assistant VM and answered the pod's ER 2 requests via a Secure Vault
  key (the key itself is not in this bundle, and never goes in chat).
- Logs are kept; originals on the pod were not deleted.
