"""Sequential autonomous evaluation: ONE continuous run, NO reset between pumps.

Reset ONCE in the south zone start (-4.5, -2.5, facing north) with target
pump1. Run the nav policy deterministically. When the robot comes within
0.35 m of the current pump stop point, record that segment, then SWITCH
the target in state.info (target_idx, target_xy, prev_dist, prev_zone,
doors_crossed, reached, command) and recompute obs via env._get_obs,
clear state.done, and CONTINUE without resetting qpos/qvel/ctrl.
Order: pump1 -> pump2 -> pump3. Stop the whole run on a fall, or at a
total step limit of 2500 steps.

Doors bookkeeping: doors_crossed in state.info is RESET to 0 at each
segment switch, so per-segment door counts are clean; the total is the
sum of the per-segment counts kept in Python. After pump1, pump2/pump3
stop points are all in the north zone (y=2.6), so later segments should
need no extra door crossings -- only the first segment crosses 2 doors
(south->middle, middle->north).

Gauge values: the verified obstacle-scene readings from
gauge_obstacle_results.json (pump1 2.5->2.5003, pump2 5.8->5.8010,
pump3 8.1->8.1005 bar). No new camera gauge run is claimed here; the
values are reused from that verified run, as in eval_nav_auto.py.

Usage:
  python eval_nav_seq.py --ckpt /home/ubuntu/go1-nav/Go1Nav-20261025-023911/checkpoints/000058982400
Output: nav_summary_seq.json, nav_trajectory_seq.npz
"""
import argparse, os, sys, math, json, time
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx

sys.path.insert(0, "/home/ubuntu/mujoco_playground")
sys.path.insert(0, "/home/ubuntu/factory")

OUTDIR = "/home/ubuntu/factory"
# (name, target_x, true_value, verified_read_value)
PUMPS = [("pump1", -3.0, 2.5, 2.5003),
         ("pump2", 0.0, 5.8, 5.8010),
         ("pump3", 3.0, 8.1, 8.1005)]
START = (-4.5, -2.5)  # south zone, same start as eval_nav_auto.py
TOTAL_STEP_LIMIT = 2500


def quat_to_yaw(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def zone_of(y):
    return 0 if y < -1.0 else (1 if y < 1.5 else 2)


def load_policy_nav(ckpt):
    from nav_env import Go1Nav
    env = Go1Nav()
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    from brax.training import checkpoint as brax_checkpoint
    obs_size = {"state": (70,), "privileged_state": (145,)}
    net = ppo_networks.make_ppo_networks(
        obs_size, 12, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state", value_obs_key="privileged_state",
        activation=jax.nn.silu)
    params = brax_checkpoint.load(ckpt)
    fn = ppo_networks.make_inference_fn(net)(params, deterministic=True)
    return env, jax.jit(fn), jax.jit(env.step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    env, jit_policy, jit_step = load_policy_nav(args.ckpt)
    print("SEQUENTIAL AUTONOMOUS: one continuous run, NO reset between pumps", flush=True)

    # --- Single reset, target pump1 ---
    rng = jax.random.PRNGKey(100)
    state = env.reset(rng)
    home_q = np.array(env.mj_model.keyframe("home").qpos, dtype=np.float64)
    qpos = np.array(state.data.qpos, dtype=np.float64)
    qpos[0], qpos[1], qpos[2] = START[0], START[1], home_q[2]
    qpos[3:7] = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]  # yaw pi/2 (north)
    qpos[7:] = home_q[7:]
    qvel = np.zeros_like(np.array(state.data.qvel, dtype=np.float64))
    ctrl = np.array(env._default_pose, dtype=np.float64)
    data = state.data.replace(qpos=jp.array(qpos), qvel=jp.array(qvel), ctrl=jp.array(ctrl))
    data = mjx.forward(env.mjx_model, data)
    state = state.replace(data=data)

    def set_target(state, pi):
        """Switch target in state.info and recompute obs. No qpos/qvel reset."""
        name, tx, _, _ = PUMPS[pi]
        x = float(state.data.qpos[0]); y = float(state.data.qpos[1])
        state.info["target_idx"] = jp.array(pi, dtype=jp.int32)
        state.info["target_xy"] = jp.array([tx, 2.6])
        state.info["prev_dist"] = jp.array(math.hypot(tx - x, 2.6 - y), dtype=jp.float32)
        state.info["prev_zone"] = jp.array(zone_of(y), dtype=jp.int32)
        state.info["doors_crossed"] = jp.array(0, dtype=jp.int32)
        state.info["reached"] = jp.array(0, dtype=jp.int32)
        state.info["command"] = jp.array([0.0, 0.0, 0.0])
        state = state.replace(obs=env._get_obs(state.data, state.info))
        state = state.replace(done=jp.array(0.0))
        return state

    state = set_target(state, 0)
    rng = jax.random.PRNGKey(42)
    t0 = time.time()

    all_qpos, all_qvel, all_ctrl, all_time, all_xy, all_yaw = [], [], [], [], [], []
    segments = []
    segment_boundaries = []  # frame index (into trajectory arrays) where each segment ends
    total_steps = 0
    fell = False
    failed_segment = None

    for pi, (name, tx, true_val, read_val) in enumerate(PUMPS):
        seg_steps = 0
        seg_xy = []
        prev_zone = zone_of(float(state.data.qpos[1]))
        doors_crossed = 0
        reached = False
        print(f"--- segment {pi+1}/3: {name} target=({tx}, 2.6) starting at "
              f"({float(state.data.qpos[0]):.2f}, {float(state.data.qpos[1]):.2f}) ---", flush=True)
        while total_steps < TOTAL_STEP_LIMIT:
            x = float(state.data.qpos[0]); y = float(state.data.qpos[1])
            z = zone_of(y)
            if z > prev_zone:
                doors_crossed += 1
            prev_zone = z
            rng, act_key = jax.random.split(rng)
            act, _ = jit_policy(state.obs, act_key)
            state = jit_step(state, act)
            seg_steps += 1; total_steps += 1
            all_qpos.append(np.array(state.data.qpos, dtype=np.float64))
            all_qvel.append(np.array(state.data.qvel, dtype=np.float64))
            all_ctrl.append(np.array(state.data.ctrl, dtype=np.float64))
            all_time.append(float(state.data.time))
            all_xy.append([float(state.data.qpos[0]), float(state.data.qpos[1])])
            all_yaw.append(quat_to_yaw(np.array(state.data.qpos[3:7])))
            seg_xy.append((float(state.data.qpos[0]), float(state.data.qpos[1])))
            if total_steps % 200 == 0:
                print(f"  total {total_steps} {name} xy=({x:.2f},{y:.2f}) "
                      f"dist={math.hypot(tx - x, 2.6 - y):.2f} zone={z}", flush=True)
            if float(state.done) > 0:
                fx, fy = float(state.data.qpos[0]), float(state.data.qpos[1])
                if math.hypot(tx - fx, 2.6 - fy) < 0.35:
                    reached = True
                else:
                    fell = True
                break
        fx, fy = float(state.data.qpos[0]), float(state.data.qpos[1])
        final_dist = math.hypot(tx - fx, 2.6 - fy)
        if final_dist < 0.35 and not fell:
            reached = True
        seg_arr = np.array(seg_xy) if seg_xy else np.zeros((0, 2))
        seg_path = float(np.sum(np.hypot(np.diff(seg_arr[:, 0]), np.diff(seg_arr[:, 1])))) if len(seg_arr) > 1 else 0.0
        segment_boundaries.append(len(all_qpos) - 1 if all_qpos else -1)
        print(f"{name}: reached={reached} fell={fell} steps={seg_steps} doors_crossed={doors_crossed} "
              f"final=({fx:.3f},{fy:.3f}) dist={final_dist:.3f} path={seg_path:.2f}m", flush=True)
        segments.append({"pump": name, "target": [tx, 2.6], "reached": bool(reached),
                         "fell": bool(fell), "steps": int(seg_steps),
                         "doors_crossed": int(doors_crossed),
                         "final_xy": [fx, fy], "final_dist": float(final_dist),
                         "path_length_m": seg_path,
                         "true_value": true_val,
                         "read_value": read_val if reached else None})
        if fell or total_steps >= TOTAL_STEP_LIMIT:
            if not reached:
                failed_segment = name
            break
        # Switch to next pump WITHOUT resetting qpos/qvel/ctrl.
        if pi + 1 < len(PUMPS):
            state = set_target(state, pi + 1)

    xy_all = np.array(all_xy) if all_xy else np.zeros((0, 2))
    total_path = float(np.sum(np.hypot(np.diff(xy_all[:, 0]), np.diff(xy_all[:, 1])))) if len(xy_all) > 1 else 0.0
    total_doors = sum(s["doors_crossed"] for s in segments)
    success = all(s["reached"] for s in segments) and len(segments) == 3 and not fell
    out = {"mode": "sequential_autonomous_no_waypoint_no_reset",
           "ckpt": args.ckpt,
           "start": list(START), "total_step_limit": TOTAL_STEP_LIMIT,
           "total_steps": int(total_steps), "total_path_m": total_path,
           "total_doors_crossed": int(total_doors),
           "pumps_reached": sum(1 for s in segments if s["reached"]),
           "pumps_total": 3, "success": bool(success), "fell": bool(fell),
           "failed_segment": failed_segment,
           "segment_boundaries": [int(b) for b in segment_boundaries],
           "segments": segments,
           "gauge_note": "Gauge values are the verified obstacle-scene readings "
                         "(gauge_obstacle_results.json), reused as in eval_nav_auto.py; "
                         "no new camera gauge run is claimed in this script."}
    print(json.dumps(out, indent=2), flush=True)
    np.savez(os.path.join(OUTDIR, "nav_trajectory_seq.npz"),
             qpos=np.array(all_qpos), qvel=np.array(all_qvel), ctrl=np.array(all_ctrl),
             time=np.array(all_time), xy=np.array(all_xy), yaw=np.array(all_yaw),
             segment_boundaries=np.array(segment_boundaries, dtype=np.int64))
    with open(os.path.join(OUTDIR, "nav_summary_seq.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE nav_summary_seq.json walltime={time.time()-t0:.0f}s success={success}", flush=True)


if __name__ == "__main__":
    main()
