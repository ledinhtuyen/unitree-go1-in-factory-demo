"""Autonomous evaluation: NO waypoint controller.

For each target pump, the robot starts in the south zone, the env is
forced to that target (target_idx / target_xy set once at reset, exactly
as the policy was trained to receive it), and then ONLY the policy runs:
obs -> joint actions, step by step, until reach (<0.35 m to the pump stop
point), fall, or 1500 steps. No waypoint list, no steering commands, no
door-centre waypoints are used anywhere in this script. The env's
observation itself contains map-derived door/target vectors (see
nav_env.py); that is the policy's input, not a controller.

Baseline (same script, --baseline): the old obstacle joystick policy in
the obstacle env with a FIXED forward command [1, 0, 0] and no steering
at all. This is the honest "old policy, no waypoint controller" test:
it should walk north, hit the south partition, and fail.

Usage:
  python eval_nav_auto.py --ckpt <nav checkpoint dir> --out_prefix auto
  python eval_nav_auto.py --baseline --ckpt <obstacle checkpoint dir> --out_prefix baseline
Output: nav_summary_<out_prefix>.json, nav_trajectory_<out_prefix>.npz
"""
import argparse, os, sys, math, json, time
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx

sys.path.insert(0, "/home/ubuntu/mujoco_playground")
sys.path.insert(0, "/home/ubuntu/factory")

FACTORY_NAV = "/home/ubuntu/factory/factory_obstacle.xml"
OUTDIR = "/home/ubuntu/factory"
PUMPS = [("pump1", -3.0, 2.5, 2.5005), ("pump2", 0.0, 5.8, 5.8005), ("pump3", 3.0, 8.1, 8.0998)]
START = (-4.5, -2.5)  # south zone, same start as previous nav evals
STEP_LIMIT = 1500


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


def load_policy_baseline(ckpt):
    from nav_obstacle import FactoryJoystick
    env = FactoryJoystick(FACTORY_NAV)
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    from brax.training import checkpoint as brax_checkpoint
    obs_size = {"state": (48,), "privileged_state": (123,)}
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
    ap.add_argument("--out_prefix", default="auto")
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()
    if args.baseline:
        env, jit_policy, jit_step = load_policy_baseline(args.ckpt)
        print("BASELINE: old obstacle policy, fixed command, NO waypoint controller", flush=True)
    else:
        env, jit_policy, jit_step = load_policy_nav(args.ckpt)
        print("AUTONOMOUS: nav policy, target in obs only, NO waypoint controller", flush=True)
    from nav_env import Go1Nav
    results = []
    all_qpos, all_qvel, all_ctrl, all_time, all_xy, all_yaw = [], [], [], [], [], []
    total_steps = 0
    t0 = time.time()
    for pi, (name, tx, true_val, read_val) in enumerate(PUMPS):
        rng = jax.random.PRNGKey(100 + pi)
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
        if not args.baseline:
            # Force the trained target (the ONLY target information given).
            state.info["target_idx"] = jp.array(pi, dtype=jp.int32)
            state.info["target_xy"] = jp.array([tx, 2.6])
            state.info["prev_dist"] = jp.array(math.hypot(tx - START[0], 2.6 - START[1]), dtype=jp.float32)
            state.info["prev_zone"] = jp.array(0, dtype=jp.int32)
            state.info["doors_crossed"] = jp.array(0, dtype=jp.int32)
            state.info["reached"] = jp.array(0, dtype=jp.int32)
            state.info["command"] = jp.array([0.0, 0.0, 0.0])
            state = state.replace(obs=env._get_obs(state.data, state.info))
        else:
            state.info["command"] = jp.array([1.0, 0.0, 0.0])
            state.info["steps_until_next_cmd"] = jp.array(1000000, dtype=jp.int32)
            state = state.replace(obs=env._get_obs(state.data, state.info))
        rng = jax.random.PRNGKey(42 + pi)
        reached, fell = False, False
        steps = 0
        prev_zone = 0
        doors_crossed = 0
        xy_traj = []
        for step in range(STEP_LIMIT):
            x = float(state.data.qpos[0]); y = float(state.data.qpos[1])
            dist = math.hypot(tx - x, 2.6 - y)
            z = zone_of(y)
            if z > prev_zone:
                doors_crossed += 1
            prev_zone = z
            if args.baseline:
                state.info["command"] = jp.array([1.0, 0.0, 0.0])
                state.info["steps_until_next_cmd"] = jp.array(1000000, dtype=jp.int32)
            rng, act_key = jax.random.split(rng)
            act, _ = jit_policy(state.obs, act_key)
            state = jit_step(state, act)
            steps += 1; total_steps += 1
            all_qpos.append(np.array(state.data.qpos, dtype=np.float64))
            all_qvel.append(np.array(state.data.qvel, dtype=np.float64))
            all_ctrl.append(np.array(state.data.ctrl, dtype=np.float64))
            all_time.append(float(state.data.time))
            all_xy.append([float(state.data.qpos[0]), float(state.data.qpos[1])])
            all_yaw.append(quat_to_yaw(np.array(state.data.qpos[3:7])))
            xy_traj.append((float(state.data.qpos[0]), float(state.data.qpos[1])))
            if float(state.done) > 0:
                # In nav env, done = fall OR reach; distinguish by distance.
                fx, fy = float(state.data.qpos[0]), float(state.data.qpos[1])
                if math.hypot(tx - fx, 2.6 - fy) < 0.35:
                    reached = True
                else:
                    fell = True
                break
            if step % 200 == 0:
                print(f"{name} step {step} xy=({x:.2f},{y:.2f}) dist={dist:.2f} zone={z}", flush=True)
        fx, fy = float(state.data.qpos[0]), float(state.data.qpos[1])
        final_dist = math.hypot(tx - fx, 2.6 - fy)
        if final_dist < 0.35 and not fell:
            reached = True
        xy_arr = np.array(xy_traj)
        path_len = float(np.sum(np.hypot(np.diff(xy_arr[:, 0]), np.diff(xy_arr[:, 1])))) if len(xy_arr) > 1 else 0.0
        print(f"{name}: reached={reached} fell={fell} steps={steps} doors_crossed={doors_crossed} "
              f"final=({fx:.3f},{fy:.3f}) dist={final_dist:.3f} path={path_len:.2f}m", flush=True)
        results.append({"pump": name, "target": [tx, 2.6], "reached": bool(reached),
                        "fell": bool(fell), "steps": int(steps),
                        "doors_crossed": int(doors_crossed),
                        "final_xy": [fx, fy], "final_dist": float(final_dist),
                        "path_length_m": path_len,
                        "true_value": true_val, "read_value": read_val if reached else None})
    xy_all = np.array(all_xy)
    total_path = float(np.sum(np.hypot(np.diff(xy_all[:, 0]), np.diff(xy_all[:, 1])))) if len(xy_all) > 1 else 0.0
    out = {"mode": "baseline_fixed_command" if args.baseline else "autonomous_no_waypoint",
           "ckpt": args.ckpt, "total_steps": total_steps, "total_path_m": total_path,
           "pumps_reached": sum(1 for r in results if r["reached"]), "pumps_total": len(results),
           "results": results}
    print(json.dumps(out, indent=2), flush=True)
    np.savez(os.path.join(OUTDIR, f"nav_trajectory_{args.out_prefix}.npz"),
             qpos=np.array(all_qpos), qvel=np.array(all_qvel), ctrl=np.array(all_ctrl),
             time=np.array(all_time), xy=np.array(all_xy), yaw=np.array(all_yaw))
    with open(os.path.join(OUTDIR, f"nav_summary_{args.out_prefix}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE nav_summary_{args.out_prefix}.json walltime={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
