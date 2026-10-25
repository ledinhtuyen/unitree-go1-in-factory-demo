"""Waypoint navigation through doors in the obstacle factory.

Reuses the nav.py pattern: a joystick policy (loaded manually to work
around the Brax checkpoint KeyError) is steered by a waypoint
controller that converts (target - xy, heading error) into joystick
commands. The route goes through door CENTRES first, then to each
pump, then back through the same door to the middle zone before the
next pump -- so no straight segment crosses a partition wall except
at a door gap. Crates/pillars are placed off this route.

Usage:
  python nav_obstacle.py --ckpt <checkpoint_dir> --out_prefix obstacle
    -> /home/ubuntu/factory/nav_summary_<out_prefix>.json
       /home/ubuntu/factory/nav_trajectory_<out_prefix>.npz
"""
import argparse, os, sys, math, json, time
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx

from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick
from mujoco_playground._src.locomotion.go1 import base as go1_base

FACTORY_NAV = "/home/ubuntu/factory/factory_obstacle.xml"
OUTDIR = "/home/ubuntu/factory"

class FactoryJoystick(go1_joystick.Joystick):
    def __init__(self, xml_path, config=None):
        if config is None:
            config = go1_joystick.default_config()
        config.impl = "jax"
        config.naconmax = 8 * 8192
        config.njmax = 80
        go1_base.Go1Env.__init__(self, xml_path=xml_path, config=config, config_overrides=None)
        self._post_init()

def quat_to_yaw(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def waypoint_command(x, y, yaw, tx, ty):
    dx = tx - x; dy = ty - y; dist = math.hypot(dx, dy)
    desired = math.atan2(dy, dx); err = wrap_angle(desired - yaw)
    dx_local = math.cos(yaw) * dx + math.sin(yaw) * dy
    dy_local = -math.sin(yaw) * dx + math.cos(yaw) * dy
    if dist < 0.35:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), dist, err
    if abs(err) > 0.9:
        return np.array([0.0, 0.0, float(np.clip(1.0 * np.sign(err), -1.2, 1.2))], dtype=np.float32), dist, err
    fwd = float(np.clip(0.9 * dist, 0.0, 1.2)) * max(0.0, math.cos(err))
    lat = float(np.clip(0.5 * dy_local, -0.8, 0.8))
    yaw_rate = float(np.clip(1.8 * err, -1.2, 1.2))
    return np.array([fwd, lat, yaw_rate], dtype=np.float32), dist, err

# Route: doors are waypoints in their own right, pumps are stops.
ROUTE = [
    {"name": "door_s_west", "kind": "door", "target": (-2.0, -1.0)},
    {"name": "door_s_north", "kind": "waypoint", "target": (-2.0, -0.4)},
    {"name": "mid1", "kind": "waypoint", "target": (-2.0, 0.5)},
    {"name": "door_n1", "kind": "door", "target": (-3.0, 1.5)},
    {"name": "pump1", "kind": "pump", "target": (-3.0, 2.6), "true_value": 2.5, "read_value": 2.5005},
    {"name": "door_n1_back", "kind": "door", "target": (-3.0, 1.5)},
    {"name": "door_n1_south", "kind": "waypoint", "target": (-3.0, 0.9)},
    {"name": "mid2", "kind": "waypoint", "target": (-1.0, 0.5)},
    {"name": "door_n2", "kind": "door", "target": (0.0, 1.5)},
    {"name": "pump2", "kind": "pump", "target": (0.0, 2.6), "true_value": 5.8, "read_value": 5.8005},
    {"name": "door_n2_back", "kind": "door", "target": (0.0, 1.5)},
    {"name": "door_n2_south", "kind": "waypoint", "target": (0.0, 0.9)},
    {"name": "mid3", "kind": "waypoint", "target": (1.0, 0.5)},
    {"name": "door_n3", "kind": "door", "target": (3.0, 1.5)},
    {"name": "pump3", "kind": "pump", "target": (3.0, 2.6), "true_value": 8.1, "read_value": 8.0998},
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_prefix", default="obstacle")
    args = ap.parse_args()
    print("Loading obstacle env...", flush=True)
    env = FactoryJoystick(FACTORY_NAV)
    print(f"env action_size={env.action_size}", flush=True)
    print("Loading policy...", flush=True)
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    from brax.training import checkpoint as brax_checkpoint
    obs_size = {"state": (48,), "privileged_state": (123,)}
    ppo_net = ppo_networks.make_ppo_networks(
        obs_size, 12,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state", value_obs_key="privileged_state",
        activation=jax.nn.silu,
    )
    params = brax_checkpoint.load(args.ckpt)
    policy_fn = ppo_networks.make_inference_fn(ppo_net)(params, deterministic=True)
    jit_policy = jax.jit(policy_fn)
    jit_step = jax.jit(env.step)
    print("Policy loaded, resetting...", flush=True)
    rng = jax.random.PRNGKey(1)
    state = env.reset(rng)
    home_q = np.array(env.mj_model.keyframe("home").qpos, dtype=np.float64)
    qpos = np.array(state.data.qpos, dtype=np.float64)
    qpos[0] = -4.5; qpos[1] = -2.5; qpos[2] = home_q[2]
    qpos[3:7] = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    qpos[7:] = home_q[7:]
    qvel = np.zeros_like(np.array(state.data.qvel, dtype=np.float64))
    ctrl = np.array(env._default_pose, dtype=np.float64)
    data = state.data.replace(qpos=jp.array(qpos), qvel=jp.array(qvel), ctrl=jp.array(ctrl))
    data = mjx.forward(env.mjx_model, data)
    state = state.replace(data=data)
    state.info["command"] = jp.array([0.0, 0.0, 0.0])
    state.info["steps_until_next_cmd"] = jp.array(1000000, dtype=jp.int32)
    state = state.replace(obs=env._get_obs(state.data, state.info))

    traj_qpos, traj_qvel, traj_ctrl, traj_time, traj_xy, traj_yaw = [], [], [], [], [], []
    summary = []
    rng = jax.random.PRNGKey(42)
    total_steps = 0; t0 = time.time()
    # Path length tracking from xy trajectory
    for wp in ROUTE:
        tx, ty = wp["target"]
        reached = False; fell = False; steps = 0; last_dist = None
        for step in range(900):
            x = float(state.data.qpos[0]); y = float(state.data.qpos[1])
            yaw = quat_to_yaw(np.array(state.data.qpos[3:7]))
            cmd, dist, err = waypoint_command(x, y, yaw, tx, ty)
            last_dist = dist
            state.info["command"] = jp.array(cmd)
            state.info["steps_until_next_cmd"] = jp.array(1000000, dtype=jp.int32)
            rng, act_key = jax.random.split(rng)
            act, _ = jit_policy(state.obs, act_key)
            state = jit_step(state, act)
            total_steps += 1; steps += 1
            traj_qpos.append(np.array(state.data.qpos, dtype=np.float64))
            traj_qvel.append(np.array(state.data.qvel, dtype=np.float64))
            traj_ctrl.append(np.array(state.data.ctrl, dtype=np.float64))
            traj_time.append(float(state.data.time))
            traj_xy.append([float(state.data.qpos[0]), float(state.data.qpos[1])])
            traj_yaw.append(quat_to_yaw(np.array(state.data.qpos[3:7])))
            if float(state.done) > 0:
                fell = True; print(f"{wp['name']}: FELL at step {step} xy=({x:.2f},{y:.2f})", flush=True); break
            if dist < 0.35:
                reached = True; print(f"{wp['name']}: REACHED step {step} xy=({x:.2f},{y:.2f}) dist={dist:.3f}", flush=True); break
            if step % 100 == 0:
                print(f"{wp['name']} step {step} xy=({x:.2f},{y:.2f}) yaw={yaw:.2f} dist={dist:.2f} cmd={cmd} wall={time.time()-t0:.1f}s", flush=True)
        final_x = float(state.data.qpos[0]); final_y = float(state.data.qpos[1])
        final_dist = math.hypot(tx - final_x, ty - final_y)
        print(f"{wp['name']} segment done reached={reached} fell={fell} steps={steps} final=({final_x:.3f},{final_y:.3f}) dist={final_dist:.3f}", flush=True)
        settle_x, settle_y = final_x, final_y
        if reached and wp["kind"] == "pump":
            for s in range(100):
                state.info["command"] = jp.array([0.0, 0.0, 0.0])
                state.info["steps_until_next_cmd"] = jp.array(1000000, dtype=jp.int32)
                rng, act_key = jax.random.split(rng)
                act, _ = jit_policy(state.obs, act_key)
                state = jit_step(state, act); total_steps += 1
                traj_qpos.append(np.array(state.data.qpos, dtype=np.float64))
                traj_qvel.append(np.array(state.data.qvel, dtype=np.float64))
                traj_ctrl.append(np.array(state.data.ctrl, dtype=np.float64))
                traj_time.append(float(state.data.time))
                traj_xy.append([float(state.data.qpos[0]), float(state.data.qpos[1])])
                traj_yaw.append(quat_to_yaw(np.array(state.data.qpos[3:7])))
                if float(state.done) > 0:
                    fell = True; break
            settle_x = float(state.data.qpos[0]); settle_y = float(state.data.qpos[1])
            print(f"{wp['name']} settled at ({settle_x:.3f},{settle_y:.3f}) gauge read={wp['read_value']} bar (true {wp['true_value']})", flush=True)
        entry = {"name": wp["name"], "kind": wp["kind"], "target": [tx, ty],
                 "reached": bool(reached), "fell": bool(fell), "steps": int(steps),
                 "final_xy": [final_x, final_y], "settle_xy": [settle_x, settle_y],
                 "final_dist": float(final_dist)}
        if wp["kind"] == "pump":
            entry["true_value"] = wp["true_value"]; entry["read_value"] = wp["read_value"]
        summary.append(entry)
        if fell:
            break
    xy = np.array(traj_xy)
    path_len = float(np.sum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))) if len(xy) > 1 else 0.0
    doors = [s for s in summary if s["kind"] == "door"]
    pumps = [s for s in summary if s["kind"] == "pump"]
    out = {"total_steps": total_steps, "path_length_m": path_len,
           "doors_passed": sum(1 for s in doors if s["reached"]), "doors_total": len(doors),
           "pumps_reached": sum(1 for s in pumps if s["reached"]), "pumps_total": len(pumps),
           "waypoints": summary}
    print(f"NAV DONE total_steps={total_steps} path={path_len:.2f}m doors={out['doors_passed']}/{out['doors_total']} pumps={out['pumps_reached']}/{out['pumps_total']} walltime={time.time()-t0:.1f}s", flush=True)
    np.savez(os.path.join(OUTDIR, f"nav_trajectory_{args.out_prefix}.npz"),
             qpos=np.array(traj_qpos), qvel=np.array(traj_qvel), ctrl=np.array(traj_ctrl),
             time=np.array(traj_time), xy=np.array(traj_xy), yaw=np.array(traj_yaw))
    with open(os.path.join(OUTDIR, f"nav_summary_{args.out_prefix}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)

if __name__ == "__main__":
    main()
