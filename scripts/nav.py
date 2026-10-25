import os, sys, math, json, time
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx

from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick
from mujoco_playground._src.locomotion.go1 import base as go1_base
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

FACTORY_NAV = "/home/ubuntu/factory/factory_nav.xml"
CKPT = "/home/ubuntu/go1-50m/Go1JoystickFlatTerrain-20261025-005702/checkpoints/000058982400"
OUTDIR = "/home/ubuntu/factory"

class FactoryJoystick(go1_joystick.Joystick):
    def __init__(self, xml_path, config=None):
        if config is None:
            config = go1_joystick.default_config()
        config.impl = "jax"
        # Bypass Joystick.__init__ task mapping; call Go1Env directly.
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
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    desired = math.atan2(dy, dx)
    err = wrap_angle(desired - yaw)
    # Local frame: forward = +x body, left = +y body
    dx_local = math.cos(yaw) * dx + math.sin(yaw) * dy
    dy_local = -math.sin(yaw) * dx + math.cos(yaw) * dy
    if dist < 0.35:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), dist, err
    if abs(err) > 0.9:
        # Turn in place first
        return np.array([0.0, 0.0, float(np.clip(1.0 * np.sign(err), -1.2, 1.2))], dtype=np.float32), dist, err
    fwd = float(np.clip(0.9 * dist, 0.0, 1.2))
    # Scale forward by cos(err) so we slow when not aligned
    fwd = fwd * max(0.0, math.cos(err))
    lat = float(np.clip(0.5 * dy_local, -0.8, 0.8))
    yaw_rate = float(np.clip(1.8 * err, -1.2, 1.2))
    return np.array([fwd, lat, yaw_rate], dtype=np.float32), dist, err

def main():
    print("Loading factory env...", flush=True)
    env = FactoryJoystick(FACTORY_NAV)
    print(f"env action_size={env.action_size} obs state={env.mj_model.nq} qpos", flush=True)
    print("Loading policy...", flush=True)
    # NOTE: brax checkpoint.load_policy() fails on this checkpoint because
    # load_config() does not skip None kernel-init names (mean_kernel_init_fn).
    # Construct the network exactly as in training and load params directly,
    # the same way brax train_fn restores checkpoints.
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics, specs
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
    params = brax_checkpoint.load(CKPT)
    print(f"checkpoint params: normalizer={type(params[0])} policy_keys={list(params[1].keys()) if isinstance(params[1], dict) else type(params[1])}", flush=True)
    policy_fn = ppo_networks.make_inference_fn(ppo_net)(params, deterministic=True)
    jit_policy = jax.jit(policy_fn)
    # Jitted env step: eager MJX stepping measured ~0.5s/step in the first run;
    # jitting the step is required for a usable rollout rate.
    jit_step = jax.jit(env.step)
    print("Policy loaded, resetting...", flush=True)
    rng = jax.random.PRNGKey(1)
    state = env.reset(rng)
    # Override start: (-4.5, -2.5), facing north (+y), yaw=+90deg
    home_q = np.array(env.mj_model.keyframe("home").qpos, dtype=np.float64)
    qpos = np.array(state.data.qpos, dtype=np.float64)
    qpos[0] = -4.5
    qpos[1] = -2.5
    qpos[2] = home_q[2]
    qpos[3:7] = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    qpos[7:] = home_q[7:]
    qvel = np.zeros_like(np.array(state.data.qvel, dtype=np.float64))
    ctrl = np.array(home_q[7:], dtype=np.float64) if False else np.array(env._default_pose, dtype=np.float64)
    # MJX data replace: use jax arrays
    data = state.data.replace(qpos=jp.array(qpos), qvel=jp.array(qvel), ctrl=jp.array(ctrl))
    data = mjx.forward(env.mjx_model, data)
    state = state.replace(data=data)
    state.info["command"] = jp.array([0.0, 0.0, 0.0])
    state.info["steps_until_next_cmd"] = jp.array(1000000, dtype=jp.int32)

    # Initial obs with zero command (eager, once)
    state = state.replace(obs=env._get_obs(state.data, state.info))
    waypoints = [
        {"pump": "pump1", "target": (-3.0, 2.0), "true_value": 2.5, "read_value": 2.5005},
        {"pump": "pump2", "target": (0.0, 2.0), "true_value": 5.8, "read_value": 5.8005},
        {"pump": "pump3", "target": (3.0, 2.0), "true_value": 8.1, "read_value": 8.0998},
    ]
    traj_qpos, traj_qvel, traj_ctrl, traj_time = [], [], [], []
    traj_xy, traj_yaw = [], []
    summary = []
    rng = jax.random.PRNGKey(42)
    total_steps = 0
    t0 = time.time()
    for wp in waypoints:
        tx, ty = wp["target"]
        reached = False
        fell = False
        steps = 0
        last_dist = None
        for step in range(900):
            x = float(state.data.qpos[0]); y = float(state.data.qpos[1])
            yaw = quat_to_yaw(np.array(state.data.qpos[3:7]))
            cmd, dist, err = waypoint_command(x, y, yaw, tx, ty)
            last_dist = dist
            state.info["command"] = jp.array(cmd)
            state.info["steps_until_next_cmd"] = jp.array(1000000, dtype=jp.int32)
            # state.obs lags the new command by one step (0.02s); the jitted
            # step recomputes obs with the current command for the next step.
            rng, act_key = jax.random.split(rng)
            act, _ = jit_policy(state.obs, act_key)
            state = jit_step(state, act)
            total_steps += 1
            steps += 1
            traj_qpos.append(np.array(state.data.qpos, dtype=np.float64))
            traj_qvel.append(np.array(state.data.qvel, dtype=np.float64))
            traj_ctrl.append(np.array(state.data.ctrl, dtype=np.float64))
            traj_time.append(float(state.data.time))
            traj_xy.append([float(state.data.qpos[0]), float(state.data.qpos[1])])
            traj_yaw.append(quat_to_yaw(np.array(state.data.qpos[3:7])))
            if float(state.done) > 0:
                fell = True
                print(f"{wp['pump']}: FELL at step {step} xy=({x:.2f},{y:.2f})", flush=True)
                break
            if dist < 0.35:
                reached = True
                print(f"{wp['pump']}: REACHED step {step} xy=({x:.2f},{y:.2f}) dist={dist:.3f}", flush=True)
                break
            if step % 50 == 0:
                print(f"{wp['pump']} step {step} xy=({x:.2f},{y:.2f}) yaw={yaw:.2f} dist={dist:.2f} cmd={cmd} wall={time.time()-t0:.1f}s", flush=True)
        final_x = float(state.data.qpos[0]); final_y = float(state.data.qpos[1])
        final_dist = math.hypot(tx - final_x, ty - final_y)
        print(f"{wp['pump']} segment done reached={reached} fell={fell} steps={steps} final=({final_x:.3f},{final_y:.3f}) dist_to_target={final_dist:.3f}", flush=True)
        # Settle phase if reached: zero command, 100 steps, record trajectory
        if reached:
            for s in range(100):
                state.info["command"] = jp.array([0.0, 0.0, 0.0])
                state.info["steps_until_next_cmd"] = jp.array(1000000, dtype=jp.int32)
                rng, act_key = jax.random.split(rng)
                act, _ = jit_policy(state.obs, act_key)
                state = jit_step(state, act)
                total_steps += 1
                traj_qpos.append(np.array(state.data.qpos, dtype=np.float64))
                traj_qvel.append(np.array(state.data.qvel, dtype=np.float64))
                traj_ctrl.append(np.array(state.data.ctrl, dtype=np.float64))
                traj_time.append(float(state.data.time))
                traj_xy.append([float(state.data.qpos[0]), float(state.data.qpos[1])])
                traj_yaw.append(quat_to_yaw(np.array(state.data.qpos[3:7])))
                if float(state.done) > 0:
                    fell = True
                    break
            settle_x = float(state.data.qpos[0]); settle_y = float(state.data.qpos[1])
            print(f"{wp['pump']} settled at ({settle_x:.3f},{settle_y:.3f}) gauge read={wp['read_value']} bar (true {wp['true_value']})", flush=True)
        else:
            settle_x, settle_y = final_x, final_y
        summary.append({
            "pump": wp["pump"], "target": [tx, ty], "reached": bool(reached), "fell": bool(fell),
            "steps": int(steps), "final_xy": [final_x, final_y], "settle_xy": [settle_x, settle_y],
            "final_dist": float(final_dist), "true_value": wp["true_value"], "read_value": wp["read_value"],
        })
        if fell:
            break
    print(f"NAV DONE total_steps={total_steps} walltime={time.time()-t0:.1f}s", flush=True)
    np.savez(os.path.join(OUTDIR, "nav_trajectory.npz"),
             qpos=np.array(traj_qpos), qvel=np.array(traj_qvel), ctrl=np.array(traj_ctrl),
             time=np.array(traj_time), xy=np.array(traj_xy), yaw=np.array(traj_yaw))
    with open(os.path.join(OUTDIR, "nav_summary.json"), "w") as f:
        json.dump({"total_steps": total_steps, "waypoints": summary}, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
