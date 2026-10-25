"""Go1NavSeq (v2) -- multi-target navigation env, derived from nav_env.py (v1).

CHANGES vs v1 (obs size 70/145, network [512,256,128], reward scales UNCHANGED):
1. Start distribution: v1 was 80% south / 20% middle, NEVER north.
   v2 is 1/3 south (y -3.2..-1.6), 1/3 middle (y -0.4..1.0), 1/3 north
   (y 1.9..2.4, x -4.5..4.5, pushed out of north crates). North starts get
   equal weight because the v1 sequential failure was exactly this case:
   after reaching pump1 in the north zone, the policy had never trained
   starting in the north zone and walked north instead of east to pump2.
2. Multi-target episodes: v1 terminated on reach (done = fall | reached).
   v2 terminates ONLY on fall. On a reach step (dist < 0.35 m, not fall):
   the +100 reach bonus is still given for that step (in _get_reward,
   before the switch), then the env samples a NEW target pump different
   from the current one, updates target_idx/target_xy/prev_dist in info,
   increments targets_reached, and recomputes obs with the new target, so
   the next policy action uses the new target immediately. The episode
   continues to the 1000-step Brax truncation or a fall. This is done
   with jp.where (JAX-traceable), not a Python if, because step() is jitted.
3. Bookkeeping: doors_crossed stays cumulative across targets in an
   episode (eval scripts keep their own per-segment counts); reached is
   cleared to 0 on a switch; prev_zone stays the current zone.
4. Everything else is identical to v1: same 70/145 obs layout, same
   clearance/door/target features, same reward scales, same PD/action
   handling. A checkpoint trained on v1 (same obs/network size) restores
   verbatim into v2 with no weight surgery.
NOTE on evaluation: eval_nav_seq.py is run UNCHANGED (it imports the v1
Go1Nav from nav_env.py, which still terminates on reach and is then
manually switched by the eval script). That eval behaviour is exactly the
continuation v2 trains for, so v1-eval + v2-trained policy is a valid
transfer test and avoids changing the eval to fit the new env.
"""
"""Autonomous navigation env for Go1 in the obstacle factory.

No waypoint controller at test time. The policy outputs joint actions
directly. Its observation is proprioception PLUS map-derived navigation
features (documented below); the environment computes those features from
the robot pose and the KNOWN factory map (pump positions, door positions,
wall segments). This is map-derived navigation sensing, NOT a camera or
lidar, and NOT an external waypoint controller: at evaluation, the only
input given to the policy is the target pump index (part of obs); the
policy itself must steer through doors to reach it. No code outside the
policy switches waypoints or feeds joystick commands during evaluation.

Map (from factory_obstacle.xml, hall 12x8 m):
  Pumps (stop points in front of pumps, y=2.6): (-3, 2.6), (0, 2.6), (3, 2.6)
  South doors: (-2, -1.0), (+2, -1.0)   [partition y=-1.0]
  North doors: (-3, +1.5), (0, +1.5), (+3, +1.5)  [partition y=+1.5]
  Zones: south y < -1.0, middle -1.0 <= y < 1.5, north y >= 1.5
  "Current door" (a map-derived obs feature, not a controller): south door
  nearest the target while in the south zone, the north door for the target
  pump while in the middle zone, the target itself in the north zone.

Observation, policy "state" (70 dims):
  proprioception (45): local linvel 3, gyro 3, gravity 3,
    joint angles - default pose 12, joint velocities 12, last action 12
  navigation (25), in this exact order:
    [0:2]  target local dx, dy / 6
    [2]    target distance / 6
    [3:5]  sin, cos of heading error to target
    [5:7]  current-door local dx, dy / 6
    [7]    current-door distance / 6
    [8:10] sin, cos of heading error to current door
    [10:15] distances / 6 to all 5 doors (S-west, S-east, N-west, N-mid, N-east)
    [15:18] zone one-hot (south, middle, north)
    [18:22] analytic clearance / 3 in body directions (fwd, left, right, back),
           raycast against known axis-aligned wall segments and circles
           (crates/pillars + 0.25 m robot radius), capped at 3 m
    [22:25] target one-hot (pump1, pump2, pump3)
  Value "privileged_state" (145 dims) = state (70) + the same 75 extra
  features as the joystick env (clean gyro/accel/gravity/linvel/angvel,
  joint angles/velocities, actuator forces, foot contacts, foot velocities,
  foot air times, torso external force, perturbation flag).

Reward (per step, summed then * dt = 0.02, clipped to [0, 10000] as in the
joystick env, so a +50 door bonus contributes +1.0 and a +100 reach bonus
+2.0 to the episode return):
  joystick stability terms, same scales as Go1JoystickFlatTerrain:
    tracking_lin_vel 1.0 (tracks the map-derived desired velocity below),
    tracking_ang_vel 0.5, lin_vel_z -0.5, ang_vel_xy -0.05,
    orientation -5.0, dof_pos_limits -1.0, pose 0.5, termination -1.0,
    stand_still -1.0, torques -0.0002, action_rate -0.01, energy -0.001,
    feet_clearance -2.0, feet_height -0.2, feet_slip -0.1, feet_air_time 0.1
  navigation terms (these dominate door/target behaviour):
    nav_progress 1.0: (prev_dist_to_target - dist_to_target) / dt,
      i.e. velocity toward the target in m/s, clipped to [-3, 3]
    door_bonus 50.0: +50 once per upward zone change (south->middle or
      middle->north); this can only happen through a door gap, since the
      partitions are 1.2 m high walls
    reach_bonus 100.0: +100 on the step the target is reached (< 0.35 m);
      reaching also terminates the episode
    stall_penalty -1.0: -1.0 per step when speed < 0.15 m/s while the
      current-door/target waypoint is > 0.5 m away (stuck against a wall)
    nav_time -0.1: small constant time penalty per step
  Desired velocity (reward shaping ONLY, never fed to the policy and never
  used at evaluation): velocity toward the current door (or the target in
  the north zone), speed = clip(dist_to_waypoint, 0, 1.0) m/s, yaw rate =
  clip(1.5 * heading_error, -1.2, 1.2). At evaluation the policy receives
  only nav features; it has internalised the mapping nav -> joint actions.

Termination v2: fall ONLY (upvector z < 0). Reach (dist < 0.35 m) gives
the reach bonus and switches the target (see header); it does NOT end the
episode. Episode length 1000 steps is handled by the Brax training wrapper.
"""
import sys
sys.path.insert(0, "/home/ubuntu/mujoco_playground")

from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts

OBSTACLE_XML = "/home/ubuntu/factory/factory_obstacle.xml"

# Pump stop points (in front of each pump) and doors, in world coords.
PUMP_X = jp.array([-3.0, 0.0, 3.0])
PUMP_Y = 2.6
DOOR_S = jp.array([[-2.0, -1.0], [2.0, -1.0]])          # south partition
DOOR_N = jp.array([[-3.0, 1.5], [0.0, 1.5], [3.0, 1.5]])  # north partition
ALL_DOORS = jp.array([[-2.0, -1.0], [2.0, -1.0],
                      [-3.0, 1.5], [0.0, 1.5], [3.0, 1.5]])
# Horizontal wall segments (y, x_min, x_max): outer walls + partitions.
SEG_H = jp.array([
    [4.0, -6.0, 6.0], [-4.0, -6.0, 6.0],
    [-1.0, -6.0, -2.5], [-1.0, -1.5, 1.5], [-1.0, 2.5, 6.0],
    [1.5, -6.0, -3.5], [1.5, -2.5, -0.5], [1.5, 0.5, 2.5], [1.5, 3.5, 6.0],
])
# Vertical wall segments (x, y_min, y_max).
SEG_V = jp.array([[-6.0, -4.0, 4.0], [6.0, -4.0, 4.0]])
# Obstacle circles (cx, cy, radius incl. 0.25 m robot margin).
CIRC = jp.array([
    [-5.0, -0.5, 0.55], [4.5, 0.0, 0.55], [-5.0, 2.8, 0.55],
    [5.0, 2.8, 0.55], [1.5, -2.5, 0.55],
    [5.0, -2.5, 0.40], [-5.5, -3.0, 0.40], [2.0, -0.2, 0.40],
])


def default_nav_config() -> config_dict.ConfigDict:
    cfg = go1_joystick.default_config()
    cfg.naconmax = 8 * 8192
    cfg.njmax = 80
    cfg.reward_config.scales["nav_progress"] = 1.0
    cfg.reward_config.scales["door_bonus"] = 50.0
    cfg.reward_config.scales["reach_bonus"] = 100.0
    cfg.reward_config.scales["stall_penalty"] = -1.0
    cfg.reward_config.scales["nav_time"] = -0.1
    return cfg


def _yaw_from_quat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(a):
    return jp.arctan2(jp.sin(a), jp.cos(a))


def _zone(y):
    # 0 = south, 1 = middle, 2 = north
    return jp.where(y < -1.0, 0, jp.where(y < 1.5, 1, 2)).astype(jp.int32)


def _current_door(x, y, target_x):
    """Map-derived current door: south door nearest target in south zone,
    north door for the target pump in middle zone, target in north zone."""
    south_door_x = jp.where(target_x <= 0.0, -2.0, 2.0)
    # For target_x == 0, choose the south door nearer the robot.
    south_door_x = jp.where(
        target_x == 0.0, jp.where(x < 0.0, -2.0, 2.0), south_door_x)
    south_door = jp.array([south_door_x, -1.0])
    north_door = jp.array([target_x, 1.5])
    target_pt = jp.array([target_x, PUMP_Y])
    zone = _zone(y)
    door = jp.where(zone == 0, south_door,
                    jp.where(zone == 1, north_door, target_pt))
    return door


def _ray_clearance(x, y, yaw):
    """Analytic clearance in body fwd/left/right/back directions, /3, cap 1."""
    dirs_world = jp.array([
        [jp.cos(yaw), jp.sin(yaw)],          # fwd
        [-jp.sin(yaw), jp.cos(yaw)],         # left
        [jp.sin(yaw), -jp.cos(yaw)],         # right
        [-jp.cos(yaw), -jp.sin(yaw)],        # back
    ])
    out = []
    for d in range(4):
        dx, dy = dirs_world[d, 0], dirs_world[d, 1]
        best = jp.array(3.0)
        # Horizontal segments: t = (y_seg - y) / dy
        for i in range(SEG_H.shape[0]):
            y_seg, xmin, xmax = SEG_H[i, 0], SEG_H[i, 1], SEG_H[i, 2]
            t = (y_seg - y) / jp.where(jp.abs(dy) < 1e-6, 1e-6, dy)
            x_hit = x + t * dx
            hit = (t > 0.0) & (jp.abs(dy) >= 1e-6) & (x_hit >= xmin) & (x_hit <= xmax)
            best = jp.where(hit, jp.minimum(best, t), best)
        # Vertical segments: t = (x_seg - x) / dx
        for i in range(SEG_V.shape[0]):
            x_seg, ymin, ymax = SEG_V[i, 0], SEG_V[i, 1], SEG_V[i, 2]
            t = (x_seg - x) / jp.where(jp.abs(dx) < 1e-6, 1e-6, dx)
            y_hit = y + t * dy
            hit = (t > 0.0) & (jp.abs(dx) >= 1e-6) & (y_hit >= ymin) & (y_hit <= ymax)
            best = jp.where(hit, jp.minimum(best, t), best)
        # Circles: |o + t d - c|^2 = R^2, smallest positive root.
        for i in range(CIRC.shape[0]):
            cx, cy, r = CIRC[i, 0], CIRC[i, 1], CIRC[i, 2]
            ox, oy = x - cx, y - cy
            b = ox * dx + oy * dy
            c = ox * ox + oy * oy - r * r
            disc = b * b - c
            t = -b - jp.sqrt(jp.maximum(disc, 0.0))
            hit = (disc > 0.0) & (t > 0.0)
            best = jp.where(hit, jp.minimum(best, t), best)
        out.append(jp.clip(best, 0.0, 3.0) / 3.0)
    return jp.array(out)


def _nav_features(data, info):
    x, y = data.qpos[0], data.qpos[1]
    yaw = _yaw_from_quat(data.qpos[3:7])
    target_idx = info["target_idx"]
    target_x = PUMP_X[target_idx]
    target = jp.array([target_x, PUMP_Y])
    door = _current_door(x, y, target_x)
    # World -> body frame.
    def to_local(pt):
        dx, dy = pt[0] - x, pt[1] - y
        return jp.array([jp.cos(yaw) * dx + jp.sin(yaw) * dy,
                         -jp.sin(yaw) * dx + jp.cos(yaw) * dy])
    t_local = to_local(target)
    d_local = to_local(door)
    t_dist = jp.sqrt(jp.sum(t_local ** 2))
    d_dist = jp.sqrt(jp.sum(d_local ** 2))
    t_err = _wrap_angle(jp.arctan2(t_local[1], t_local[0]))
    d_err = _wrap_angle(jp.arctan2(d_local[1], d_local[0]))
    door_dists = jp.sqrt(jp.sum((ALL_DOORS - jp.array([x, y])) ** 2, axis=1)) / 6.0
    zone = _zone(y)
    zone_oh = jp.array([zone == 0, zone == 1, zone == 2], dtype=jp.float32)
    target_oh = jp.array([target_idx == 0, target_idx == 1, target_idx == 2],
                         dtype=jp.float32)
    clear = _ray_clearance(x, y, yaw)
    nav = jp.hstack([
        jp.clip(t_local / 6.0, -1.0, 1.0), jp.clip(t_dist / 6.0, 0.0, 1.0),
        jp.sin(t_err), jp.cos(t_err),
        jp.clip(d_local / 6.0, -1.0, 1.0), jp.clip(d_dist / 6.0, 0.0, 1.0),
        jp.sin(d_err), jp.cos(d_err),
        jp.clip(door_dists, 0.0, 1.0), zone_oh, clear, target_oh,
    ])
    return nav, target, door, t_dist, zone


def _desired_command(data, target, door, zone):
    """Map-derived desired velocity for reward shaping only."""
    x, y = data.qpos[0], data.qpos[1]
    yaw = _yaw_from_quat(data.qpos[3:7])
    wp = jp.where(zone == 2, target, door)
    dx, dy = wp[0] - x, wp[1] - y
    dist = jp.sqrt(dx * dx + dy * dy)
    speed = jp.clip(dist, 0.0, 1.0)
    vx = jp.where(dist > 1e-6, dx / jp.maximum(dist, 1e-6) * speed, 0.0)
    vy = jp.where(dist > 1e-6, dy / jp.maximum(dist, 1e-6) * speed, 0.0)
    vx_l = jp.cos(yaw) * vx + jp.sin(yaw) * vy
    vy_l = -jp.sin(yaw) * vx + jp.cos(yaw) * vy
    err = _wrap_angle(jp.arctan2(dy, dx) - yaw)
    yaw_rate = jp.clip(1.5 * err, -1.2, 1.2)
    return jp.array([jp.clip(vx_l, -1.5, 1.5), jp.clip(vy_l, -0.8, 0.8), yaw_rate])


class Go1NavSeq(go1_joystick.Joystick):
    """Go1NavSeq: v2 multi-target nav env (see header)."""

    def __init__(self, config=None, config_overrides=None):
        if config is None:
            config = default_nav_config()
        go1_base.Go1Env.__init__(
            self, xml_path=OBSTACLE_XML, config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        rng, key = jax.random.split(rng)
        target_idx = jax.random.randint(key, (), 0, 3)
        target_x = PUMP_X[target_idx]
        # Start zone v2: 1/3 south, 1/3 middle, 1/3 north (see header).
        rng, key = jax.random.split(rng)
        u_zone = jax.random.uniform(key)
        is_south = u_zone < (1.0 / 3.0)
        is_middle = (u_zone >= (1.0 / 3.0)) & (u_zone < (2.0 / 3.0))
        rng, key = jax.random.split(rng)
        sx = jax.random.uniform(key, (), minval=-4.5, maxval=4.5)
        rng, key = jax.random.split(rng)
        sy = jax.random.uniform(key, (), minval=-3.2, maxval=-1.6)
        rng, key = jax.random.split(rng)
        mx = jax.random.uniform(key, (), minval=-4.5, maxval=4.5)
        rng, key = jax.random.split(rng)
        my = jax.random.uniform(key, (), minval=-0.4, maxval=1.0)
        rng, key = jax.random.split(rng)
        nx = jax.random.uniform(key, (), minval=-4.5, maxval=4.5)
        rng, key = jax.random.split(rng)
        ny = jax.random.uniform(key, (), minval=1.9, maxval=2.4)
        x0 = jp.where(is_south, sx, jp.where(is_middle, mx, nx))
        y0 = jp.where(is_south, sy, jp.where(is_middle, my, ny))
        # Push out of obstacle circles (south/middle crates/pillars + north crates).
        for (cx, cy) in [(1.5, -2.5), (5.0, -2.5), (2.0, -0.2),
                          (-5.0, 2.8), (5.0, 2.8)]:
            dx, dy = x0 - cx, y0 - cy
            dist = jp.sqrt(dx * dx + dy * dy)
            push = jp.where(dist < 1.0,
                            jp.array([cx + dx / jp.maximum(dist, 1e-3) * 1.0,
                                      cy + dy / jp.maximum(dist, 1e-3) * 1.0]),
                            jp.array([x0, y0]))
            x0, y0 = push[0], push[1]
        qpos = qpos.at[0:2].set(jp.array([x0, y0]))
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (), minval=-3.14, maxval=3.14)
        from mujoco.mjx._src import math as mjx_math
        quat = mjx_math.axis_angle_to_quat(jp.array([0, 0, 1]),
                                           jp.array([yaw]))
        new_quat = mjx_math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))
        data = mjx_env.make_data(
            self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax, njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)
        target = jp.array([target_x, PUMP_Y])
        prev_dist = jp.sqrt(jp.sum((jp.array([data.qpos[0], data.qpos[1]]) - target) ** 2))
        info = {
            "rng": rng,
            "command": jp.zeros(3),
            "steps_until_next_cmd": jp.array(1000000, dtype=jp.int32),
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            "steps_until_next_pert": jp.array(1000000, dtype=jp.int32),
            "pert_duration_seconds": jp.array(0.0),
            "pert_duration": jp.array(0, dtype=jp.int32),
            "steps_since_last_pert": jp.array(0, dtype=jp.int32),
            "pert_steps": jp.array(0, dtype=jp.int32),
            "pert_dir": jp.zeros(3),
            "pert_mag": jp.array(0.0),
            "target_idx": target_idx,
            "target_xy": target,
            "prev_dist": prev_dist,
            "prev_zone": _zone(data.qpos[1]),
            "doors_crossed": jp.array(0, dtype=jp.int32),
            "reached": jp.array(0, dtype=jp.int32),
            "targets_reached": jp.array(0, dtype=jp.int32),
        }
        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps)
        contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
            for sensorid in self._feet_floor_found_sensor
        ])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)
        # Nav geometry from the NEW state (before reward/obs).
        nav, target, door, t_dist, zone = _nav_features(data, state.info)
        state.info["command"] = _desired_command(data, target, door, zone)
        obs = self._get_obs(data, state.info)
        fall = self.get_upvector(data)[-1] < 0.0
        reached = t_dist < 0.35
        done = fall  # v2: reach does NOT terminate; target switches below
        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        # Update nav bookkeeping AFTER reward (reward used old prev_dist).
        door_crossed = (zone > state.info["prev_zone"]) & (~reached)
        state.info["doors_crossed"] = state.info["doors_crossed"] + door_crossed.astype(jp.int32)
        state.info["prev_dist"] = t_dist
        state.info["prev_zone"] = zone
        state.info["reached"] = state.info["reached"] | reached.astype(jp.int32)
        # v2 multi-target switch (JAX-traceable; sample always, use only on reach).
        state.info["rng"], _key = jax.random.split(state.info["rng"])
        _offset = jax.random.randint(_key, (), 0, 2)
        _old_idx = state.info["target_idx"]
        _new_idx = (_old_idx + 1 + _offset) % 3
        _new_x = PUMP_X[_new_idx]
        _new_target = jp.array([_new_x, PUMP_Y])
        _new_dist = jp.sqrt(jp.sum((jp.array([data.qpos[0], data.qpos[1]]) - _new_target) ** 2))
        _switched = reached & (~fall)
        state.info["target_idx"] = jp.where(_switched, _new_idx, _old_idx)
        state.info["target_xy"] = jp.where(_switched, _new_target, state.info["target_xy"])
        state.info["prev_dist"] = jp.where(_switched, _new_dist, state.info["prev_dist"])
        state.info["targets_reached"] = state.info["targets_reached"] + _switched.astype(jp.int32)
        state.info["reached"] = jp.where(_switched, jp.array(0, dtype=jp.int32), state.info["reached"])
        _obs_new = self._get_obs(data, state.info)
        obs = {"state": jp.where(_switched, _obs_new["state"], obs["state"]),
               "privileged_state": jp.where(_switched, _obs_new["privileged_state"], obs["privileged_state"])}
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
        done_f = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done_f)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        return self.get_upvector(data)[-1] < 0.0

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, jax.Array]:
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (gyro + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
                      * self._config.noise_config.level * self._config.noise_config.scales.gyro)
        gravity = self.get_gravity(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (gravity + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
                         * self._config.noise_config.level * self._config.noise_config.scales.gravity)
        joint_angles = data.qpos[7:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (joint_angles + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
                              * self._config.noise_config.level * self._config.noise_config.scales.joint_pos)
        joint_vel = data.qvel[6:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (joint_vel + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
                           * self._config.noise_config.level * self._config.noise_config.scales.joint_vel)
        linvel = self.get_local_linvel(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linvel = (linvel + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
                        * self._config.noise_config.level * self._config.noise_config.scales.linvel)
        nav, _, _, _, _ = _nav_features(data, info)
        state = jp.hstack([
            noisy_linvel, noisy_gyro, noisy_gravity,
            noisy_joint_angles - self._default_pose, noisy_joint_vel,
            info["last_act"], nav,
        ])
        accelerometer = self.get_accelerometer(data)
        angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        privileged_state = jp.hstack([
            state, gyro, accelerometer, gravity, linvel, angvel,
            joint_angles - self._default_pose, joint_vel, data.actuator_force,
            info["last_contact"], feet_vel, info["feet_air_time"],
            data.xfrc_applied[self._torso_body_id, :3],
            info["steps_since_last_pert"] >= info["steps_until_next_pert"],
        ])
        return {"state": state, "privileged_state": privileged_state}

    def _get_reward(self, data: mjx.Data, action: jax.Array,
                    info: Dict[str, Any], metrics: Dict[str, Any],
                    done: jax.Array, first_contact: jax.Array,
                    contact: jax.Array) -> Dict[str, jax.Array]:
        base = {
            "tracking_lin_vel": self._reward_tracking_lin_vel(
                info["command"], self.get_local_linvel(data)),
            "tracking_ang_vel": self._reward_tracking_ang_vel(
                info["command"], self.get_gyro(data)),
            "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
            "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
            "orientation": self._cost_orientation(self.get_upvector(data)),
            "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
            "termination": self._cost_termination(done),
            "pose": self._reward_pose(data.qpos[7:]),
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]),
            "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
            "feet_slip": self._cost_feet_slip(data, contact, info),
            "feet_clearance": self._cost_feet_clearance(data),
            "feet_height": self._cost_feet_height(
                info["swing_peak"], first_contact, info),
            "feet_air_time": self._reward_feet_air_time(
                info["feet_air_time"], first_contact, info["command"]),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        }
        # Nav terms. info["prev_dist"] is still the pre-step distance here.
        _, _, _, t_dist, zone = _nav_features(data, info)
        progress = jp.clip((info["prev_dist"] - t_dist) / self.dt, -3.0, 3.0)
        door_crossed = (zone > info["prev_zone"]).astype(jp.float32)
        reached = (t_dist < 0.35).astype(jp.float32)
        speed = jp.sqrt(jp.sum(self.get_local_linvel(data)[:2] ** 2))
        # Distance to current waypoint (door or target) for stall detection.
        _, _, door, _, _ = _nav_features(data, info)
        wp = jp.where(zone == 2, info["target_xy"], door)
        wp_dist = jp.sqrt(jp.sum((jp.array([data.qpos[0], data.qpos[1]]) - wp) ** 2))
        stall = ((speed < 0.15) & (wp_dist > 0.5)).astype(jp.float32)
        base["nav_progress"] = progress
        base["door_bonus"] = door_crossed
        base["reach_bonus"] = reached
        base["stall_penalty"] = stall
        base["nav_time"] = jp.array(1.0)
        return base
