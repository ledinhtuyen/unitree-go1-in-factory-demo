import os
os.environ["MUJOCO_GL"] = "egl"
import sys
sys.path.insert(0, "/workspace/llm_mjx_go1_playground")
os.chdir("/workspace/llm_mjx_go1_playground")
import jax, numpy as np
from functools import partial
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x
import mujoco
from mujoco_playground._src import mjx_env
from typing import List
from src.environment.generate_maze_scene_xml import generate_maze_xml
from src.environment.custom_go1_env import CustomGo1JoystickEnv
from src.environment.env_wrapper import Go1Env
from src.environment.traversability_map import TraversabilityMap, TraversabilityMapConfig
from src.control.controller_factory import ControllerFactory
from src.control.position_controller import PositionController, PositionControllerParams
from src.control.algorithms.mlp import MLPPolicy, MLPPolicyParams
from src.mission_executer import EpisodeResult, MissionConfig, MissionExecuter
from src.utils import load_yaml_with_jinja
from google import genai
from src.planning.er2_navigator import GeminiRoboticsER2Navigator

print("MAZE_GEN", flush=True)
grid, random_maze_xml = generate_maze_xml()
print("GRID:\n", grid, flush=True)
open("scene_mjx_maze_flat_terrain.xml", "w").write(random_maze_xml)
system_config = load_yaml_with_jinja("examples/configs/llm_navigation.yaml")
rng = jax.random.PRNGKey(system_config["environment"]["mujoco_playground"]["random_key"])
mpenv = CustomGo1JoystickEnv(xml_path="scene_mjx_maze_flat_terrain.xml")
env = Go1Env(env=mpenv, env_cfg=mpenv.env_cfg)
trav_map_config = TraversabilityMapConfig.from_dict(system_config["environment"]["map"])
trav_map = TraversabilityMap(trav_map_config)
trav_map.load_from_grid(grid)

velocity_kick_range = [0.0, 0.0]
kick_duration_range = [0.05, 0.2]
def sample_pert(rng, env, state):
    rng, key1, key2 = jax.random.split(rng, 3)
    pert_mag = jax.random.uniform(key1, minval=velocity_kick_range[0], maxval=velocity_kick_range[1])
    duration_seconds = jax.random.uniform(key2, minval=kick_duration_range[0], maxval=kick_duration_range[1])
    duration_steps = jax.numpy.round(duration_seconds / env.dt).astype(jax.numpy.int32)
    state.info["pert_mag"] = pert_mag
    state.info["pert_duration"] = duration_steps
    state.info["pert_duration_seconds"] = duration_seconds
    return rng
def update_position_history(position_history, current_position):
    if len(position_history) == 0:
        position_history.append(current_position.astype(np.int32)); return position_history
    if np.array_equal(position_history[-1].astype(np.int32), current_position.astype(np.int32)):
        return position_history
    position_history.append(current_position.astype(np.int32)); return position_history
def run_one_episode(env, command_generator, command_follower, traversability_map, waypoints, max_sim_steps, rng):
    rollout = []; position_history = []; waypoint_idx = 0
    _, rng = jax.random.split(rng)
    state = env.reset(rng)
    for i in range(max_sim_steps):
        _, rng = jax.random.split(rng)
        rng = sample_pert(rng, env, state)
        cell_center_offset = 0.5
        target_position = waypoints[waypoint_idx] + cell_center_offset
        pos_command = command_generator.compute_command(state, target_position)
        state.info["command"] = pos_command.value
        ctrl = command_follower.control(state.obs["state"])
        state = env.step(state, ctrl)
        rollout.append(state)
        position_history = update_position_history(position_history, pos_command.info.pos[:2])
        if not traversability_map.is_valid_position(pos_command.info.pos[:2]):
            return EpisodeResult(status="Stop", position_history=position_history, rollout=rollout)
        if pos_command.info.is_arrived:
            waypoint_idx = min(waypoint_idx + 1, len(waypoints))
            if waypoint_idx == len(waypoints):
                return EpisodeResult(status="Success", position_history=position_history, rollout=rollout)
    return EpisodeResult(status="Timeout", position_history=position_history, rollout=rollout)

factory = ControllerFactory()
mlp_params = MLPPolicyParams.from_dict(system_config["control"]["locomotion_controller"])
command_follower = factory.build(params=mlp_params)
pc_config = PositionControllerParams.from_dict(system_config["control"]["position_controller"])
command_generator = PositionController(factory=factory, config=pc_config)
instruction = open("examples/prompt.txt").read()
mission_config = MissionConfig.from_dict(system_config["mission"])
mission_config.max_attempts = 5
mission_config.retry_delay_sec = 2
mission_executer = MissionExecuter(config=mission_config, instruction_prompt=instruction)
run_one_episode_func = partial(run_one_episode, env=env, command_generator=command_generator, command_follower=command_follower, traversability_map=trav_map)
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
planner = GeminiRoboticsER2Navigator(client)
print("MISSION_START max_attempts=5 goal=(4,4)", flush=True)
result = mission_executer.execute_mission(planner=planner, execute_single_attempt=run_one_episode_func, rng=rng)
print("MISSION_RESULT status:", result.status, flush=True)
print("MISSION_WAYPOINTS:", [tuple(float(x) for x in wp) for wp in result.waypoints], flush=True)
print("MISSION_POSITION_HISTORY:", [tuple(int(x) for x in p) for p in result.position_history], flush=True)
print("MISSION_ROLLOUT_LEN:", len(result.rollouts), flush=True)

print("RENDER_START", flush=True)
try:
    import mujoco as _mj
    def modify_scn_fn(scene):
        scene.lights[0].ambient = [0.0, 0.0, 0.0]
        scene.lights[0].diffuse = [0.1, 0.1, 0.1]
        scene.lights[0].specular = [0.0, 0.0, 0.0]
        scene.lights[1].ambient = [0.0, 0.0, 0.0]
        scene.lights[1].exponent = 50.0
        scene.lights[1].cutoff = 65.0
        scene.lights[1].diffuse = [1.0, 1.0, 1.0]
        scene.lights[1].specular = [0.0, 0.0, 0.0]
    camera = _mj.MjvCamera()
    _mj.mjv_defaultCamera(camera)
    camera.lookat[:] = np.array([2.0, 1.5, 1.5])
    camera.distance = (6 / 2) / np.tan(np.radians(45 / 2)) * 1
    camera.azimuth = 90
    camera.elevation = -70
    scene_option = _mj.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[_mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[_mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    scene_option.flags[_mj.mjtVisFlag.mjVIS_TRANSPARENT] = False
    render_every = 10
    fps = 3.0 / env.dt / render_every
    traj = result.rollouts[::render_every]
    print(f"RENDER_FRAMES traj={len(traj)} fps={fps}", flush=True)
    frames = env.render(traj, camera=camera, scene_option=scene_option, height=480, width=640, modify_scene_fns=[modify_scn_fn] * len(traj))
    print(f"RENDER_DONE frames={len(frames)} shape={frames[0].shape}", flush=True)
    import subprocess
    proc = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "/workspace/er2_nav_full.mp4"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close(); proc.wait()
    print("VIDEO_DONE rc=", proc.returncode, flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print("RENDER_FAILED", type(e).__name__, str(e)[:500], flush=True)
