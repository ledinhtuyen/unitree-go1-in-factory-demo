import os, sys, json
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "/workspace/llm_mjx_go1_playground")
sys.path.insert(0, "/workspace/factory_er2")
os.chdir("/workspace/llm_mjx_go1_playground")
import jax, numpy as np
from functools import partial
import mujoco
from mujoco_playground._src import mjx_env
from typing import List
from src.environment.generate_maze_scene_xml import BASE_ENV_XML, BOUNDARY_AND_GOAL_XML, generate_wall_body_geom
from src.environment.custom_go1_env import CustomGo1JoystickEnv
from src.environment.env_wrapper import Go1Env
from src.environment.traversability_map import TraversabilityMap, TraversabilityMapConfig
from src.control.controller_factory import ControllerFactory
from src.control.position_controller import PositionController, PositionControllerParams
from src.control.algorithms.mlp import MLPPolicy, MLPPolicyParams
from src.mission_executer import EpisodeResult
from src.utils import load_yaml_with_jinja
from factory_grid import grid, GAUGES, DOORS, START

# Build factory XML
walls_xml = ""
wid = 0
for x in range(grid.shape[0]):
    for y in range(grid.shape[1]):
        if grid[x, y] == 0:
            wid += 1
            walls_xml += generate_wall_body_geom(x, y, wid)
gauge_colors = {"G1": "0.9 0.1 0.1 1.0", "G2": "0.1 0.4 0.9 1.0", "G3": "0.1 0.8 0.1 1.0", "G4": "0.9 0.7 0.0 1.0", "G5": "0.6 0.1 0.9 1.0"}
gauge_xml = ""
for name, ((gx, gy), val) in GAUGES.items():
    gauge_xml += f"""<body name="gauge_{name}" pos="{gx+0.5} {gy+0.5} 0.05"><geom name="gauge_geom_{name}" type="box" size="0.4 0.4 0.05" material="box_material" rgba="{gauge_colors[name]}" contype="0" conaffinity="0"/></body>\n"""
boundary_xml = """
    <body name="b_south" pos="6 -0.5 0.5"><geom name="b_south_g" type="box" size="6.5 0.5 0.5" material="box_material" rgba="0.0 0.72 0.53 1.0" contype="0" conaffinity="0"/></body>
    <body name="b_north" pos="6 8.5 0.5"><geom name="b_north_g" type="box" size="6.5 0.5 0.5" material="box_material" rgba="0.0 0.72 0.53 1.0" contype="0" conaffinity="0"/></body>
    <body name="b_west" pos="-0.5 4 0.5"><geom name="b_west_g" type="box" size="0.5 4.5 0.5" material="box_material" rgba="0.0 0.72 0.53 1.0" contype="0" conaffinity="0"/></body>
    <body name="b_east" pos="12.5 4 0.5"><geom name="b_east_g" type="box" size="0.5 4.5 0.5" material="box_material" rgba="0.0 0.72 0.53 1.0" contype="0" conaffinity="0"/></body>
"""
keyframe = BOUNDARY_AND_GOAL_XML[BOUNDARY_AND_GOAL_XML.index("<keyframe>"):]
scene_xml = BASE_ENV_XML + walls_xml + gauge_xml + boundary_xml + "</worldbody>\n\n" + keyframe
open("/workspace/factory_er2/factory_scene.xml", "w").write(scene_xml)
print("SCENE_WRITTEN walls=", wid, "gauges=", len(GAUGES), flush=True)

system_config = load_yaml_with_jinja("examples/configs/llm_navigation.yaml")
rng = jax.random.PRNGKey(0)
mpenv = CustomGo1JoystickEnv(xml_path="/workspace/factory_er2/factory_scene.xml")
env = Go1Env(env=mpenv, env_cfg=mpenv.env_cfg)
trav_map = TraversabilityMap(TraversabilityMapConfig(grid_size=(12, 8), image_size=120, threshold=125))
trav_map.load_from_grid(grid)
waypoints = [np.array(w, dtype=float) for w in json.load(open("/workspace/factory_er2/waypoints.json"))]
print("WAYPOINTS", len(waypoints), [tuple(float(x) for x in w) for w in waypoints], flush=True)

def update_position_history(ph, cur):
    if len(ph) == 0:
        ph.append(cur.astype(np.int32)); return ph
    if np.array_equal(ph[-1].astype(np.int32), cur.astype(np.int32)):
        return ph
    ph.append(cur.astype(np.int32)); return ph

rollout = []; position_history = []; waypoint_idx = 0
gauge_visited = []; doors_crossed = []; wall_hit = None
_, rng = jax.random.split(rng)
state = env.reset(rng)
factory = ControllerFactory()
mlp_params = MLPPolicyParams.from_dict(system_config["control"]["locomotion_controller"])
command_follower = factory.build(params=mlp_params)
pc_config = PositionControllerParams.from_dict(system_config["control"]["position_controller"])
command_generator = PositionController(factory=factory, config=pc_config)
max_steps = 6000
status = "Timeout"
for i in range(max_steps):
    _, rng = jax.random.split(rng)
    state.info["pert_mag"] = jax.numpy.array(0.0); state.info["pert_duration"] = jax.numpy.array(0); state.info["pert_duration_seconds"] = jax.numpy.array(0.0)
    target_position = waypoints[waypoint_idx] + 0.5
    pos_command = command_generator.compute_command(state, target_position)
    state.info["command"] = pos_command.value
    ctrl = command_follower.control(state.obs["state"])
    state = env.step(state, ctrl)
    rollout.append(state)
    position_history = update_position_history(position_history, pos_command.info.pos[:2])
    cell = tuple(int(x) for x in pos_command.info.pos[:2].astype(int))
    for name, ((gx, gy), val) in GAUGES.items():
        if cell == (gx, gy) and name not in [g[0] for g in gauge_visited]:
            gauge_visited.append((name, val, i)); print(f"GAUGE {name} {val}bar step={i} cell={cell}", flush=True)
    if cell in DOORS and cell not in doors_crossed:
        doors_crossed.append(cell); print(f"DOOR {cell} step={i}", flush=True)
    if not trav_map.is_valid_position(pos_command.info.pos[:2]):
        status = "Stop"; wall_hit = cell; print(f"STOP wall_cell={cell} step={i}", flush=True); break
    if pos_command.info.is_arrived:
        waypoint_idx = min(waypoint_idx + 1, len(waypoints))
        if waypoint_idx == len(waypoints):
            status = "Success"; print(f"SUCCESS step={i}", flush=True); break
print("FACTORY_RESULT status:", status, flush=True)
print("FACTORY_WAYPOINTS:", [tuple(float(x) for x in w) for w in waypoints], flush=True)
print("FACTORY_POSITION_HISTORY:", [tuple(int(x) for x in p) for p in position_history], flush=True)
print("FACTORY_GAUGES_VISITED:", gauge_visited, flush=True)
print("FACTORY_DOORS_CROSSED:", doors_crossed, flush=True)
print("FACTORY_WALL_HIT:", wall_hit, flush=True)
print("FACTORY_ROLLOUT_LEN:", len(rollout), flush=True)
print("RENDER_START", flush=True)
try:
    def modify_scn_fn(scene):
        scene.lights[0].ambient = [0.0, 0.0, 0.0]; scene.lights[0].diffuse = [0.1, 0.1, 0.1]; scene.lights[0].specular = [0.0, 0.0, 0.0]
        scene.lights[1].ambient = [0.0, 0.0, 0.0]; scene.lights[1].exponent = 50.0; scene.lights[1].cutoff = 65.0; scene.lights[1].diffuse = [1.0, 1.0, 1.0]; scene.lights[1].specular = [0.0, 0.0, 0.0]
    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = np.array([6.0, 4.0, 1.5]); camera.distance = (12 / 2) / np.tan(np.radians(45 / 2)) * 1; camera.azimuth = 90; camera.elevation = -70
    scene_option = mujoco.MjvOption(); scene_option.geomgroup[2] = True; scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True; scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False; scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    render_every = 20; fps = 3.0 / env.dt / render_every
    traj = rollout[::render_every]
    print(f"RENDER_FRAMES traj={len(traj)} fps={fps}", flush=True)
    frames = env.render(traj, camera=camera, scene_option=scene_option, height=480, width=640, modify_scene_fns=[modify_scn_fn] * len(traj))
    print(f"RENDER_DONE frames={len(frames)}", flush=True)
    import subprocess
    proc = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "/workspace/factory_er2/factory_er2_full.mp4"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for fr in frames: proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close(); proc.wait(); print("VIDEO_DONE rc=", proc.returncode, flush=True)
except Exception as e:
    import traceback; traceback.print_exc(); print("RENDER_FAILED", type(e).__name__, str(e)[:500], flush=True)
