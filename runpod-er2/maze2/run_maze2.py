import os, sys, json, math
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "/workspace/llm_mjx_go1_playground")
sys.path.insert(0, "/workspace/maze2")
os.chdir("/workspace/llm_mjx_go1_playground")
import jax, numpy as np
import mujoco
from mujoco_playground._src.locomotion.go1.base import get_assets
from src.environment.generate_maze_scene_xml import BASE_ENV_XML, BOUNDARY_AND_GOAL_XML, generate_wall_body_geom
from src.environment.custom_go1_env import CustomGo1JoystickEnv
from src.environment.env_wrapper import Go1Env
from src.environment.traversability_map import TraversabilityMap, TraversabilityMapConfig
from src.control.controller_factory import ControllerFactory
from src.control.position_controller import PositionController, PositionControllerParams
from src.control.algorithms.mlp import MLPPolicy, MLPPolicyParams
from src.utils import load_yaml_with_jinja
from maze_grid import grid, GAUGES, DOORS, START

def quat_z(a):
    return f"{math.cos(a/2):.6f} 0 0 {math.sin(a/2):.6f}"
def needle_angle_deg(pressure):
    return -120.0 + pressure/10.0*240.0
walls_xml = ""; wid = 0
for x in range(grid.shape[0]):
    for y in range(grid.shape[1]):
        if grid[x, y] == 0:
            wid += 1; walls_xml += generate_wall_body_geom(x, y, wid)
gauge_xml = ""
for name, ((gx, gy), val) in GAUGES.items():
    cx, cy = gx+0.5, gy+0.5
    phi = math.radians(needle_angle_deg(val))
    dx, dy = math.sin(phi), math.cos(phi)
    # dial disc
    gauge_xml += f"""<body name="dial_{name}" pos="{cx} {cy} 0.02"><geom name="dial_geom_{name}" type="cylinder" size="0.35 0.02" rgba="1 1 1 1" contype="0" conaffinity="0"/></body>\n"""
    # ticks at -120,-60,0,60,120
    for tdeg in [-120,-60,0,60,120]:
        tphi = math.radians(tdeg); tdx, tdy = math.sin(tphi), math.cos(tphi)
        gauge_xml += f"""<body name="tick_{name}_{tdeg}" pos="{cx+0.29*tdx:.3f} {cy+0.29*tdy:.3f} 0.045"><geom name="tick_geom_{name}_{tdeg}" type="box" size="0.02 0.06 0.01" rgba="0 0 0 1" quat="{quat_z(-tphi)}" contype="0" conaffinity="0"/></body>\n"""
    # needle, red, tip at 0.34 from centre
    gauge_xml += f"""<body name="needle_{name}" pos="{cx+0.12*dx:.3f} {cy+0.12*dy:.3f} 0.05"><geom name="needle_geom_{name}" type="box" size="0.03 0.22 0.01" rgba="1 0 0 1" quat="{quat_z(-phi)}" contype="0" conaffinity="0"/></body>\n"""
    # hub
    gauge_xml += f"""<body name="hub_{name}" pos="{cx} {cy} 0.055"><geom name="hub_geom_{name}" type="cylinder" size="0.06 0.015" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/></body>\n"""
    print(f"GAUGE_VISUAL {name} pos=({gx},{gy}) pressure={val} needle_deg_from_north={needle_angle_deg(val):.1f}", flush=True)
boundary_xml = """
    <body name="b_south" pos="8 -0.5 0.5"><geom name="b_south_g" type="box" size="8.5 0.5 0.5" material="box_material" rgba="0.0 0.72 0.53 1.0" contype="0" conaffinity="0"/></body>
    <body name="b_north" pos="8 10.5 0.5"><geom name="b_north_g" type="box" size="8.5 0.5 0.5" material="box_material" rgba="0.0 0.72 0.53 1.0" contype="0" conaffinity="0"/></body>
    <body name="b_west" pos="-0.5 5 0.5"><geom name="b_west_g" type="box" size="0.5 5.5 0.5" material="box_material" rgba="0.0 0.72 0.53 1.0" contype="0" conaffinity="0"/></body>
    <body name="b_east" pos="16.5 5 0.5"><geom name="b_east_g" type="box" size="0.5 5.5 0.5" material="box_material" rgba="0.0 0.72 0.53 1.0" contype="0" conaffinity="0"/></body>
"""
keyframe = BOUNDARY_AND_GOAL_XML[BOUNDARY_AND_GOAL_XML.index("<keyframe>"):]
scene_xml = BASE_ENV_XML + walls_xml + gauge_xml + boundary_xml + "</worldbody>\n\n" + keyframe
open("/workspace/maze2/maze_scene.xml", "w").write(scene_xml)
print("SCENE_WRITTEN walls=", wid, "gauges=", len(GAUGES), flush=True)
# Lidar CPU model (geometry only, no grid map)
try:
    lm = mujoco.MjModel.from_xml_string(scene_xml, assets=get_assets()); ld = mujoco.MjData(lm); mujoco.mj_forward(lm, ld); lidar_ok = True
    print("LIDAR_MODEL_OK ngeom=", lm.ngeom, flush=True)
except Exception as e:
    lidar_ok = False; print("LIDAR_MODEL_FAILED", e, flush=True)
def lidar_scan(cx, cy):
    if not lidar_ok: return None
    dists = []
    for k in range(36):
        ang = math.radians(k*10); vec = np.array([math.cos(ang), math.sin(ang), 0.0])
        gid = np.zeros((1,1), dtype=np.int32); gg = np.ones((6,1), dtype=np.uint8)
        dval = mujoco.mj_ray(lm, ld, np.array([cx, cy, 0.5]).reshape(3,1), vec.reshape(3,1), gg, 1, -1, gid)
        dists.append(float(dval) if dval > 0 else 6.0)
    return dists

system_config = load_yaml_with_jinja("examples/configs/llm_navigation.yaml")
rng = jax.random.PRNGKey(0)
mpenv = CustomGo1JoystickEnv(xml_path="/workspace/maze2/maze_scene.xml")
env = Go1Env(env=mpenv, env_cfg=mpenv.env_cfg)
trav_map = TraversabilityMap(TraversabilityMapConfig(grid_size=(16, 10), image_size=160, threshold=125))
trav_map.load_from_grid(grid)
waypoints = [np.array(w, dtype=float) for w in json.load(open("/workspace/maze2/waypoints.json"))]
print("WAYPOINTS", len(waypoints), flush=True)
def update_ph(ph, cur):
    if len(ph)==0: ph.append(cur.astype(np.int32)); return ph
    if np.array_equal(ph[-1].astype(np.int32), cur.astype(np.int32)): return ph
    ph.append(cur.astype(np.int32)); return ph
rollout=[]; position_history=[]; waypoint_idx=0; gauge_visited=[]; doors_crossed=[]; wall_hit=None; lidar_logs=[]
_, rng = jax.random.split(rng); state = env.reset(rng)
factory = ControllerFactory(); mlp_params = MLPPolicyParams.from_dict(system_config["control"]["locomotion_controller"]); command_follower = factory.build(params=mlp_params)
pc_config = PositionControllerParams.from_dict(system_config["control"]["position_controller"]); command_generator = PositionController(factory=factory, config=pc_config)
max_steps=10000; status="Timeout"
for i in range(max_steps):
    _, rng = jax.random.split(rng)
    state.info["pert_mag"]=jax.numpy.array(0.0); state.info["pert_duration"]=jax.numpy.array(0); state.info["pert_duration_seconds"]=jax.numpy.array(0.0)
    target_position = waypoints[waypoint_idx] + 0.5
    pos_command = command_generator.compute_command(state, target_position); state.info["command"]=pos_command.value
    ctrl = command_follower.control(state.obs["state"]); state = env.step(state, ctrl); rollout.append(state)
    position_history = update_ph(position_history, pos_command.info.pos[:2])
    cell = tuple(int(x) for x in pos_command.info.pos[:2].astype(int))
    for name, ((gx, gy), val) in GAUGES.items():
        if cell==(gx,gy) and name not in [g[0] for g in gauge_visited]:
            gauge_visited.append((name, val, i)); print(f"GAUGE {name} {val}bar step={i} cell={cell} needle_deg={needle_angle_deg(val):.1f}", flush=True)
    if cell in DOORS and cell not in doors_crossed:
        doors_crossed.append(cell); print(f"DOOR {cell} step={i}", flush=True)
    if not trav_map.is_valid_position(pos_command.info.pos[:2]):
        status="Stop"; wall_hit=cell; print(f"STOP wall_cell={cell} step={i}", flush=True); break
    if pos_command.info.is_arrived:
        # lidar scan at arrival cell, travel direction to next waypoint
        cur_wp = waypoints[waypoint_idx]
        nxt_wp = waypoints[waypoint_idx+1] if waypoint_idx+1 < len(waypoints) else cur_wp
        dvec = nxt_wp - cur_wp; travel_deg = math.degrees(math.atan2(dvec[1], dvec[0])) if np.linalg.norm(dvec)>0 else 0.0
        scan = lidar_scan(float(cur_wp[0]+0.5), float(cur_wp[1]+0.5))
        if scan is not None:
            # front ray index closest to travel direction
            front_idx = int(round(travel_deg/10.0)) % 36
            lidar_logs.append((tuple(int(x) for x in cur_wp), travel_deg, scan))
            print(f"LIDAR cell={tuple(int(x) for x in cur_wp)} travel_deg={travel_deg:.0f} front_dist={scan[front_idx]:.2f} min_dist={min(scan):.2f} scan={[round(x,2) for x in scan]}", flush=True)
        waypoint_idx = min(waypoint_idx+1, len(waypoints))
        if waypoint_idx==len(waypoints):
            status="Success"; print(f"SUCCESS step={i}", flush=True); break
print("MAZE2_RESULT status:", status, flush=True)
print("MAZE2_POSITION_HISTORY:", [tuple(int(x) for x in p) for p in position_history], flush=True)
print("MAZE2_GAUGES_VISITED:", gauge_visited, flush=True)
print("MAZE2_DOORS_CROSSED:", doors_crossed, flush=True)
print("MAZE2_WALL_HIT:", wall_hit, flush=True)
print("MAZE2_ROLLOUT_LEN:", len(rollout), flush=True)
print("RENDER_START", flush=True)
try:
    def modify_scn_fn(scene):
        scene.lights[0].ambient=[0,0,0]; scene.lights[0].diffuse=[0.1,0.1,0.1]; scene.lights[0].specular=[0,0,0]
        scene.lights[1].ambient=[0,0,0]; scene.lights[1].exponent=50.0; scene.lights[1].cutoff=65.0; scene.lights[1].diffuse=[1,1,1]; scene.lights[1].specular=[0,0,0]
    camera=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera); camera.lookat[:]=np.array([8.0,5.0,1.5]); camera.distance=(16/2)/np.tan(np.radians(45/2)); camera.azimuth=90; camera.elevation=-70
    scene_option=mujoco.MjvOption(); scene_option.geomgroup[2]=True; scene_option.geomgroup[3]=False; scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=True; scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=False; scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]=False
    render_every=20; fps=3.0/env.dt/render_every; traj=rollout[::render_every]; print(f"RENDER_FRAMES traj={len(traj)} fps={fps}", flush=True)
    frames=env.render(traj, camera=camera, scene_option=scene_option, height=480, width=640, modify_scene_fns=[modify_scn_fn]*len(traj)); print(f"RENDER_DONE frames={len(frames)}", flush=True)
    import subprocess; proc=subprocess.Popen(["ffmpeg","-y","-f","rawvideo","-pix_fmt","rgb24","-s","640x480","-r",str(fps),"-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p","/workspace/maze2/maze2_full.mp4"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for fr in frames: proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close(); proc.wait(); print("VIDEO_DONE rc=", proc.returncode, flush=True)
except Exception as e:
    import traceback; traceback.print_exc(); print("RENDER_FAILED", type(e).__name__, str(e)[:500], flush=True)
