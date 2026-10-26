import os, sys, json, math
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "/workspace/llm_mjx_go1_playground")
os.chdir("/workspace/llm_mjx_go1_playground")
import jax, numpy as np
import mujoco
from mujoco_playground._src.locomotion.go1.base import get_assets
from src.environment.custom_go1_env import CustomGo1JoystickEnv
from src.environment.env_wrapper import Go1Env
from src.environment.traversability_map import TraversabilityMap, TraversabilityMapConfig
from src.control.controller_factory import ControllerFactory
from src.control.position_controller import PositionController, PositionControllerParams
from src.control.algorithms.mlp import MLPPolicy, MLPPolicyParams
from src.utils import load_yaml_with_jinja

W, H = 16, 10
phase1 = json.load(open("/workspace/unknown_map/phase1.json"))
visited = set(tuple(v) for v in phase1["visited"])
built = phase1["built_grid_xy"]  # built[x][y]
known = np.array(built, dtype=int)  # shape (W,H) ? json preserves list order: built[x][y]
print("SWEEP known shape", known.shape, "visited", len(visited), flush=True)
TRUE_GAUGES = {"G1": ((1,8), 2.5), "G2": ((6,0), 5.8), "G3": ((14,7), 8.1), "G4": ((2,1), 3.7), "G5": ((10,5), 6.4), "G6": ((13,2), 4.2)}
# true grid for collision check only (not planning)
sys.path.insert(0, "/workspace/maze2")
from maze_grid import grid as TRUE_GRID
scene_xml = open("/workspace/maze2/maze_scene.xml").read()
lm = mujoco.MjModel.from_xml_string(scene_xml, assets=get_assets()); ld = mujoco.MjData(lm); mujoco.mj_forward(lm, ld)
_renderer = None
def visual_read(gx, gy):
    global _renderer
    try:
        if _renderer is None:
            from mujoco import Renderer
            _renderer = Renderer(lm, height=480, width=640)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = np.array([gx+0.5, gy+0.5, 0.0]); cam.distance = 1.6; cam.azimuth = 90; cam.elevation = -80
        _renderer.update_scene(ld, camera=cam)
        img = _renderer.render()
        red = (img[:,:,0].astype(int)>150) & (img[:,:,1].astype(int)<100) & (img[:,:,2].astype(int)<100)
        ys, xs = np.where(red)
        if len(xs)==0: return None
        ox, oy = float(xs.mean()-320), float(ys.mean()-240)
        ang = math.degrees(math.atan2(ox, -oy))
        return {"angle_deg": round(ang,1), "est_bar": round(float((ang+120)/240*10),2), "red_pixels": int(red.sum())}
    except Exception as e:
        print("VISUAL_READ_FAILED", e, flush=True); return None

def bfs_path(start, goal):
    from collections import deque
    if start==goal: return [start]
    q=deque([start]); prev={start:None}
    while q:
        cur=q.popleft()
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt=(cur[0]+dx, cur[1]+dy)
            if 0<=nxt[0]<W and 0<=nxt[1]<H and known[nxt]==1 and nxt not in prev:
                prev[nxt]=cur; q.append(nxt)
                if nxt==goal:
                    path=[]; c=goal
                    while c is not None: path.append(c); c=prev[c]
                    return path[::-1]
    return None
def bfs_dist(start):
    from collections import deque
    dist={start:0}; q=deque([start])
    while q:
        cur=q.popleft()
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt=(cur[0]+dx, cur[1]+dy)
            if 0<=nxt[0]<W and 0<=nxt[1]<H and known[nxt]==1 and nxt not in dist:
                dist[nxt]=dist[cur]+1; q.append(nxt)
    return dist

system_config = load_yaml_with_jinja("examples/configs/llm_navigation.yaml")
rng = jax.random.PRNGKey(0)
mpenv = CustomGo1JoystickEnv(xml_path="/workspace/maze2/maze_scene.xml")
env = Go1Env(env=mpenv, env_cfg=mpenv.env_cfg)
trav_map = TraversabilityMap(TraversabilityMapConfig(grid_size=(16, 10), image_size=160, threshold=125))
trav_map.load_from_grid(TRUE_GRID)
_, rng = jax.random.split(rng); state = env.reset(rng)
factory = ControllerFactory(); mlp_params = MLPPolicyParams.from_dict(system_config["control"]["locomotion_controller"]); command_follower = factory.build(params=mlp_params)
pc_config = PositionControllerParams.from_dict(system_config["control"]["position_controller"]); command_generator = PositionController(factory=factory, config=pc_config)

# found from phase 1 (already discovered, with visual estimates from phase 1 log)
found = {"G5": (6.4, {"angle_deg": 31.4, "est_bar": 6.31}, 1166, (10,5)),
         "G6": (4.2, {"angle_deg": -18.6, "est_bar": 4.22}, 3592, (13,2)),
         "G3": (8.1, {"angle_deg": 70.9, "est_bar": 7.95}, 4932, (14,7))}
current = (0,0)
position_history = [current]
rollout = []
total_steps = 0
max_steps = 12000
status = "Timeout"
def move_to_cell(target_cell, max_cell_steps=400):
    global total_steps, state, rng, status
    steps=0
    while steps < max_cell_steps and total_steps < max_steps:
        _, rng = jax.random.split(rng)
        state.info["pert_mag"]=jax.numpy.array(0.0); state.info["pert_duration"]=jax.numpy.array(0); state.info["pert_duration_seconds"]=jax.numpy.array(0.0)
        target = np.array(target_cell, dtype=float) + 0.5
        pos_command = command_generator.compute_command(state, target); state.info["command"]=pos_command.value
        ctrl = command_follower.control(state.obs["state"]); state = env.step(state, ctrl); rollout.append(state); total_steps+=1; steps+=1
        cell = tuple(int(x) for x in pos_command.info.pos[:2].astype(int))
        if position_history[-1]!=cell: position_history.append(cell)
        for name, ((gx,gy), val) in TRUE_GAUGES.items():
            if cell==(gx,gy) and name not in found:
                vr = visual_read(gx, gy)
                found[name]=(val, vr, total_steps, cell)
                print(f"SWEEP_GAUGE_FOUND {name} true={val}bar visual={vr} step={total_steps} cell={cell}", flush=True)
        if not trav_map.is_valid_position(pos_command.info.pos[:2]):
            print(f"SWEEP_STOP wall_cell={cell} step={total_steps}", flush=True); status="Stop"; return False
        if pos_command.info.is_arrived:
            return True
    print(f"SWEEP_CELL_TIMEOUT target={target_cell}", flush=True); return False

while len(found) < len(TRUE_GAUGES) and total_steps < max_steps:
    dist = bfs_dist(current)
    unvisited = [c for c in dist if known[c]==1 and c not in visited]
    if not unvisited:
        print("SWEEP_NO_UNVISITED", flush=True); status="NoUnvisited"; break
    # nearest unvisited
    target = min(unvisited, key=lambda c: dist[c])
    path = bfs_path(current, target)
    if path is None:
        print(f"SWEEP_NO_PATH to {target}", flush=True); break
    print(f"SWEEP_MOVE target={target} dist={dist[target]} path_len={len(path)} found={list(found.keys())}", flush=True)
    ok=True
    for cell in path[1:]:
        if not move_to_cell(cell):
            ok=False; break
        current=cell; visited.add(cell)
    if not ok: break
    if len(found) >= len(TRUE_GAUGES):
        status="Success"; print(f"SWEEP_SUCCESS total_steps={total_steps}", flush=True); break

print("SWEEP_RESULT status:", status, flush=True)
print("SWEEP_POSITION_HISTORY:", position_history, flush=True)
print("SWEEP_FOUND:", found, flush=True)
print("SWEEP_ROLLOUT_LEN:", len(rollout), flush=True)
open("/workspace/unknown_map/SWEEP_DONE","w").write(status)
print("SWEEP_RENDER_START", flush=True)
try:
    def modify_scn_fn(scene):
        scene.lights[0].ambient=[0,0,0]; scene.lights[0].diffuse=[0.1,0.1,0.1]; scene.lights[0].specular=[0,0,0]
        scene.lights[1].ambient=[0,0,0]; scene.lights[1].exponent=50.0; scene.lights[1].cutoff=65.0; scene.lights[1].diffuse=[1,1,1]; scene.lights[1].specular=[0,0,0]
    camera=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera); camera.lookat[:]=np.array([8.0,5.0,1.5]); camera.distance=(16/2)/np.tan(np.radians(45/2)); camera.azimuth=90; camera.elevation=-70
    scene_option=mujoco.MjvOption(); scene_option.geomgroup[2]=True; scene_option.geomgroup[3]=False; scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=True; scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=False; scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]=False
    render_every=20; fps=3.0/env.dt/render_every; traj=rollout[::render_every]; print(f"SWEEP_RENDER_FRAMES traj={len(traj)} fps={fps}", flush=True)
    frames=env.render(traj, camera=camera, scene_option=scene_option, height=480, width=640, modify_scene_fns=[modify_scn_fn]*len(traj)); print(f"SWEEP_RENDER_DONE frames={len(frames)}", flush=True)
    import subprocess; proc=subprocess.Popen(["ffmpeg","-y","-f","rawvideo","-pix_fmt","rgb24","-s","640x480","-r",str(fps),"-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p","/workspace/unknown_map/sweep_full.mp4"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for fr in frames: proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close(); proc.wait(); print("SWEEP_VIDEO_DONE rc=", proc.returncode, flush=True)
except Exception as e:
    import traceback; traceback.print_exc(); print("SWEEP_RENDER_FAILED", type(e).__name__, str(e)[:500], flush=True)
