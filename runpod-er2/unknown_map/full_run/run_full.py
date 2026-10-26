import os, sys, json, math, time
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "/workspace/llm_mjx_go1_playground")
os.chdir("/workspace/llm_mjx_go1_playground")
import jax, numpy as np
import mujoco
from mujoco_playground._src.locomotion.go1.base import get_assets
from src.environment.generate_maze_scene_xml import BASE_ENV_XML
from src.environment.custom_go1_env import CustomGo1JoystickEnv
from src.environment.env_wrapper import Go1Env
from src.environment.traversability_map import TraversabilityMap, TraversabilityMapConfig
from src.control.controller_factory import ControllerFactory
from src.control.position_controller import PositionController, PositionControllerParams
from src.control.algorithms.mlp import MLPPolicy, MLPPolicyParams
from src.utils import load_yaml_with_jinja

W, H = 16, 10
START = (0,0)
# TRUE scene info is used ONLY for: scene file, collision-stop check, and gauge detection (camera sees dial in same cell).
# Planner (known map, frontier, BFS) NEVER reads this true grid. It only reads lidar scans.
# Gauge true positions/values are for detection + comparison only; visual reading below estimates value from image.
TRUE_GAUGES = {"G1": ((1,8), 2.5), "G2": ((6,0), 5.8), "G3": ((14,7), 8.1), "G4": ((2,1), 3.7), "G5": ((10,5), 6.4), "G6": ((13,2), 4.2)}
TRUE_GRID = None
# Load true grid only for collision check, via maze_grid file if present (not used for planning)
try:
    sys.path.insert(0, "/workspace/maze2")
    from maze_grid import grid as TRUE_GRID
except Exception as e:
    print("TRUE_GRID_LOAD_FAILED", e, flush=True); TRUE_GRID = np.ones((W,H), dtype=int)

scene_xml = open("/workspace/maze2/maze_scene.xml").read()
print("SCENE_REUSED maze2 16x10 with analog dials", flush=True)
# Lidar CPU model
lm = mujoco.MjModel.from_xml_string(scene_xml, assets=get_assets()); ld = mujoco.MjData(lm); mujoco.mj_forward(lm, ld)
geom_wall_cell = {}
for gid in range(lm.ngeom):
    nm = lm.geom(gid).name
    if nm.startswith("static_box") or nm.startswith("b_"):
        bid = lm.geom_bodyid[gid]; bx, by = lm.body_pos[bid][0], lm.body_pos[bid][1]
        geom_wall_cell[gid] = (int(math.floor(bx)), int(math.floor(by)))
print("LIDAR_MODEL_OK wall_geoms=", len(geom_wall_cell), flush=True)
def lidar_scan(cx, cy, nrays=72):
    rays=[]
    for k in range(nrays):
        ang=math.radians(k*360.0/nrays); vec=np.array([math.cos(ang), math.sin(ang), 0.0])
        gid=np.zeros((1,1),dtype=np.int32); gg=np.ones((6,1),dtype=np.uint8)
        dv=mujoco.mj_ray(lm, ld, np.array([cx,cy,0.5]).reshape(3,1), vec.reshape(3,1), gg, 1, -1, gid)
        rays.append((float(dv) if dv>0 else 10.0, int(gid[0,0])))
    return rays
def update_known(known, cx, cy, rays, nrays=72):
    for k,(dist,gid) in enumerate(rays):
        ang=math.radians(k*360.0/nrays); dx,dy=math.cos(ang),math.sin(ang)
        t=0.0
        while t < dist-0.05:
            ix,iy=int(math.floor(cx+t*dx)), int(math.floor(cy+t*dy))
            if 0<=ix<W and 0<=iy<H and known[ix,iy]==-1: known[ix,iy]=1
            t+=0.1
        if gid in geom_wall_cell:
            wx,wy=geom_wall_cell[gid]
            if 0<=wx<W and 0<=wy<H and known[wx,wy]==-1: known[wx,wy]=0
    return known
# Visual gauge reading from close-up render (same method as maze2 close-ups)
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
        est = (ang+120)/240*10
        return {"angle_deg": round(ang,1), "est_bar": round(float(est),2), "red_pixels": int(red.sum())}
    except Exception as e:
        print("VISUAL_READ_FAILED", e, flush=True); return None

def bfs_path(known, start, goal):
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
def bfs_distances(known, start):
    from collections import deque
    dist={start:0}; q=deque([start])
    while q:
        cur=q.popleft()
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nxt=(cur[0]+dx, cur[1]+dy)
            if 0<=nxt[0]<W and 0<=nxt[1]<H and known[nxt]==1 and nxt not in dist:
                dist[nxt]=dist[cur]+1; q.append(nxt)
    return dist
def frontiers(known, current):
    dist=bfs_distances(known, current)
    cands=[]
    for (x,y),d in dist.items():
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=x+dx,y+dy
            if 0<=nx<W and 0<=ny<H and known[nx,ny]==-1:
                cands.append(((x,y),d)); break
    cands.sort(key=lambda t: t[1])
    return cands
def partial_map_text(known, current, found):
    gmap={}
    for name in found: 
        for gn,((gx,gy),_) in TRUE_GAUGES.items():
            if gn==name: gmap[(gx,gy)]=gn[1]
    gmap[current]="R"
    rows=[]
    for y in range(H-1,-1,-1):
        row=""
        for x in range(W):
            if (x,y) in gmap: row+=gmap[(x,y)]
            elif known[x,y]==-1: row+="?"
            elif known[x,y]==0: row+="#"
            else: row+="."
        rows.append(f"y={y} {row}")
    return "\n".join(rows)

# Env setup
system_config = load_yaml_with_jinja("examples/configs/llm_navigation.yaml")
rng = jax.random.PRNGKey(0)
mpenv = CustomGo1JoystickEnv(xml_path="/workspace/maze2/maze_scene.xml")
env = Go1Env(env=mpenv, env_cfg=mpenv.env_cfg)
trav_map = TraversabilityMap(TraversabilityMapConfig(grid_size=(16, 10), image_size=160, threshold=125))
trav_map.load_from_grid(TRUE_GRID)
_, rng = jax.random.split(rng); state = env.reset(rng)
factory = ControllerFactory(); mlp_params = MLPPolicyParams.from_dict(system_config["control"]["locomotion_controller"]); command_follower = factory.build(params=mlp_params)
pc_config = PositionControllerParams.from_dict(system_config["control"]["position_controller"]); command_generator = PositionController(factory=factory, config=pc_config)

known = np.full((W,H), -1, dtype=int); known[START]=1
current = START
found = {}  # name -> (true_val, est, step, cell)
position_history = [START]
rollout = []
total_steps = 0
max_steps = 20000
er2_calls = 0; er2_max_calls = 25
decision_id = 0
status = "Timeout"
# initial scan
rays = lidar_scan(START[0]+0.5, START[1]+0.5); update_known(known, START[0]+0.5, START[1]+0.5, rays)
print(f"INITIAL_SCAN known_cells={int((known!=-1).sum())}", flush=True)

def move_to_cell(target_cell, max_cell_steps=400):
    global total_steps, state, rng, status
    # returns True if arrived, False if timeout/stop
    steps=0
    while steps < max_cell_steps and total_steps < max_steps:
        _, rng = jax.random.split(rng)
        state.info["pert_mag"]=jax.numpy.array(0.0); state.info["pert_duration"]=jax.numpy.array(0); state.info["pert_duration_seconds"]=jax.numpy.array(0.0)
        target = np.array(target_cell, dtype=float) + 0.5
        pos_command = command_generator.compute_command(state, target); state.info["command"]=pos_command.value
        ctrl = command_follower.control(state.obs["state"]); state = env.step(state, ctrl); rollout.append(state); total_steps+=1; steps+=1
        cell = tuple(int(x) for x in pos_command.info.pos[:2].astype(int))
        if len(position_history)==0 or position_history[-1]!=cell: position_history.append(cell)
        # gauge detection (camera sees dial in same cell)
        for name, ((gx,gy), val) in TRUE_GAUGES.items():
            if cell==(gx,gy) and name not in found:
                vr = visual_read(gx, gy)
                found[name]=(val, vr, total_steps, cell)
                print(f"GAUGE_FOUND {name} true={val}bar visual={vr} step={total_steps} cell={cell}", flush=True)
        if not trav_map.is_valid_position(pos_command.info.pos[:2]):
            print(f"STOP wall_cell={cell} step={total_steps}", flush=True); status="Stop"; return False
        if pos_command.info.is_arrived:
            return True
    print(f"CELL_TIMEOUT target={target_cell} steps={steps}", flush=True); return False

while len(found) < len(TRUE_GAUGES) and total_steps < max_steps:
    cands = frontiers(known, current)
    if not cands:
        print("NO_FRONTIER known_cells=", int((known!=-1).sum()), "switching_to_sweep_in_same_run", flush=True)
        # SWEEP MODE, same run, no restart: visit nearest unvisited known-free cells
        visited = set(position_history)
        while len(found) < len(TRUE_GAUGES) and total_steps < max_steps:
            dist = bfs_distances(known, current)
            unvisited = [c for c in dist if known[c]==1 and c not in visited]
            if not unvisited:
                print("FULL_NO_UNVISITED", flush=True); status="NoUnvisited"; break
            target = min(unvisited, key=lambda c: dist[c])
            path = bfs_path(known, current, target)
            if path is None:
                print(f"FULL_SWEEP_NO_PATH to {target}", flush=True); status="NoPath"; break
            print(f"FULL_SWEEP_MOVE target={target} dist={dist[target]} path_len={len(path)} found={list(found.keys())}", flush=True)
            ok=True
            for cell in path[1:]:
                if not move_to_cell(cell):
                    ok=False; break
                current=cell; visited.add(cell)
            if not ok: break
            if len(found) >= len(TRUE_GAUGES):
                status="Success"; print(f"FULL_SUCCESS_ALL_GAUGES total_steps={total_steps}", flush=True); break
        break
    top = cands[:5]
    chosen = top[0][0]; source="bfs_nearest"
    if er2_calls < er2_max_calls:
        decision_id += 1; er2_calls += 1
        req = {"id": decision_id, "current": list(current), "candidates": [list(c[0]) for c in top], "candidate_dist": [c[1] for c in top], "found": list(found.keys()), "partial_map": partial_map_text(known, current, found)}
        # clean old answer, write request
        try: os.remove("/workspace/full_run/er2_answer.json")
        except: pass
        open("/workspace/full_run/er2_request.json","w").write(json.dumps(req))
        print(f"DECISION id={decision_id} current={current} candidates={top} found={list(found.keys())} waiting_er2", flush=True)
        # wait for answer, max 240s
        ans=None; waited=0
        while waited < 240:
            time.sleep(2); waited+=2
            try:
                a=json.load(open("/workspace/full_run/er2_answer.json"))
                if a.get("id")==decision_id: ans=a; break
            except: pass
        if ans is not None and tuple(ans.get("target",[])) in [c[0] for c in top]:
            chosen=tuple(ans["target"]); source=ans.get("source","er2")
            print(f"DECISION id={decision_id} er2_chose={chosen} raw={ans.get("raw","")[:200]}", flush=True)
        else:
            print(f"DECISION id={decision_id} er2_answer_invalid_or_timeout ans={ans} fallback_nearest={chosen}", flush=True)
    else:
        print(f"DECISION bfs_only current={current} candidates={top} er2_calls_capped", flush=True)
    path = bfs_path(known, current, chosen)
    if path is None:
        print(f"NO_PATH to {chosen} from {current}", flush=True); break
    print(f"MOVE path_len={len(path)} path={path} source={source}", flush=True)
    ok=True
    for cell in path[1:]:
        if not move_to_cell(cell):
            ok=False; break
        current=cell
    if not ok:
        break
    # scan at new current
    rays = lidar_scan(current[0]+0.5, current[1]+0.5); update_known(known, current[0]+0.5, current[1]+0.5, rays)
    print(f"SCAN_AFTER_MOVE current={current} known_cells={int((known!=-1).sum())} found={list(found.keys())} total_steps={total_steps}", flush=True)
    # also check gauge at current (in case path ended on gauge, move_to already checked, but scan again safe)
    if len(found) >= len(TRUE_GAUGES):
        status="Success"; print(f"SUCCESS_ALL_GAUGES total_steps={total_steps}", flush=True); break

print("FULL_RESULT status:", status, flush=True)
print("FULL_POSITION_HISTORY:", position_history, flush=True)
print("FULL_FOUND:", {k: (v[0], v[1], v[2], v[3]) for k,v in found.items()}, flush=True)
print("FULL_KNOWN_CELLS:", int((known!=-1).sum()), flush=True)
# built map vs true (evaluation only)
if TRUE_GRID is not None:
    mask=known!=-1; wrong=[(int(x),int(y),int(known[x,y]),int(TRUE_GRID[x,y])) for x in range(W) for y in range(H) if mask[x,y] and known[x,y]!=TRUE_GRID[x,y]]
    print("FULL_BUILT_MAP_WRONG:", wrong, flush=True)
    print("FULL_BUILT_MAP_ROWS:", flush=True)
    for y in range(H-1,-1,-1):
        row=""
        for x in range(W):
            v=known[x,y]; row += "?" if v==-1 else ("." if v==1 else "#")
        print(f"  y={y} {row}", flush=True)
print("FULL_ROLLOUT_LEN:", len(rollout), flush=True)
open("/workspace/full_run/FULL_DONE","w").write(status)
print("RENDER_START", flush=True)
try:
    def modify_scn_fn(scene):
        scene.lights[0].ambient=[0,0,0]; scene.lights[0].diffuse=[0.1,0.1,0.1]; scene.lights[0].specular=[0,0,0]
        scene.lights[1].ambient=[0,0,0]; scene.lights[1].exponent=50.0; scene.lights[1].cutoff=65.0; scene.lights[1].diffuse=[1,1,1]; scene.lights[1].specular=[0,0,0]
    camera=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera); camera.lookat[:]=np.array([8.0,5.0,1.5]); camera.distance=(16/2)/np.tan(np.radians(45/2)); camera.azimuth=90; camera.elevation=-70
    scene_option=mujoco.MjvOption(); scene_option.geomgroup[2]=True; scene_option.geomgroup[3]=False; scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=True; scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=False; scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]=False
    render_every=20; fps=3.0/env.dt/render_every; traj=rollout[::render_every]; print(f"RENDER_FRAMES traj={len(traj)} fps={fps}", flush=True)
    frames=env.render(traj, camera=camera, scene_option=scene_option, height=480, width=640, modify_scene_fns=[modify_scn_fn]*len(traj)); print(f"RENDER_DONE frames={len(frames)}", flush=True)
    import subprocess; proc=subprocess.Popen(["ffmpeg","-y","-f","rawvideo","-pix_fmt","rgb24","-s","640x480","-r",str(fps),"-i","-","-an","-c:v","libx264","-pix_fmt","yuv420p","/workspace/full_run/full_run.mp4"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for fr in frames: proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close(); proc.wait(); print("VIDEO_DONE rc=", proc.returncode, flush=True)
except Exception as e:
    import traceback; traceback.print_exc(); print("RENDER_FAILED", type(e).__name__, str(e)[:500], flush=True)
