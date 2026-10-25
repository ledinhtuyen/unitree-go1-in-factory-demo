"""Render an autonomous nav trajectory to video (no waypoint overlay tricks:
the trajectory is the policy's own run, including any failure).

Usage:
  python render_nav_auto.py --prefix auto --out factory_nav_auto_demo.mp4
Input: /home/ubuntu/factory/nav_trajectory_<prefix>.npz,
       /home/ubuntu/factory/nav_summary_<prefix>.json
Output: /home/ubuntu/factory/<out> (640x480, 25fps), gauge close-ups appended.
Gauge images/labels use the previously verified obstacle-scene readings.
"""
import argparse, os, json
import numpy as np
import mujoco
import cv2

os.environ.setdefault("MUJOCO_GL", "egl")
NAV = "/home/ubuntu/factory/factory_obstacle.xml"
W, H, FPS = 640, 480, 25

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="auto")
ap.add_argument("--out", default="factory_nav_auto_demo.mp4")
args = ap.parse_args()
TRAJ = f"/home/ubuntu/factory/nav_trajectory_{args.prefix}.npz"
SUMMARY = f"/home/ubuntu/factory/nav_summary_{args.prefix}.json"
OUT = f"/home/ubuntu/factory/{args.out}"

model = mujoco.MjModel.from_xml_path(NAV)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
cam.distance = 5.0
cam.azimuth = 180.0
cam.elevation = -20.0
cam.lookat = np.array([0.0, 0.0, 0.5])
traj = np.load(TRAJ)
qpos, qvel, ctrl, xy = traj["qpos"], traj["qvel"], traj["ctrl"], traj["xy"]
print(f"traj frames={len(qpos)}")
try:
    with open(SUMMARY) as f:
        s = json.load(f)
    mode = s.get("mode", "autonomous")
    overlay2 = f"Autonomous (no waypoint controller): {s['pumps_reached']}/{s['pumps_total']} pumps reached"
    per = "; ".join(f"{r['pump']} {'OK' if r['reached'] else 'FAIL'} {r['final_dist']:.2f}m"
                    for r in s["results"])
except Exception as e:
    overlay2 = f"Autonomous nav (summary unreadable: {e})"
    per = ""
frames = []
for idx in range(0, len(qpos), 4):
    data.qpos[:] = qpos[idx]
    data.qvel[:] = qvel[idx]
    data.ctrl[:] = ctrl[idx]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.putText(img, "Autonomous door-finding - Go1 (no waypoint controller)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"step {idx}  pos=({xy[idx][0]:.2f},{xy[idx][1]:.2f})", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, overlay2, (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 0), 2, cv2.LINE_AA)
    cv2.putText(img, per, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    frames.append(img)
    if idx % 200 == 0:
        print(f"rendered {idx}", flush=True)
gauges = [
    ("/home/ubuntu/factory/gauge_obstacle_pump1.png", "Pump1 gauge: true 2.50 bar, read 2.5003 bar"),
    ("/home/ubuntu/factory/gauge_obstacle_pump2.png", "Pump2 gauge: true 5.80 bar, read 5.8010 bar"),
    ("/home/ubuntu/factory/gauge_obstacle_pump3.png", "Pump3 gauge: true 8.10 bar, read 8.1005 bar"),
]
for path, label in gauges:
    g = cv2.imread(path)
    if g is None:
        print(f"missing {path}"); continue
    g = cv2.resize(g, (W, H))
    cv2.putText(g, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    for _ in range(50):
        frames.append(g.copy())
print(f"total frames={len(frames)} duration={len(frames)/FPS:.1f}s")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))
for f in frames:
    writer.write(f)
writer.release()
print(f"WROTE {OUT} size={os.path.getsize(OUT)} bytes")
