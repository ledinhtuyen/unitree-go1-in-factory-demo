"""Render the SEQUENTIAL autonomous trajectory to video.

Usage:
  python render_nav_seq.py
Input: /home/ubuntu/factory/nav_trajectory_seq.npz,
       /home/ubuntu/factory/nav_summary_seq.json
Output: /home/ubuntu/factory/factory_nav_seq_demo.mp4 (640x480, 25fps).
Frames are the actual sequential trajectory (every 4th step), with the
current segment (pump1/2/3) labelled on each frame. After each segment's
last sampled frame, that pump's gauge close-up is inserted (50 frames,
2 s) -- i.e. gauge close-ups at segment boundaries in the video. If a
segment failed, its FAIL label is shown and no success is claimed.
Gauge images/labels use the previously verified obstacle-scene readings.
"""
import os, json
import numpy as np
import mujoco
import cv2

os.environ.setdefault("MUJOCO_GL", "egl")
NAV = "/home/ubuntu/factory/factory_obstacle.xml"
W, H, FPS = 640, 480, 25
TRAJ = "/home/ubuntu/factory/nav_trajectory_seq.npz"
SUMMARY = "/home/ubuntu/factory/nav_summary_seq.json"
OUT = "/home/ubuntu/factory/factory_nav_seq_demo.mp4"

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
boundaries = list(traj["segment_boundaries"]) if "segment_boundaries" in traj else []
print(f"traj frames={len(qpos)} boundaries={boundaries}")
with open(SUMMARY) as f:
    s = json.load(f)
segs = s["segments"]
overlay2 = (f"Sequential autonomous (no reset, no waypoint): "
            f"{s['pumps_reached']}/{s['pumps_total']} pumps, "
            f"{s['total_steps']} steps, success={s['success']}")
per = "; ".join(f"{r['pump']} {'OK' if r['reached'] else 'FAIL'} {r['final_dist']:.2f}m"
                for r in segs)
gauges = [
    ("/home/ubuntu/factory/gauge_obstacle_pump1.png", "Pump1 gauge: true 2.50 bar, read 2.5003 bar"),
    ("/home/ubuntu/factory/gauge_obstacle_pump2.png", "Pump2 gauge: true 5.80 bar, read 5.8010 bar"),
    ("/home/ubuntu/factory/gauge_obstacle_pump3.png", "Pump3 gauge: true 8.10 bar, read 8.1005 bar"),
]

def segment_of_frame(idx):
    # idx is a full trajectory frame index; segment = first segment whose
    # boundary is >= idx, else the last segment.
    for si, b in enumerate(boundaries):
        if idx <= b:
            return si
    return len(segs) - 1 if segs else 0

frames = []
sampled = list(range(0, len(qpos), 4))
for pos, idx in enumerate(sampled):
    data.qpos[:] = qpos[idx]
    data.qvel[:] = qvel[idx]
    data.ctrl[:] = ctrl[idx]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    si = segment_of_frame(idx)
    seg = segs[si] if si < len(segs) else segs[-1]
    cv2.putText(img, "Sequential autonomous - Go1 (one run, no reset, no waypoint)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"step {idx}  pos=({xy[idx][0]:.2f},{xy[idx][1]:.2f})  segment {si+1}: {seg['pump']}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, overlay2, (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 0), 2, cv2.LINE_AA)
    cv2.putText(img, per, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    frames.append(img)
    # Gauge close-up at a segment boundary: after the last sampled frame
    # of a segment (or the final frame), insert that pump's gauge.
    is_last_sampled_of_segment = False
    if si < len(boundaries):
        # next sampled frame belongs to a later segment (or this is the last frame)
        nxt = sampled[pos + 1] if pos + 1 < len(sampled) else None
        if nxt is None or segment_of_frame(nxt) != si:
            is_last_sampled_of_segment = True
    # Only insert a gauge close-up for a segment that was actually reached;
    # a failed segment's gauge was not read in this run, so showing its
    # verified value here would mislead.
    if is_last_sampled_of_segment and si < len(gauges) and segs[si]["reached"]:
        path, label = gauges[si]
        g = cv2.imread(path)
        if g is not None:
            g = cv2.resize(g, (W, H))
            cv2.putText(g, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            for _ in range(50):
                frames.append(g.copy())
        else:
            print(f"missing {path}")
    if idx % 200 == 0:
        print(f"rendered {idx}", flush=True)
print(f"total frames={len(frames)} duration={len(frames)/FPS:.1f}s")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))
for f in frames:
    writer.write(f)
writer.release()
print(f"WROTE {OUT} size={os.path.getsize(OUT)} bytes")
