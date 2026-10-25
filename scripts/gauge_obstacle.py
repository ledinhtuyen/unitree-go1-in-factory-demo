"""Render gauges from the obstacle scene cameras and read them.

Reuses the read_gauge logic from gauge_test.py, but loads
factory_obstacle.xml (fixed needle geoms, same true values) and
its gauge1/2/3 cameras at y=2.2 (north of the north partition).
"""
import os, math, json, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco, cv2
sys.path.insert(0, "/home/ubuntu/factory")
# Copy read_gauge here to avoid importing gauge_test (which runs main on import? it guards main, so import is safe, but keep standalone)
def angle_to_value(deg):
    return (deg + 120.0) / 240.0 * 10.0

def read_gauge(img_rgb):
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    b0, g0, r0 = cv2.split(img)
    black = (b0 < 80) & (g0 < 80) & (r0 < 80)
    ys0, xs0 = np.nonzero(black)
    if len(xs0):
        d0 = np.hypot(xs0.astype(float) - cx, ys0.astype(float) - cy)
        near = d0 < 40
        if near.sum() >= 50:
            cx = float(xs0[near].mean()); cy = float(ys0[near].mean())
    b, g, r = cv2.split(img.astype(np.int32))
    red = (r > 150) & ((r - g) > 80) & ((r - b) > 80)
    ys, xs = np.nonzero(red)
    if len(xs) == 0:
        return None, (cx, cy), 0
    dx = xs.astype(float) - cx; dy = ys.astype(float) - cy
    dist = np.hypot(dx, dy); keep = dist > 18
    if keep.sum() < 20:
        keep = dist > 0
    mx = float(dx[keep].mean()); my = float(dy[keep].mean())
    theta_deg = math.degrees(math.atan2(mx, -my))
    return angle_to_value(theta_deg), (cx, cy), int(keep.sum())

def main():
    model = mujoco.MjModel.from_xml_path("/home/ubuntu/factory/factory_obstacle.xml")
    print(f"LOAD OK nq={model.nq} ngeom={model.ngeom}")
    data = mujoco.MjData(model)
    home = model.keyframe("home")
    data.qpos[:19] = home.qpos[:19]; data.ctrl[:] = home.ctrl[:]
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    results = []
    for name, cam, true_val in [("pump1", "gauge1", 2.5), ("pump2", "gauge2", 5.8), ("pump3", "gauge3", 8.1)]:
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        read_val, centre, nred = read_gauge(img)
        err = None if read_val is None else abs(read_val - true_val)
        print(f"{name} true={true_val} read={read_val} err={err} red_px={nred}")
        results.append({"pump": name, "true_value": true_val, "read_value": read_val, "error": err, "red_px": nred})
        cv2.imwrite(f"/home/ubuntu/factory/gauge_obstacle_{name}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    renderer.close()
    with open("/home/ubuntu/factory/gauge_obstacle_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE /home/ubuntu/factory/gauge_obstacle_results.json")

if __name__ == "__main__":
    main()
