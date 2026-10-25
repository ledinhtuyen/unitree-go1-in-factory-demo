import os, sys, math, json
import numpy as np
import mujoco
import cv2

FACTORY = "/home/ubuntu/factory/factory.xml"
PUMPS = [
    ("pump1", "gauge1", 2.5, -60.0),
    ("pump2", "gauge2", 5.8, 19.2),
    ("pump3", "gauge3", 8.1, 74.4),
]
NEEDLE_JOINT_IDX = {"pump1": 19, "pump2": 20, "pump3": 21}

def angle_to_value(deg):
    return (deg + 120.0) / 240.0 * 10.0

def read_gauge(img_rgb):
    # img_rgb: HxWx3 uint8, camera looking at gauge, gauge centred near image centre
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Centre: black hub near image centre (ticks are black too, but >40px away)
    cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    b0, g0, r0 = cv2.split(img)
    black = (b0 < 80) & (g0 < 80) & (r0 < 80)
    ys0, xs0 = np.nonzero(black)
    if len(xs0):
        d0 = np.hypot(xs0.astype(float) - cx, ys0.astype(float) - cy)
        near = d0 < 40
        if near.sum() >= 50:
            cx = float(xs0[near].mean())
            cy = float(ys0[near].mean())
    # Red needle: BGR red is (0,0,255); require R high, R-G and R-B large
    b, g, r = cv2.split(img.astype(np.int32))
    red = (r > 150) & ((r - g) > 80) & ((r - b) > 80)
    ys, xs = np.nonzero(red)
    if len(xs) == 0:
        return None, (cx, cy), 0
    # Exclude hub region (within 18 px of centre) so the mean points to the tip side
    dx = xs.astype(float) - cx
    dy = ys.astype(float) - cy
    dist = np.hypot(dx, dy)
    keep = dist > 18
    if keep.sum() < 20:
        keep = dist > 0
    mx = float(dx[keep].mean())
    my = float(dy[keep].mean())
    # Image y is down, needle up is -my. Angle from vertical, positive to the right.
    theta_deg = math.degrees(math.atan2(mx, -my))
    return angle_to_value(theta_deg), (cx, cy), int(keep.sum())

def main():
    print("MUJOCO_GL=", os.environ.get("MUJOCO_GL"))
    model = mujoco.MjModel.from_xml_path(FACTORY)
    print(f"LOAD OK nq={model.nq} nu={model.nu}")
    data = mujoco.MjData(model)
    home = model.keyframe("home")
    data.qpos[:19] = home.qpos[:19]
    data.ctrl[:] = home.ctrl[:]
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    results = []
    # Primary: true pump values
    for name, cam, true_val, true_deg in PUMPS:
        data.qpos[19] = math.radians(-60.0)
        data.qpos[20] = math.radians(19.2)
        data.qpos[21] = math.radians(74.4)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        read_val, centre, nred = read_gauge(img)
        err = None if read_val is None else abs(read_val - true_val)
        print(f"{name} true={true_val:.1f} bar angle={true_deg:.1f} deg read={read_val} centre={centre} red_px={nred} err={err}")
        results.append({"pump": name, "true_value": true_val, "true_angle_deg": true_deg, "read_value": read_val, "error": err, "red_px": nred})
        cv2.imwrite(f"/home/ubuntu/factory/gauge_{name}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # Robustness: sweep needle angles on pump1 camera
    print("--- sweep pump1 ---")
    for deg in [-120.0, -60.0, 0.0, 60.0, 120.0]:
        data.qpos[19] = math.radians(deg)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="gauge1")
        img = renderer.render()
        read_val, centre, nred = read_gauge(img)
        true_val = angle_to_value(deg)
        print(f"sweep angle={deg:.0f} true={true_val:.2f} read={read_val} err={None if read_val is None else abs(read_val-true_val):} red_px={nred}")
        results.append({"pump": "pump1_sweep", "true_value": true_val, "true_angle_deg": deg, "read_value": read_val, "error": None if read_val is None else abs(read_val-true_val), "red_px": nred})
    renderer.close()
    with open("/home/ubuntu/factory/gauge_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE /home/ubuntu/factory/gauge_results.json")

if __name__ == "__main__":
    main()
