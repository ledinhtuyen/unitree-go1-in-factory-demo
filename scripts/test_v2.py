import sys
sys.path.insert(0, "/home/ubuntu/mujoco_playground")
sys.path.insert(0, "/home/ubuntu/factory")
import jax, jax.numpy as jp, numpy as np
from nav_env_v2 import Go1NavSeq
from mujoco import mjx

env = Go1NavSeq()
print("action_size", env.action_size)
print("observation_size", env.observation_size)

# Start zone distribution: 150 resets (enough to check ~1/3 split, faster)
from collections import Counter
cnt = Counter()
for i in range(150):
    st = env.reset(jax.random.PRNGKey(i))
    y = float(st.data.qpos[1])
    z = 0 if y < -1.0 else (1 if y < 1.5 else 2)
    cnt[z] += 1
    assert st.obs["state"].shape == (70,), st.obs["state"].shape
    assert st.obs["privileged_state"].shape == (145,)
print("start zones south/middle/north (expect ~50/50/50):", cnt)
assert all(30 <= cnt[z] <= 70 for z in range(3)), cnt

# JIT step sanity
jit_step = jax.jit(env.step)
st = env.reset(jax.random.PRNGKey(999))
st = jit_step(st, jp.zeros(12))
print("jit step OK reward", float(st.reward), "done", float(st.done))

# Reach switch test: place robot near its target, step, check switch
st = env.reset(jax.random.PRNGKey(7))
ti = int(st.info["target_idx"]); tx = float(st.info["target_xy"][0])
print("reach test target_idx", ti, "target_xy", np.array(st.info["target_xy"]))
# Place 0.2 m south of stop point, facing north, home pose
home_q = np.array(env.mj_model.keyframe("home").qpos, dtype=np.float64)
qpos = np.array(st.data.qpos, dtype=np.float64)
qpos[0], qpos[1], qpos[2] = tx - 0.2, 2.6, home_q[2]
qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
qpos[7:] = home_q[7:]
qvel = np.zeros_like(np.array(st.data.qvel, dtype=np.float64))
ctrl = np.array(env._default_pose, dtype=np.float64)
data = st.data.replace(qpos=jp.array(qpos), qvel=jp.array(qvel), ctrl=jp.array(ctrl))
data = mjx.forward(env.mjx_model, data)
st = st.replace(data=data)
st.info["prev_dist"] = jp.array(0.2, dtype=jp.float32)
st = st.replace(obs=env._get_obs(st.data, st.info))
print("before step dist to target ~0.2, done", float(st.done))
st2 = jit_step(st, jp.zeros(12))
ti2 = int(st2.info["target_idx"])
print("after step target_idx", ti2, "done", float(st2.done),
      "targets_reached", int(st2.info["targets_reached"]),
      "new target_xy", np.array(st2.info["target_xy"]))
assert ti2 != ti, "target did not switch on reach"
assert float(st2.done) == 0.0, "reach should not terminate in v2"
assert int(st2.info["targets_reached"]) == 1
# Next step should not immediately re-switch (new target is >=3 m away)
st3 = jit_step(st2, jp.zeros(12))
print("second step target_idx", int(st3.info["target_idx"]),
      "targets_reached", int(st3.info["targets_reached"]), "done", float(st3.done))
assert int(st3.info["target_idx"]) == ti2
print("V2 TEST OK")
