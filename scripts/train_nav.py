"""Train the Go1Nav autonomous navigation policy with Brax PPO.

Weight transfer (documented, not silent): the nav observation is larger
than the joystick observation (policy 70 vs 48, value 145 vs 123), so the
obstacle checkpoint cannot be restored verbatim. Surgery:
  - policy first-layer kernel (48,512) -> (70,512): rows 0:45 (proprioception,
    same order in both obs) copied verbatim; rows 45:70 (nav features)
    initialised N(0, 0.01) so the transferred locomotion behaviour is
    initially almost unchanged and nav influence grows during training.
  - value first-layer kernel (123,512) -> (145,512): rows 0:45 copied,
    rows 45:70 nav N(0, 0.01), rows 70:145 = old rows 48:123 (privileged extras).
  - deeper layers, biases, output layers copied verbatim.
  - normalizer: proprio means/stds copied; nav means 0, stds 1,
    summed_variance set to count * 1.0 (count estimated from an old dim).
  - optimizer state is fresh (Brax restore_params only carries
    normalizer/policy/value).
If --fresh is given, no surgery is done and training starts from scratch.

Usage:
  python train_nav.py --num_timesteps 50000000 --logdir /home/ubuntu/go1-nav [--fresh]
  Short smoke test: --num_timesteps 500000 --logdir /home/ubuntu/go1-nav-smoke
"""
import sys, os, copy, argparse, functools, time
sys.path.insert(0, "/home/ubuntu/mujoco_playground")
sys.path.insert(0, "/home/ubuntu/factory")

import jax
import jax.numpy as jp
import numpy as np

OLD_CKPT = "/home/ubuntu/go1-obstacle/Go1JoystickObstacle-20261025-014038/checkpoints/000058982400"


def register():
    from nav_env import Go1Nav, default_nav_config
    from mujoco_playground._src import locomotion as loc
    loc.register_environment("Go1Nav", Go1Nav, default_nav_config)
    try:
        from mujoco_playground import registry as reg
        if hasattr(reg, "register_environment"):
            reg.register_environment("Go1Nav", Go1Nav, default_nav_config)
    except Exception as e:
        print(f"registry alias register skipped: {e}", flush=True)
    from mujoco_playground.config import locomotion_params
    _orig = locomotion_params.brax_ppo_config
    def _patched(env_name, impl=None):
        if env_name == "Go1Nav":
            return _orig("Go1JoystickFlatTerrain", impl)
        return _orig(env_name, impl)
    locomotion_params.brax_ppo_config = _patched
    print("Registered Go1Nav", flush=True)


def surgery(rng):
    """Return restore_params [normalizer, policy_dict, value_dict] for nav sizes."""
    from brax.training import checkpoint as brax_checkpoint
    from brax.training.acme import running_statistics
    old = brax_checkpoint.load(OLD_CKPT)
    old_norm, old_pol, old_val = old[0], old[1], old[2]
    rng, k1, k2 = jax.random.split(rng, 3)
    # --- normalizer ---
    # State: old 48 -> new 70 (proprio 45 + nav 25).
    # Privileged: old 123 -> new 145 (proprio 45 + nav 25 + extras 75).
    # Build state normalizer (70): rows 0:45 old 0:45, rows 45:70 nav.
    def norm_70():
        om, os_, osv = old_norm.mean["state"], old_norm.std["state"], old_norm.summed_variance["state"]
        cnt = float(osv[0] / max(float(os_[0]) ** 2, 1e-9))
        mean = np.zeros(70, dtype=np.float32); std = np.ones(70, dtype=np.float32)
        sv = np.full(70, cnt, dtype=np.float32)
        mean[:45] = np.array(om[:45]); std[:45] = np.array(os_[:45]); sv[:45] = np.array(osv[:45])
        return mean, std, sv
    def norm_145():
        om, os_, osv = old_norm.mean["privileged_state"], old_norm.std["privileged_state"], old_norm.summed_variance["privileged_state"]
        cnt = float(osv[0] / max(float(os_[0]) ** 2, 1e-9))
        mean = np.zeros(145, dtype=np.float32); std = np.ones(145, dtype=np.float32)
        sv = np.full(145, cnt, dtype=np.float32)
        mean[:45] = np.array(om[:45]); std[:45] = np.array(os_[:45]); sv[:45] = np.array(osv[:45])
        mean[70:] = np.array(om[48:]); std[70:] = np.array(os_[48:]); sv[70:] = np.array(osv[48:])
        return mean, std, sv
    m70, s70, sv70 = norm_70()
    m145, s145, sv145 = norm_145()
    new_norm = running_statistics.RunningStatisticsState(
        mean={"state": jp.array(m70), "privileged_state": jp.array(m145)},
        std={"state": jp.array(s70), "privileged_state": jp.array(s145)},
        summed_variance={"state": jp.array(sv70), "privileged_state": jp.array(sv145)},
        count=old_norm.count, std_eps=old_norm.std_eps,
    )
    # --- policy network ---
    new_pol = copy.deepcopy(old_pol)
    old_k = np.array(old_pol["params"]["hidden_0"]["kernel"])  # (48, 512)
    new_k = np.zeros((70, 512), dtype=np.float32)
    new_k[:45] = old_k[:45]
    new_k[45:] = np.array(jax.random.normal(k1, (25, 512)) * 0.01, dtype=np.float32)
    new_pol["params"]["hidden_0"]["kernel"] = jp.array(new_k)
    # --- value network ---
    new_val = copy.deepcopy(old_val)
    old_kv = np.array(old_val["params"]["hidden_0"]["kernel"])  # (123, 512)
    new_kv = np.zeros((145, 512), dtype=np.float32)
    new_kv[:45] = old_kv[:45]
    new_kv[45:70] = np.array(jax.random.normal(k2, (25, 512)) * 0.01, dtype=np.float32)
    new_kv[70:] = old_kv[48:]
    new_val["params"]["hidden_0"]["kernel"] = jp.array(new_kv)
    print(f"SURGERY OK: policy kernel {old_k.shape} -> {new_k.shape}, "
          f"value kernel {old_kv.shape} -> {new_kv.shape}", flush=True)
    return [new_norm, new_pol, new_val]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_timesteps", type=int, default=50000000)
    ap.add_argument("--logdir", default="/home/ubuntu/go1-nav")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    register()
    from mujoco_playground.config import locomotion_params
    from mujoco_playground._src import locomotion as loc
    from mujoco_playground._src import wrapper
    from brax.training.agents.ppo.train import train as ppo_train
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics

    ppo_params = locomotion_params.brax_ppo_config("Go1Nav")
    ppo_params.num_timesteps = args.num_timesteps
    ppo_params.num_envs = 1024
    ppo_params.num_evals = 10
    ppo_params.episode_length = 1000
    env = loc.load("Go1Nav", config=loc.get_default_config("Go1Nav"))
    eval_env = loc.load("Go1Nav", config=loc.get_default_config("Go1Nav"))
    print(f"env obs size: {env.observation_size} action_size: {env.action_size}", flush=True)

    restore_params = None
    if not args.fresh:
        restore_params = surgery(jax.random.PRNGKey(123))
    else:
        print("FRESH training: no weight transfer", flush=True)

    # Checkpoint layout like train_jax_ppo: logdir/<env>-<timestamp>/checkpoints
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = os.path.join(args.logdir, f"Go1Nav-{stamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, "checkpoints")
    print(f"Checkpoint dir: {save_path}", flush=True)

    times = [time.monotonic()]
    def progress(num_steps, metrics):
        times.append(time.monotonic())
        if "eval/episode_reward" in metrics:
            print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f} "
                  f"(JIT+train so far {times[-1]-times[0]:.0f}s)", flush=True)
        elif "episode/sum_reward" in metrics:
            print(f"{num_steps}: mean episode reward={metrics['episode/sum_reward']:.3f}", flush=True)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state", value_obs_key="privileged_state",
    )
    train_fn = functools.partial(
        ppo_train,
        num_timesteps=ppo_params.num_timesteps,
        num_evals=ppo_params.num_evals,
        reward_scaling=ppo_params.reward_scaling,
        episode_length=ppo_params.episode_length,
        normalize_observations=ppo_params.normalize_observations,
        action_repeat=ppo_params.action_repeat,
        unroll_length=ppo_params.unroll_length,
        num_minibatches=ppo_params.num_minibatches,
        num_updates_per_batch=ppo_params.num_updates_per_batch,
        discounting=ppo_params.discounting,
        learning_rate=ppo_params.learning_rate,
        entropy_cost=ppo_params.entropy_cost,
        num_envs=ppo_params.num_envs,
        batch_size=ppo_params.batch_size,
        max_grad_norm=ppo_params.max_grad_norm,
        network_factory=network_factory,
        seed=args.seed,
        restore_params=restore_params,
        save_checkpoint_path=save_path,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=128,
        num_resets_per_eval=ppo_params.num_resets_per_eval,
    )
    make_inference_fn, params, _ = train_fn(
        environment=env, progress_fn=progress, eval_env=eval_env)
    print("Done training.", flush=True)
    print(f"Time to JIT compile: {times[1]-times[0]:.1f}s (first progress callback)", flush=True)


if __name__ == "__main__":
    main()
