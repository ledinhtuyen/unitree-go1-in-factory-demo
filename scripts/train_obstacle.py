"""Train Go1 joystick policy in the obstacle factory scene.

Registers a custom playground env (Go1JoystickObstacle) that loads
/home/ubuntu/factory/factory_obstacle.xml, then reuses the stock
train_jax_ppo training loop. No playground package file is edited:
registration happens at runtime via register_environment.

Restore: starts from the 50M flat-terrain checkpoint, with the SAME
network sizes (512/256/128, state/privileged_state) passed as flags,
so Brax checkpoint restore shapes match. If restore fails, the
script exits early (before long training) and the caller should
re-run without --load_checkpoint_path for a fresh run.

Known limitation (stated honestly in the report): this env's
observation is still proprioceptive only (same 48-dim state as the
flat env). It does NOT see doors/obstacles. Training here teaches
robust locomotion under contact with walls/crates (bumping,
recovery, turning when velocity is blocked), NOT autonomous
door perception. Door routing in evaluation is therefore done by
the waypoint controller, which steers through door centres.
The second route (navigation observations + nav reward) is only
needed if waypoint-through-doors evaluation fails.
"""
import sys
sys.path.insert(0, "/home/ubuntu/mujoco_playground")

OBSTACLE_XML = "/home/ubuntu/factory/factory_obstacle.xml"
CKPT_50M = "/home/ubuntu/go1-50m/Go1JoystickFlatTerrain-20261025-005702/checkpoints"  # parent dir: train_jax_ppo picks the latest numeric subdir (000058982400)


def register():
    from mujoco_playground._src.locomotion.go1 import joystick as go1_joystick
    from mujoco_playground._src.locomotion.go1 import base as go1_base
    from mujoco_playground._src import locomotion as loc

    class ObstacleJoystick(go1_joystick.Joystick):
        def __init__(self, config=None, config_overrides=None):
            if config is None:
                config = go1_joystick.default_config()
            # More contacts in the obstacle hall than on flat ground.
            config.naconmax = 8 * 8192
            config.njmax = 80
            go1_base.Go1Env.__init__(
                self, xml_path=OBSTACLE_XML, config=config,
                config_overrides=config_overrides,
            )
            self._post_init()

    loc.register_environment(
        "Go1JoystickObstacle", ObstacleJoystick, go1_joystick.default_config
    )
    try:
        from mujoco_playground import registry as reg
        if hasattr(reg, "register_environment"):
            reg.register_environment(
                "Go1JoystickObstacle", ObstacleJoystick,
                go1_joystick.default_config,
            )
    except Exception as e:  # registry may be the same module; ignore
        print(f"registry alias register skipped: {e}", flush=True)

    # locomotion_params.brax_ppo_config whitelists env names; map our
    # custom name to the Go1 flat config (same network sizes, same
    # episode length) by monkey-patching at runtime -- no file edited.
    from mujoco_playground.config import locomotion_params
    _orig_brax_ppo_config = locomotion_params.brax_ppo_config
    def _patched_brax_ppo_config(env_name, impl=None):
        if env_name == "Go1JoystickObstacle":
            return _orig_brax_ppo_config("Go1JoystickFlatTerrain", impl)
        return _orig_brax_ppo_config(env_name, impl)
    locomotion_params.brax_ppo_config = _patched_brax_ppo_config
    print("Patched locomotion_params.brax_ppo_config for Go1JoystickObstacle", flush=True)
    print("Registered Go1JoystickObstacle ->", OBSTACLE_XML, flush=True)


def main():
    register()
    sys.argv = [
        "train-jax-ppo",
        "--env_name", "Go1JoystickObstacle",
        "--num_envs", "1024",
        "--num_timesteps", "50000000",
        "--seed", "1",
        "--logdir", "/home/ubuntu/go1-obstacle",
        "--load_checkpoint_path", CKPT_50M,
        "--policy_hidden_layer_sizes", "512,256,128",
        "--value_hidden_layer_sizes", "512,256,128",
        "--policy_obs_key", "state",
        "--value_obs_key", "privileged_state",
    ]
    from learning import train_jax_ppo
    train_jax_ppo.run()


if __name__ == "__main__":
    main()
