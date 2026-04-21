# SonoGym/scripts/run_us_guidance.py

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg

# FORCE registration
import spinal_surgery.tasks.robot_US_guidance.robotic_US_guidance


def main():
    # Parse config
    env_cfg = parse_env_cfg(
        "Isaac-robot-US-guidance-v0",
        num_envs=1,
    )

    # CREATE environment
    env = gym.make(
        "Isaac-robot-US-guidance-v0",
        cfg=env_cfg,
        render_mode="human",
    )

    # THIS IS THE MISSING LINE
    env.reset()

    # Keep sim alive
    while True:
        env.step(env.action_space.sample())


if __name__ == "__main__":
    main()

