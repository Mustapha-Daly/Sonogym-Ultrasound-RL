# Copyright (c) 2022-2024
# SPDX-License-Identifier: BSD-3-Clause
#cd ~/IsaacLab
#PYTHONPATH=$HOME/ws/sonogym/SonoGym/source/spinal_surgery:$PYTHONPATH ./isaaclab.sh -p ~/ws/sonogym/SonoGym/workflows/teleoperation/teleop_se3_agent.py --enable_cameras --task Isaac-robot-US-guidance-v0 --num_envs 8

"""
Keyboard teleoperation for SonoGym ultrasound guidance task (Isaac Lab 5.x).
"""

import argparse
import cProfile

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Keyboard teleoperation (US guidance)")
parser.add_argument("--task", type=str, default="Isaac-robot-US-guidance-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ------------------------------------------------------------l-----------------
# Launch Isaac Sim FIRST
# -----------------------------------------------------------------------------
#app_launcher = AppLauncher(headless=args_cli.headless)
app_launcher = AppLauncher(
    headless=args_cli.headless,
    enable_cameras=args_cli.enable_cameras,)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports AFTER SimulationApp
# -----------------------------------------------------------------------------
import torch
import gymnasium as gym

import isaaclab_tasks  # registers default tasks
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab.devices import Se3Keyboard
from isaaclab.devices import Se3KeyboardCfg

# Register SonoGym task
import spinal_surgery.tasks.robot_US_guidance  # noqa: F401


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _as_torch(x, device):
    """Convert numpy / list / tensor to torch.Tensor on correct device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _make_actions(delta_pose_raw, device, num_envs):
    """
    Map SE(3) keyboard delta to US-guidance action space.

    Keyboard gives:
        [dx, dy, dz, droll, dpitch, dyaw]

    US guidance expects:
        [dx, dy, dz]
    """
    dp = _as_torch(delta_pose_raw, device=device).flatten()

    # Translation only
    transl = dp[:3]

    # IMPORTANT: scale (keyboard deltas are tiny)
    transl = 5.0 * transl

    # (num_envs, 3)
    return transl.unsqueeze(0).repeat(num_envs, 1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Parse env config
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] env.action_space = {env.action_space}")
    print(f"[INFO] num_envs = {env.unwrapped.num_envs}")

    # Keyboard device (CFG-BASED API, REQUIRED)
    kb_cfg = Se3KeyboardCfg(
        pos_sensitivity=0.05,
        rot_sensitivity=0.05,
    )
    teleop_interface = Se3Keyboard(kb_cfg)

    print(teleop_interface)

    # Reset
    env.reset()
    teleop_interface.reset()

    # -----------------------------------------------------------------------------
    # Simulation loop
    # -----------------------------------------------------------------------------
    while simulation_app.is_running():
        with torch.inference_mode():
            # IMPORTANT: advance() returns ONLY ONE value
            delta_pose = teleop_interface.advance()

            actions = _make_actions(
                delta_pose_raw=delta_pose,
                device=env.unwrapped.device,
                num_envs=env.unwrapped.num_envs,
            )

            env.step(actions)

    env.close()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    profiler.dump_stats("teleop_stats.prof")
    simulation_app.close()

