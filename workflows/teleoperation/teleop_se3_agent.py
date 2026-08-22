# Copyright (c) 2022-2024
# SPDX-License-Identifier: BSD-3-Clause
#cd ~/IsaacLab
#PYTHONPATH=$HOME/ws/sonogym/SonoGym/source/spinal_surgery:$PYTHONPATH ./isaaclab.sh -p ~/ws/sonogym/SonoGym/workflows/teleoperation/teleop_se3_agent.py --enable_cameras --task Isaac-robot-US-guidance-v0 --num_envs 1

"""
Keyboard teleoperation for SonoGym ultrasound guidance task (Isaac Lab 5.x).
"""

import argparse
import cProfile
import os

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

# -----------------------------------------------------------f-l-----------------
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


# Changed this part so it consideres rotations too
def _make_actions(delta_pose_raw, device, num_envs):
    dp = _as_torch(delta_pose_raw, device=device).flatten()

    action = dp[:6]

    # action scaling
    action[0] = 4.0 * action[0]  
    action[1] = 4.0 * action[1]  
    action[2] = 4.0 * action[2]  
    action[3] = 0.5 * action[3]
    action[4] = 0.5 * action[4]
    action[5] = 0.5 * action[5]



    return action.unsqueeze(0).repeat(num_envs, 1)



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    os.environ["SONOGYN_TELEOP"] = "1"
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped

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
    step_count = 0

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
            step_count += 1

            if step_count % 50 == 0 and hasattr(raw_env, "US_slicer"):
                print(
                    f"[POSE {step_count}] "
                    f"x={raw_env.US_slicer.current_x_z_x_angle_cmd[0, 0].item():.1f} "
                    f"z={raw_env.US_slicer.current_x_z_x_angle_cmd[0, 1].item():.1f} "
                    f"angle={raw_env.US_slicer.current_x_z_x_angle_cmd[0, 2].item():.3f} "
                    f"roll={raw_env.US_slicer.roll_adj[0, 0].item():.3f}"
                )

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
