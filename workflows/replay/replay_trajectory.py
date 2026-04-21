# workflows/replay/replay_trajectory.py

from isaaclab.app import AppLauncher
import argparse
import torch
import time
import gymnasium as gym
import numpy as np

# -------------------------------------------------
# CLI
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-robot-US-guidance-v0")
parser.add_argument("--traj", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda:0")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.enable_cameras = True

# -------------------------------------------------
# Launch Isaac Sim
# -------------------------------------------------
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -------------------------------------------------
# Imports AFTER SimulationApp
# -------------------------------------------------
import spinal_surgery
import isaaclab_tasks  # noqa
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# -------------------------------------------------
# Load trajectory
# -------------------------------------------------
traj = torch.load(args.traj, map_location=args.device)

# shape handling
if traj.ndim == 3:
    traj = traj[0]  # take env 0

print("Trajectory shape:", traj.shape)

# -------------------------------------------------
# Create environment
# -------------------------------------------------
env = gym.make(
    args.task,
    num_envs=args.num_envs,
    render_mode="rgb_array",
)

if isinstance(env.unwrapped, DirectMARLEnv):
    env = multi_agent_to_single_agent(env)

env = SkrlVecEnvWrapper(env, ml_framework="torch")

obs, _ = env.reset()

env_unwrapped = env.unwrapped

# -------------------------------------------------
# Replay loop
# -------------------------------------------------
print("Replaying trajectory...")

for t in range(len(traj)):
    actions = traj[t].unsqueeze(0)

    # step environment
    obs, reward, terminated, truncated, info = env.step(actions)

    # ---- LABEL MAP VISUALIZATION ----
    label = env_unwrapped.US_slicer.label_img_tensor[0, :, :, 0]

    kidney_pixels = (label == env_unwrapped.cfg.KIDNEY_LABEL_ID).sum().item()
    vertebra_pixels = (label == env_unwrapped.cfg.VERTEBRA_LABEL_ID).sum().item()

    print(
        f"t={t:04d} | kidney_pixels={kidney_pixels:6d} | vertebra_pixels={vertebra_pixels:6d}"
    )

    # visualize segmentation if enabled
    if env_unwrapped.sim_cfg["vis_seg_map"]:
        env_unwrapped.US_slicer.visualize("seg")

    # stop if terminated
    if terminated.any():
        print("Termination condition reached.")
        break

    time.sleep(env_unwrapped.physics_dt)

# -------------------------------------------------
# Cleanup
# -------------------------------------------------
env.close()
simulation_app.close()
