# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.
"""

# -----------------------------------------------------------------------------
# Launch Isaac Sim first
# -----------------------------------------------------------------------------

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path
import pickle
import yaml

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-robot-US-guidance-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO", "SAC", "TD3", "DQN", "A2C", "CEM"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args (device/headless/enable_cameras, etc.)
AppLauncher.add_app_launcher_args(parser)

# parse args (keep hydra args)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports AFTER SimulationApp
# -----------------------------------------------------------------------------

import gymnasium as gym
import skrl
from packaging import version

# force registration of custom envs
import spinal_surgery  # noqa: F401
import isaaclab_tasks  # noqa: F401

# skrl runner
if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner
else:
    raise ValueError(f"Unsupported ml_framework: {args_cli.ml_framework}")

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import wandb

# -----------------------------------------------------------------------------
# Version check
# -----------------------------------------------------------------------------

SKRL_MIN_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_MIN_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_MIN_VERSION}'"
    )
    raise SystemExit(1)

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


def _ensure_agent_experiment_block(agent_cfg: dict) -> dict:
    """
    Hydra configs sometimes place 'experiment' at different levels.
    This function normalizes it to agent_cfg['agent']['experiment'].
    """
    if "agent" not in agent_cfg:
        raise KeyError("agent_cfg is missing top-level key 'agent'")

    # If experiment is incorrectly at the top-level, move it under agent
    if "experiment" in agent_cfg and "experiment" not in agent_cfg["agent"]:
        agent_cfg["agent"]["experiment"] = agent_cfg.pop("experiment")

    # Ensure the dict exists
    agent_cfg["agent"].setdefault("experiment", {})

    return agent_cfg


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""

    agent_cfg = _ensure_agent_experiment_block(agent_cfg)

    # -----------------------------
    # override configurations with CLI
    # -----------------------------
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # max iterations for training (skrl trainer expects timesteps)
    if args_cli.max_iterations:
        # typical pattern in IsaacLab scripts: iterations * rollouts
        rollouts = agent_cfg["agent"].get("rollouts", 32)
        agent_cfg.setdefault("trainer", {})
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * rollouts

    # ensure trainer doesn't close env
    agent_cfg.setdefault("trainer", {})
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # seed handling
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set seeds
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
        env_cfg.seed = args_cli.seed

    # -----------------------------
    # logging paths (filesystem)
    # -----------------------------
    # This is the ONLY place where we build filesystem paths.
    # The value in agent_cfg["agent"]["experiment"]["directory"] must remain a simple name like "US_guidance"
    # OR a stable relative path under logs/skrl. We will force a stable structure here.

    # If your YAML already contains directory: "US_guidance" keep it.
    exp_cfg = agent_cfg["agent"]["experiment"]
    exp_name = exp_cfg.get("directory", "US_guidance")
    if not isinstance(exp_name, str) or not exp_name:
        exp_name = "US_guidance"

    # Make log root on disk: logs/skrl/<exp_name>
    log_root_path = Path("logs") / "skrl" / exp_name
    log_root_path.mkdir(parents=True, exist_ok=True)

    # Run folder name: timestamp + algorithm + framework (similar to original scripts)
    run_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"

    # If YAML sets experiment_name, append it (optional)
    requested = exp_cfg.get("experiment_name")
    if isinstance(requested, str) and requested.strip():
        run_dir_name += f"_{requested.strip()}"

    log_dir = log_root_path / run_dir_name
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"[INFO] Run directory: {log_dir}")

    # -----------------------------
    # FORCE checkpoint & logging behavior (so you ALWAYS get .pt files)
    # -----------------------------
    # IMPORTANT:
    # - directory must point to the filesystem root where skrl writes (we give it log_root_path)
    # - experiment_name must match run_dir_name (so checkpoints land inside this run directory)
    exp_cfg["directory"] = str(log_root_path)          # filesystem root
    exp_cfg["experiment_name"] = run_dir_name          # run folder name
    exp_cfg["write_interval"] = 1000
    exp_cfg["checkpoint_interval"] = 5000
    exp_cfg["save_best"] = True
    # keep layout standard
    exp_cfg["store_separately"] = False

    # -----------------------------
    # save configs to params/
    # -----------------------------
    # env_cfg is a dataclass-like object; dumping it as yaml can fail depending on versions.
    # So we store agent_cfg and a lightweight env summary.
    with open(params_dir / "agent.yaml", "w") as f:
        yaml.safe_dump(agent_cfg, f, sort_keys=False)

    with open(params_dir / "agent.pkl", "wb") as f:
        pickle.dump(agent_cfg, f)

    # minimal env snapshot (safe across versions)
    env_snapshot = {
        "task": args_cli.task,
        "device": str(env_cfg.sim.device),
        "num_envs": int(env_cfg.scene.num_envs),
        "seed": getattr(env_cfg, "seed", None),
    }
    with open(params_dir / "env_snapshot.yaml", "w") as f:
        yaml.safe_dump(env_snapshot, f, sort_keys=False)

    # -----------------------------
    # wandb
    # -----------------------------
    wandb.init(project=args_cli.task, config=agent_cfg)

    # -----------------------------
    # create environment
    # -----------------------------
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": str(log_dir / "videos" / "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # -----------------------------
    # training
    # -----------------------------
    runner = Runner(env, agent_cfg)

    # resume training if checkpoint provided
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    print("[INFO] Expected checkpoints folder:")
    print(f"       {log_dir / 'checkpoints'}")
    print("[INFO] Training started...")

    runner.run()

    print("[INFO] Training finished.")
    print("[INFO] Checkpoints should be here:")
    print(f"       {log_dir / 'checkpoints'}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

