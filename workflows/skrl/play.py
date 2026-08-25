# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#~$ PYTHONPATH=$HOME/ws/sonogym/SonoGym/source/spinal_surgery:$PYTHONPATH ./isaaclab.sh -p $HOME/ws/sonogym/SonoGym/workflows/skrl/play.py   --task Isaac-robot-US-guidance-v0   --checkpoint /home/yue/IsaacLab/logs/skrl/US_guidance/2026-03-22_00-00-53_ppo_torch_PPO_US/checkpoints/best_agent.pt   --num_envs 16   --enable_cameras


"""""
cd ~/IsaacLab
PYTHONPATH=$HOME/ws/sonogym/SonoGym/source/spinal_surgery:$PYTHONPATH \
  ./isaaclab.sh -p ~/ws/sonogym/SonoGym/workflows/skrl/play.py \
  --task Isaac-robot-US-guidance-v0 \
  --checkpoint ~/IsaacLab/logs/skrl/US_guidance/2026-08-15_13-45-36_ppo_torch_PPO_US/ccheckpoints/best_agent.pt \
  --num_envs 1 \
  --enable_cameras \
  --noise_k 0.0
"""    
# ~/IsaacLab/logs/skrl/US_guidance/2026-07-11_17-41-50_ppo_torch_PPO_US/checkpoints/best_agent.pt \

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default='Isaac-robot-US-guided-surgery-v0', help="Name of the task.")
parser.add_argument("--patient_id", type=str, default=None, help="Override patient.id_list[0] from YAML")
parser.add_argument("--checkpoint", type=str, default='/home/yunkao/git/IsaacLabExtensionTemplate/logs/experiments/us-guided-surgery/single/model-based-sim/PPO/2025-04-25_18-39-42_ppo_torch_PPO_default_US_net/checkpoints/best_agent.pt', help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
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
    choices=["AMP", "PPO", "IPPO", "MAPPO", "A2C"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--num_steps", type=int, default=0, help="Stop after this many steps (0 = run forever).")
parser.add_argument("--noise_k", type=float, default=1.0, help="Noise blend: action = mean + k*(sample-mean). 0=deterministic (pure mean), 1=full stochastic.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import cProfile

if args_cli.patient_id:
    os.environ["SONOGYM_PATIENT_ID"] = args_cli.patient_id

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch.running_standard_scaler import RunningStandardScaler

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import spinal_surgery
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from spinal_surgery.lab.agents.skrl_actor_critic import SharedModel
import wandb

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # signal to the env that we are in inference mode (enables episode-end console prints)
    os.environ["SONOGYM_INFERENCE"] = "1"

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (physics) dt for real-time evaluation
    try:
        dt = env.physics_dt
    except AttributeError:
        dt = env.unwrapped.physics_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # init wandb
    #wandb.init(project=args_cli.task, config=env_cfg)

    # handle to the raw IsaacLab env (for probe-pose printing in the loop below)
    _raw_env = env.unwrapped

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # build SharedModel  
    _policy_cfg = experiment_cfg.get("models", {}).get("policy", {})
    device = env_cfg.sim.device
    models = {}
    models["policy"] = SharedModel(
        env.observation_space, env.action_space, device,
        min_log_std=_policy_cfg.get("min_log_std", -3.0),
        max_log_std=_policy_cfg.get("max_log_std", 1.0),
        initial_log_std=_policy_cfg.get("initial_log_std", 0.0),
    )
    models["value"] = models["policy"]

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update(experiment_cfg.get("agent", {}))
    if ppo_cfg.get("value_preprocessor") == "RunningStandardScaler":
        ppo_cfg["value_preprocessor"] = RunningStandardScaler
        ppo_cfg["value_preprocessor_kwargs"] = {"size": 1}
    ppo_cfg["learning_rate_scheduler"] = None  # scheduler not needed in eval mode
    ppo_cfg["experiment"]["write_interval"] = 0
    ppo_cfg["experiment"]["checkpoint_interval"] = 0

    memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=device, replacement=False)
    agent = PPO(models=models, memory=memory, cfg=ppo_cfg,
                observation_space=env.observation_space,
                action_space=env.action_space, device=device)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    agent.load(resume_path)
    agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent actions
            else:
                # noise-blend: action = mean + k*(sample - mean)
                #   --noise_k 0.0  → deterministic (pure mean, tests μ alone)
                #   --noise_k 1.0  → full stochastic (mean + full noise)
                # Sweep k = 0, 0.25, 0.5, 1.0 to see if search relies on noise (luck)
                # or holds at low k (skill).
                mean = outputs[-1].get("mean_actions", outputs[0])
                sample = outputs[0]
                actions = mean + args_cli.noise_k * (sample - mean)
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        timestep += 1

        # Print the probe 4-DoF pose every 10 steps so we can see whether the
        # policy is actually searching or just repeating a fixed sweep.
        if timestep % 50 == 0:
            try:
                cmd = _raw_env.US_slicer.current_x_z_x_angle_cmd[0]
                roll = _raw_env.US_slicer.roll_adj[0, 0]
                print(
                    f"[POSE {timestep}] "
                    f"x={cmd[0].item():.1f} "
                    f"z={cmd[1].item():.1f} "
                    f"angle={cmd[2].item():.3f} "
                    f"roll={roll.item():.3f}"
                )
            except Exception:
                pass

        if args_cli.video:
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.num_steps > 0 and timestep >= args_cli.num_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("main_stats.prof")
    # close sim app
    simulation_app.close()
