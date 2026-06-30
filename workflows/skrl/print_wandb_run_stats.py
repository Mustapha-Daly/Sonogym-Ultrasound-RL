#!/usr/bin/env python3

import argparse
import math
import re
from pathlib import Path

import wandb
from ruamel.yaml import YAML


DEFAULT_KEYS = [
    "liver_fraction",
    "reward_mean",
    "bone_fraction",
    "shadow_fraction",
    "progress",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_CFG_PATH = REPO_ROOT / "source/spinal_surgery/spinal_surgery/tasks/robot_US_guidance/cfgs/robotic_US_guidance.yaml"
MODEL_PATH = REPO_ROOT / "source/spinal_surgery/spinal_surgery/lab/agents/skrl_actor_critic.py"


def load_metric(run, key):
    values = []
    for row in run.scan_history(keys=[key]):
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def print_metric_stats(key, values):
    print(f"\n{key}:")
    if not values:
        print("  logged_points: 0")
        print("  no data found")
        return

    print(f"  logged_points: {len(values)}")
    print(f"  mean_over_run: {sum(values) / len(values):.6f}")
    print(f"  final: {values[-1]:.6f}")
    print(f"  max: {max(values):.6f}")
    print(f"  min: {min(values):.6f}")


def load_task_cfg():
    yaml = YAML(typ="safe")
    with TASK_CFG_PATH.open("r", encoding="utf-8") as f:
        return yaml.load(f)


def load_initial_std():
    text = MODEL_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"log_std_parameter\s*=\s*nn\.Parameter\(\s*torch\.full\(\(self\.num_actions,\),\s*(.*?)\)\s*\)",
        text,
        re.DOTALL,
    )
    if not match:
        return None, None

    expr = match.group(1).strip()

    try:
        if expr.startswith("math.log(") and expr.endswith(")"):
            target_std = float(expr[len("math.log("):-1])
            return expr, target_std

        log_std = float(expr)
        return expr, math.exp(log_std)
    except ValueError:
        return expr, None


def print_setup_summary():
    cfg = load_task_cfg()
    sim_cfg = cfg["sim"]
    motion_cfg = cfg["motion_planning"]
    expr, actual_std = load_initial_std()

    print("\nsetup:")
    print(f"  patient_xz_range: {sim_cfg['patient_xz_range']}")
    print(f"  patient_xz_init_range: {sim_cfg['patient_xz_init_range']}")
    print(
        "  fixed_init_angle: "
        f"{sim_cfg['patient_xz_init_range'][0][2]} -> {sim_cfg['patient_xz_init_range'][1][2]}"
    )
    print(f"  max_roll_adj: {motion_cfg['max_roll_adj']}")
    print(f"  roll_init: {motion_cfg['roll_init']}")
    print(f"  episode_length_s: {sim_cfg['episode_length']}")
    if expr is not None:
        print(f"  log_std_code: {expr}")
    if actual_std is not None:
        print(f"  actual_initial_std: {actual_std:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Print full-history W&B stats and current local setup."
    )
    parser.add_argument(
        "run_path",
        help=(
            "W&B run path, for example: "
            "mustaphadaly-technical-university-of-munich/"
            "Isaac-robot-US-guidance-v0/unique-star-13"
        ),
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=DEFAULT_KEYS,
        help="Metrics to summarize. Default: liver_fraction reward_mean bone_fraction shadow_fraction progress",
    )
    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(args.run_path)

    print(f"run_path: {'/'.join(run.path)}")
    print(f"run_name: {run.name}")
    print(f"run_state: {run.state}")

    print_setup_summary()

    for key in args.keys:
        values = load_metric(run, key)
        print_metric_stats(key, values)


if __name__ == "__main__":
    main()
