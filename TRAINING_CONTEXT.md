# Training Context — Liver Intercostal US Guidance RL

## Goal
Train a PPO agent (skrl + IsaacLab) to sweep a robotic ultrasound probe over a **15mm spherical target volume inside the liver** of patient s0030, achieving ≥85% 3D voxel coverage. Based on **Bi et al. 2026** paper. Currently in **overfitting phase** — one patient, one target, learn it perfectly before generalising.

---

## Key Files

| File | Purpose |
|---|---|
| `source/spinal_surgery/spinal_surgery/tasks/robot_US_guidance/robotic_US_guidance_liver.py` | Main env: reward, coverage buffers, wandb logging |
| `source/spinal_surgery/spinal_surgery/tasks/robot_US_guidance/cfgs/robotic_US_guidance.yaml` | Task config: reward params, init position, episode length, target volume |
| `source/spinal_surgery/spinal_surgery/tasks/robot_US_guidance/agents/skrl_ppo_cfg.yaml` | PPO config: LR, rollouts, log_std, network architecture |
| `source/spinal_surgery/spinal_surgery/lab/agents/skrl_actor_critic.py` | SharedModel: CNN+MLP policy+value, log_std init |
| `workflows/skrl/train.py` | Training entry point: reads YAML, builds models, launches skrl |

---

## Reward Function (Paper Bi et al. 2026)

All implemented in `_get_rewards()`:

```
Eq. 2 — Coverage:          rc = new_coverage_count / N
Eq. 3 — Attenuation:       ra = exp(-dt / Rc)         dt = probe XZ distance to target center (voxels)
Eq. 4 — Shadow fraction:   pt = shadow_pixels / Nt    Nt = shadow_pixels + non_shadow_pixels
         Shadow avoidance: rs = 1 - pt
Eq. 5 — Combined:          rt = w_coverage*rc + alpha1*ra + alpha2*rs
Eq. 7 — Gate:              reward = rt if pt < shadow_thresh else -0.1
```

**Current YAML params:**
```yaml
reward:
  w_coverage: 5.0        # amplifies sparse rc signal for PPO
  shadow_thresh: 0.30    # raised from 0.15 (old gate fired too aggressively)
  alpha1: 1.0
  alpha2: 0.5
  attenuation_Rc: 30.0   # lowered from 100 (stronger pull to target center)
```

**Why w_coverage=5**: paper uses coefficient 1 with DQN discrete; we use PPO continuous so rc signal needs amplification.

**Shadow denominator fix (important):** `_label_for_task` is post-shadow so `(label!=0)` excludes shadow pixels. Correct Nt = `non_shadow_pixels + shadow_pixels` (both `_get_rewards` and `_update_coverage` use this).

**Shadow gate in `_update_coverage`:** when `shadow_ok=False`, voxels are NOT marked in `scanned_target_mask` AND `new_coverage_count` stays 0 — matching paper: "scanned volume is disregarded".

---

## Target Volume

```yaml
target_volume:
  enabled: True
  center_voxel: [206, 175, 226]   # cx, cy, cz in s0030 voxel space
  radius_mm: 15                    # → radius_voxels = 15/1.5 = 10 voxels
  label_id: 200
```

**Math:** sphere of radius 10 vox centered at (206,175,226), intersected with liver voxels only. **4163 voxels total (N)** (verified from `[TARGET VOLUME]` print at startup).

**Init position:** `patient_xz_init_range: [[196, 216, 0.0], [216, 236, 0.0]]` — random ±10 vox from sphere center XZ. Agent starts somewhere inside/near the sphere and must sweep to cover it.

**Why not fixed at center:** Fixed init (206,226) caused the random policy to trivially achieve 85% coverage in 300–450 steps just by wandering, giving near-zero advantage signal and no real learning. Random ±10 vox offset forces the agent to actually learn a directed sweep pattern.

**To cover the full sphere:** agent must reach ±10 voxels from center in X and Z. With action scale [2,2,...] and max_action [4,4,...], this takes ~5 steps per dimension.

---

## Episode Structure

```yaml
sim:
  episode_length: 100          # seconds → 6000 agent steps (decimation=2, sim_dt=1/120)
  patient_xz_range: [[180, 200, -0.5], [248, 280, 0.5]]   # full liver zone
  patient_xz_init_range: [[206, 226, 0.0], [206, 226, 0.0]]  # fixed at sphere center
```

- **Termination:** coverage_fraction ≥ 0.85 OR 6000 steps elapsed
- **Action space:** [dx, dz, d_angle, d_roll], scale [2,2,0.25,0.15], max [4,4,0.5,0.3]
- **Observation:** seg label image (3-frame stack) + pose history (12 values)

---

## Policy (SharedModel)

`source/spinal_surgery/spinal_surgery/lab/agents/skrl_actor_critic.py` — class `SharedModel`

- Shared CNN backbone for policy and value heads
- CNN: 4× Conv2d layers → flatten → 512-dim features
- Pose MLP: 12 → 128 → 64
- Policy head: (512+64) → 256 → 128 → num_actions (mean)
- Value head: (512+64) → 256 → 128 → 1
- `log_std_parameter` = learnable per-action, initialised from YAML

**log_std values (all from YAML, wired via train.py):**
```yaml
min_log_std: -3.0      # σ_min = exp(-3) ≈ 0.05 → ±0.1 vox (near-deterministic)
max_log_std: 1.0       # σ_max = exp(1)  ≈ 2.72 → ±5.4 vox (caps wild exploration)
initial_log_std: 0.0   # σ_init = exp(0) = 1.0  → ±2.0 vox (reasonable start)
```

**Formula:** `real_noise = exp(log_std) × action_scale`

---

## PPO Config (skrl_ppo_cfg.yaml)

```yaml
agent:
  rollouts: 64           # 64 × num_envs samples per update
  mini_batches: 4
  discount_factor: 0.99
  lambda: 0.95
  learning_rate: 2.0e-04
  learning_rate_scheduler: KLAdaptiveLR
  learning_rate_scheduler_kwargs:
    kl_threshold: 0.016  # LR drops if KL > 0.016, rises if KL < 0.016/1.5
  ratio_clip: 0.2
  value_clip: 0.2
  grad_norm_clip: 1.0
  value_preprocessor: RunningStandardScaler
  entropy_loss_scale: 0.0   # correct for overfitting — want deterministic convergence
  time_limit_bootstrap: True  # CRITICAL — see note below
trainer:
  timesteps: 200000
```

**KLAdaptiveLR:** NOT a warmup schedule. LR adapts to keep policy KL divergence near kl_threshold. The "spike" pattern seen in previous run was: low KL early → LR rises → big update → high KL → LR drops back.

**`entropy_loss_scale: 0.0`:** Intentionally zero for the overfitting phase. Entropy bonus fights convergence — we want the policy to become deterministic and lock in a sweep pattern. The `log_std_parameter` already handles exploration naturally.

**`time_limit_bootstrap: True` (CRITICAL):** When an episode times out without reaching 85% coverage, the value target for the last step is:
- `False` (wrong): `target = r_last` — critic treats timeout as terminal, assumes V=0 after. If last reward=1.2 but the true state value is ~45.0 (many more scannable steps ahead), the critic trains toward 1.2 → massively undertrained → noisy advantages → policy can't learn direction.
- `True` (correct): `target = r_last + γ × V(s_last)` — bootstraps the value at timeout. Critic gets an accurate signal.

In the previous run (fancy-star-20), almost ALL episodes timed out → critic was constantly undertrained → advantages were noise → this was a major cause of failure.

---

## Last Training Run — fancy-star-20 (200k steps, FAILED)

**What went wrong:**

| Problem | Root cause | Fix applied |
|---|---|---|
| Agent stuck at boundary (x=248, z=280) from step 20k | `attenuation_Rc=100` too soft — boundary penalty only 0.5 vs center 1.0 | `Rc: 30` → boundary now 0.10 vs center 1.0 |
| Shadow gate wiping all reward randomly | `shadow_thresh=0.15` too strict — frequent -0.1 episodes | `thresh: 0.30` |
| `rc_mean ≈ 0` entire run | Agent not scanning new voxels (rc signal lost against ra+rs baseline ~1.2) | Rc fix restores gradient toward center |
| LR suppressed to 2e-5 (10× below configured) | High KL from unstable reward → KLAdaptiveLR kept reducing LR | Shadow + Rc fixes should reduce variance |
| Policy std barely moved 0.5→0.46 | Policy never became confident | Expected to improve with stable reward |

**Positive signals from last run:**
- `volume_fraction_max` DID reach 0.83 — some learning happened
- Value loss decreased 0.20→0.08 — critic was learning
- `volume_fraction` mean trend was upward 0.12→0.88 (from trough)

---

## Wandb Logging — Current Setup

**Single `wandb.log()` call in `_get_dones()`, fires only when `episode_done.any()`.**

Metrics logged at episode end:
| Metric | What it means | Healthy overfitting looks like |
|---|---|---|
| `episode_volume_fraction_mean` | Mean coverage of envs that just ended | Rising toward 0.85+ |
| `episode_volume_fraction_max` | Best coverage among envs that just ended | Should hit 0.85 first, eventually stable |
| `episode_terminated_frac` | Fraction of ended episodes that hit 0.85 (vs timeout) | Rises from 0 → 1.0 |
| `rc_episode_sum_mean/max` | Sum of rc over the episode = total fraction scanned | Should equal volume_fraction (consistency check) |
| `ra_mean` | Last-step attenuation reward (probe distance to center) | Should rise toward 0.9+ |
| `rs_mean` | Last-step shadow avoidance | Should be stable near 0.95-1.0 |
| `shadow_ok_frac` | Fraction of envs with acceptable shadow at last step | Should be near 1.0 |
| `reward_mean/max` | Last-step reward | Rises then declines as buffer fills (normal) |

**Note:** `reward_mean` declining while `volume_fraction` rises is CORRECT — rc drops to 0 when buffer fills so less reward per step, but more coverage achieved. The decline is a success signal.

---

## Expected Learning Curve for Successful Overfitting

```
1. ra_mean rises         → agent finds the target location
2. shadow_ok_frac → 1    → agent avoids ribs
3. rc_episode_sum rises  → agent starts scanning new voxels
4. value_loss drops      → critic understands return structure
5. policy_loss stable    → consistent policy updates
6. policy std decreases  → policy becomes deterministic
7. volume_fraction → 0.85+ → agent covers the sphere
8. episode_terminated_frac → 1.0 → every episode succeeds
```

---

## Changes Made in This Session (ready for next run)

1. `shadow_thresh: 0.15 → 0.30` (YAML)
2. `attenuation_Rc: 100 → 30` (YAML)
3. `min/max/initial_log_std` now controlled purely from YAML (wired in train.py)
4. All wandb logging moved to episode end (was mid-episode every 200 steps)
5. `rc_episode_sum` accumulator added — correct episode-level coverage metric
6. `reward_max`, `rc_episode_sum_mean/max`, `episode_terminated_frac` added to wandb
7. `vis_seg_map: False` (was True — slows training, crashes headless)
8. Fresh training (not from checkpoint — old policy stuck in boundary-hugging minimum)

---

## Patient / Volume Info

- Patient: **s0030**, `label_res=0.0015 m/vox` (1.5 mm/vox)
- Volume shape: 309×309×317 (X×Y×Z)
- Liver XZ range: X:116–250, Z:147–294
- Target sphere center: (206, 175, 226) — confirmed intercostal pose via teleop
- Sphere radius: 10 voxels = 15mm
- Liver fraction at (206,226): ~55% of Y-column

---

## How to Launch Training

```bash
cd ~/IsaacLab
PYTHONPATH=$HOME/ws/sonogym/SonoGym/source/spinal_surgery:$PYTHONPATH \
  ./isaaclab.sh -p ~/ws/sonogym/SonoGym/workflows/skrl/train.py \
  --task Isaac-robot-US-guidance-v0 \
  --num_envs 8 \
  --headless \
  --enable_cameras
```

Tensorboard logs: `~/IsaacLab/logs/skrl/US_guidance/<timestamp>_ppo_torch_PPO_US/`
```bash
tensorboard --logdir ~/IsaacLab/logs/skrl/US_guidance/<run_dir>
```
