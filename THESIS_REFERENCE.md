# Thesis Reference — Autonomous Liver US Scanning via RL (known-target-position branch)

This document distills the technical work from the `known-target-position` branch development session: design decisions, proofs, bugs found and fixed, experiments run, and their actual results. It picks up where `TRAINING_CONTEXT.md` leaves off (that file documents the earlier single-patient, fixed-target, blind-search overfitting phase — useful as historical/motivation context, but superseded by everything below).

---

## 1. Project evolution — from blind search to known-target-position

**Earlier phase (documented in `TRAINING_CONTEXT.md`):** single patient (s0030), one fixed target sphere, the agent had to *blindly search* the liver to find it, `episode_length=6000` steps, reward closely mirroring Bi et al. 2026's formula (`alpha1=1.0`, `alpha2=0.5`).

**Motivation for the branch:** the original (non-liver) SonoGym task hands its navigation target to the agent directly (via a precomputed trajectory, executed before any RL action) rather than making the agent learn to search — the RL-learned part there is scanning, not finding. This supports doing the same for the liver task: the harder, more important remaining problem is fully scanning the target around rib/bone occlusion, not finding it. So the target's position was added as a direct observation input, and the search-era reward machinery (`w_explore`, the visitation-grid exploration bonus) was disabled.

**Core design problem:** each of the (now 7) patients' raw label-map voxel coordinates mean nothing across patients — same voxel number, different anatomy, no correspondence. A single shared-weight network can't transfer "go toward the target" skill between patients if fed raw, patient-specific numbers. This motivated the normalization scheme in §2.

---

## 2. Coordinate frames and normalization

### 2.1 Four frames, one translation point

| Frame | What lives in it |
|---|---|
| **World** | Ground-truth simulator state (probe/patient rigid-body poses) |
| **Patient (body)** | `patient_xz_range`, `target_center_per_env`, `current_x_z_x_angle_cmd`, label-map voxel indices — the frame nearly everything is expressed in |
| **Probe's own instantaneous frame** | The raw action (`actions[0]`/`actions[1]` = tangential slide relative to the probe's *current* orientation) |
| **Image/pixel frame** | The rendered 2D ultrasound slice (not a spatial coordinate frame) |

**Only one translation actually happens per step**, and it's not done by the network — it's done by the environment, in `_pre_physics_step`:
```python
human_to_ee_rot_mat = matrix_from_quat(human_to_ee_quat)  # from live world poses, re-derived every step
dx_dz_human = actions[:,0]*human_to_ee_rot_mat[:,:,0] + actions[:,1]*human_to_ee_rot_mat[:,:,1]
cmd = torch.cat([dx_dz_human[:, [0, 2]], actions[:, 2:3]], dim=-1)
self.US_slicer.update_cmd(cmd)
```
This pattern (probe-frame action → patient-frame accumulator via a live-read rotation matrix) is **inherited, original SonoGym infrastructure** — confirmed identical in the sibling non-liver `robotic_US_guidance.py` task, not something introduced on this branch.

**Why no external reference marker is needed (simulation-specific):** the patient's body is a `RigidObject` asset spawned at a known, fixed pose; its true world pose is read directly from the simulator every step (`self.human.data.body_link_state_w`), not measured or estimated. Both the probe and the target resolve back to this one known reference. This is a simulation simplification — a real physical deployment would need genuine registration (surface scanning, landmark alignment, etc.), which this training system does not attempt to solve.

**`current_x_z_x_angle_cmd` is a dead-reckoned accumulator, not independently re-measured each step** (`update_cmd` just adds a delta and clamps). This doesn't drift because (a) it's correctly initialized at each reset from `patient_xz_init_range` (same frame), and (b) the *direction* of every increment is re-grounded to the true, live-read patient pose every step via the rotation matrix above — there is no error to accumulate.

### 2.2 Normalization — proof and worked example

`_normalize_pose` / `_normalize_target_position` (`robotic_US_guidance_liver.py`) do plain min-max scaling, **per patient**:
```python
normalized = (raw_value - min) / (max - min)
```

| axis | min | max | scope |
|---|---|---|---|
| x, z | `patient_xz_range[0]` | `patient_xz_range[1]` | **per-patient** |
| angle | `-3.14` | `3.14` | fixed, shared |
| roll | `-max_roll_adj` | `+max_roll_adj` | fixed, shared |
| depth (target only) | 0 (skin) | `effective_max_depth_voxels` | fixed, shared — division, not min-max |

`effective_max_depth_voxels = max_visible_depth_voxels − radius_voxels − pose_tilt_margin_voxels = 100 − 10 − 20 = 70` (patient-agnostic; derived once from probe/target geometry, not anatomy).

**Depth normalization went through one documented mistake, corrected during design**: v1 normalized against each patient's own raw-Y min/max — wrong, because raw `y` isn't a fixed physical depth (skin height varies across `(x,z)`). v2 (current) uses `depth_from_skin / effective_max_depth_voxels` instead — physically consistent across all patients because `label_res` (voxel→meters) is a single global constant (verified no per-patient override exists), so "70 voxels" means the same real depth (10.5 cm) for every patient by construction.

**Worked example (s0030, live config: `patient_xz_range: [[131,160],[240,285]]`):**
- Target at raw `(x=206, z=226)`, depth-from-skin=50: `target_norm = [(206−131)/109, 50/70, (226−160)/125] = [0.688, 0.714, 0.528]`
- Probe at raw `(x=195, z=210, angle=−0.5, roll=0.45)`: `probe_norm = [0.587, 0.400, 0.420, 0.821]`
- `target_rel = [target_x_norm − probe_x_norm, target_y_norm, target_z_norm − probe_z_norm] = [+0.101, 0.714, +0.128]`

**Sign convention (verified, common point of confusion):** positive means the target is still *ahead* on that axis (probe needs to move further that way); it is **not** "already passed." Overshooting flips the sign negative (checked numerically: moving the probe past the target's x from 195→230 flips `dx` from +0.101 to −0.220).

**What normalization does and doesn't fix:** it makes the *observation* patient-agnostic (`0.5` means the same relative thing for any patient). It does **not** fix the *action* side — `action_scale`/`max_action` are still fixed, raw-voxel constants shared by all 7 patients regardless of workspace size. This asymmetry is the subject of the still-open action-scale investigation (§5).

---

## 3. Architecture

- **`SharedModel`** (`skrl_actor_critic.py`): shared CNN backbone (image) + MLP branches for `pose` (12-dim, 3-frame history × [x,z,angle,roll]) and `target_rel` (3-dim, `[dx, depth, dz]`), concatenated (512+64+64=640) into policy/value heads.
- **`net_target` branch** sized `3→128→128→64`, matching the existing `QNet.net_pos` precedent in the same file for embedding a 3-value position — not an arbitrary choice.
- **Observation space**: `{"image": [6,200,150], "pose": [24], "target_rel": [3]}` (note: `pose` is `[24]` = 6 frames × 4 values, not 3 frames as an earlier stale comment claimed).
- **Action space**: `[tangent_x, tangent_y, angle, roll]`, continuous, clamped to `max_action` after scaling.
- **`observation.mode: 'seg'`**: the agent trains on raw segmentation labels, not synthesized realistic ultrasound images — `sim.us` (`'conv'` vs `'net'`, two different US-image synthesizers) is consequently inert for this task's current training, since `us_img_tensor` is only consumed under `observation.mode == 'US'`.
- **`observation.3D: False`**: correct for the current architecture — the CNN is `Conv2d`-based and the observation space declares flat 2D frames; flipping this would change per-frame shape without a corresponding network redesign (`Conv3d`), not a safe toggle.

---

## 4. Reward function — current formula and the tuning journey

```python
rt = w_coverage*rc + alpha1*ra + alpha2*rs + alpha_vis*rv + r_explore
reward = rt - liver_penalty - shadow_penalty - time_penalty
reward += r_end   # once per episode, on first crossing the coverage threshold
r_end = terminal_bonus_kend * (1.0 + alpha1/D + alpha2*P)
```

### 4.1 Live values and what changed from the paper/earlier phase

| param | earlier/paper | current | why |
|---|---|---|---|
| `alpha1` (attenuation `ra`) | 1.0 | **0.0** | paper's own ablation shows negligible effect on success; passive per-step term dwarfed scarce coverage reward over long episodes |
| `alpha2` (shadow `rs`) | 0.5 | **0.15** | same passive-term-dwarfing risk, tuned down |
| `w_explore` | (search-era feature) | **0.0** | obsolete once target position is directly observed |
| `w_coverage` | 5.0 → 7.0 | **5.0** | reduced once `alpha_vis` no longer needed to out-compete `w_explore` |
| `alpha_vis` | 2.0 → 2.5 → 4 | **0.5** | its escalation to 4 was specifically to out-compete `w_explore`'s pull (now gone); real measured contribution turned out negligible (`rv` ceiling ≈0.075, geometric — a 2D slice can only ever intersect a fraction of a 3D sphere) at *any* tested weight |
| `time_penalty` | 0.02 (@720-step episodes) | **0.04** (@360-step episodes) | recalibrated to preserve the same 72%-of-terminal-bonus living-cost ratio after halving episode length |
| `liver_penalty_k`, `shadow_penalty_k` | 0.12 / (dead code) | **0.0 / 0.0** | disabled; `shadow_penalty` was found computed-but-never-applied (real bug, now wired in at weight 0 rather than left dead) |
| `terminal_bonus_kend` | 20.0 | **20.0** | unchanged |
| success/termination threshold | — | **0.85** | (variable is stale-named `reached_95`, not an actual 95% check) |

### 4.2 Key empirical/mechanistic findings from the tuning process

- **The passive-reward-vs-scarce-coverage-reward tension** (documented pre-branch) is structural: `rc` is capped at ≤1.0 summed per episode by construction; `rs`/`rv` are paid every step regardless of progress. Confirmed with real wandb data: even a *reduced* `alpha_vis=0.5` still produced a `contrib_vis_rv_per_step` an order of magnitude smaller than initially assumed (~0.002–0.06, not the ~0.3 first guessed) — the term's real-world impact was negligible at any weight tested, correcting an earlier over-estimate made in this same session.
- **The terminal bonus (`r_end`) is the dominant driver of scanning-completion behavior, not the incremental coverage term.** With `alpha1=0`, `r_end = 20 + 3·P` (P = shadow-free fraction) — almost entirely a flat 20 for crossing 85% coverage, only ±3 modulated by shadow quality, and *not* gated by `w_coverage` at all. **Ablation confirming this**: setting `w_coverage=0` (zeroing the incremental coverage reward entirely) did *not* collapse performance — `run_episode_terminated_mean_live` still climbed smoothly to ~80%. Mechanistic explanation: `coverage_fraction` (used for termination) is a physical measurement (`scanned_target_mask`), independent of any reward weight; PPO's value function, conditioned on `target_rel`, learns to bootstrap toward the sparse terminal bonus via GAE/TD credit assignment even without dense per-step shaping — early "lucky" successes under a near-random policy get reinforced backward through the critic into increasingly purposeful behavior. **Follow-up ablation proposed (not yet run)**: zero *both* `w_coverage` and `terminal_bonus_kend` together — this should remove every coverage-linked reward path and is expected to produce genuine collapse, cleanly isolating the terminal bonus's necessity (as opposed to the incremental term's, which the first ablation showed is not necessary on its own).
- **Two real bugs found and fixed via wandb-log auditing** (not hypothetical — traced to root cause in IsaacLab's own source):
  1. `episode_reward_mean`/`episode_success_count`/etc. were computed inside `_get_dones()`, which IsaacLab's `env.step()` calls *before* `_get_rewards()` each step — meaning episode-completion bookkeeping read `self.total_reward` one step stale, missing the very step that ends the episode (often the one carrying the terminal bonus). Fixed by moving that bookkeeping to run after `_get_rewards()`'s `self.total_reward += reward`.
  2. `episode_success_count`/`episode_terminated_frac` are throttled to *one recorded outcome per env per round* (a round = until all N envs complete ≥1 episode since the last flush) — fast-cycling patients' repeated successes get silently discarded while waiting on slow patients, making these metrics look far worse than reality. Confirmed with real log data: a window with 27 real episodes / 20 real successes (74%) produced a round count of only 3/7 (43%) via this mechanism. Fixed by adding `run_episode_terminated_mean_live` (and a matching per-episode console print), built from unconditional, unthrottled running counters (`run_term_sum`/`run_done_count`) that count every episode exactly once.

---

## 5. The still-open issue: action-scale is not workspace-size-aware

**Finding:** `action_scale`/`max_action` for `tangent_x`/`tangent_y` are fixed, global, raw-voxel constants — identical for every patient, despite patient workspaces varying up to ~2× in real usable target area (erosion-derived: s0006 ≈7992 vox² vs. s0012 ≈17136 vox²).

**Evidence (via added `action/clamped_frac/{patient}/{axis}` and `action/mean_abs/{patient}/{axis}` wandb metrics):** each patient chronically saturates the action ceiling specifically on **whichever axis its own workspace is elongated along** — e.g. s0029 (x-elongated) saturates `tangent_x` near 1.0 repeatedly; s0010 (z-elongated) saturates `tangent_y` instead. Even the best-performing patient chronically saturates an axis — meaning the ceiling is binding for everyone, not just the weak patients.

**Recommendation (not yet implemented):** scale `max_action` per patient (or per axis) relative to that patient's own span, analogous to the observation-side normalization in §2.2, rather than shrinking the (already correctly tightened) workspace bounds further.

**A related, separate change that *was* implemented and tested:** rotation authority (`angle`/`roll`) was originally doubled relative to the reference SonoGym implementation (`scale/max_action = [2,2,0.2,0.2]/[4,4,0.4,0.4]` vs. reference `[2,2,0.1,0.1]/[4,4,0.2,0.2]`, noting the reference is a 3-DOF task with no roll axis at all — this task added roll). Reverting rotation to the reference's finer values, tested as a controlled fresh-vs-fresh comparison (identical config otherwise):

| | old (doubled) rotation | new (reference) rotation |
|---|---|---|
| `terminated_mean` (true, run-end) | 0.8010 | 0.7995 (tied, within noise) |
| `volume_fraction_mean` (true, run-end) | 0.7919 | **0.8287** |

Modest, real, reproducible edge for the finer rotation setting (confirmed across three independent chart views: `episode_volume_fraction_mean`, per-patient breakdowns, `contrib_coverage_rc_per_episode`) — success rate unaffected, coverage meaningfully higher, no mid-training dip that the doubled-rotation run showed. (`episode_reward_mean` differed 5.6× between these two runs but was ruled *not* interpretable as a real effect — traced to a likely reward-accounting mismatch between when each run was launched relative to the ordering-bug fix in §4.2, not a genuine behavioral difference.)

---

## 6. Patient roster and workspace tightening

Active roster (current): `s0030, s0028, s0006, s0010, s0012, s0029, s0038` (7; three additional patients — s0004, s0015, s0024 — were tried and dropped after underperforming).

**`patient_xz_range` was tightened per patient** from arbitrary/generous bounds down to each patient's real erosion-valid target area plus a small buffer, verified by direct computation against the cached erosion data (all 7 patients confirmed to fully contain their real target-placement area with 0–16 voxel margins). One real overflow bug found and fixed in the process: s0030's erosion-valid area extended slightly *outside* its previously-declared `patient_xz_range` on two sides (confirmed 0 margin post-fix — technically passing but worth a larger buffer for robustness). Note: `target_volume.liver_x_range`/`liver_z_range` (both global and per-patient config fields) are **dead config** — grepped, zero references anywhere in the codebase; actual target placement is driven entirely by the label-map erosion process, independent of these fields.

---

## 7. Open items for future work

1. Per-patient/per-axis `action_scale` normalization (§5) — best-evidenced, not-yet-implemented lever.
2. Second reward ablation: zero both `w_coverage` and `terminal_bonus_kend` together (§4.2) to isolate the terminal bonus's necessity cleanly.
3. Staged success-threshold increase (0.85 → 0.90, checkpoint-compatible since it doesn't touch network/action shape) — motivated by the observation that many episodes already naturally exceed 0.90 coverage before the 0.85 threshold cuts them off.
4. `entropy_loss_scale=0.001` — `Policy/Standard deviation` has declined only slowly and steadily across multiple full runs (never plateaued, but never dropped sharply either); worth revisiting if convergence remains slow over longer runs.
5. s0006 showed real, unexplained oscillation in per-patient coverage in one run (ruled out: init-range containment mismatch) — not reproduced or root-caused, worth watching in future runs before concluding it's noise vs. a real effect (e.g. shared-network interference between patients).

---

## 8. Key file map

| File | Role |
|---|---|
| `robotic_US_guidance_liver.py` | Main env: observations, reward, coverage tracking, wandb logging |
| `cfgs/robotic_US_guidance.yaml` | Task config: reward weights, action scale, episode length, target volume, patient roster |
| `cfgs/patient_profiles.yaml` | Per-patient overrides: `patient_xz_range`, `center_voxel`, pose |
| `agents/skrl_ppo_cfg.yaml` | PPO hyperparameters, network layer sizes, `timesteps` |
| `lab/agents/skrl_actor_critic.py` | `SharedModel` — CNN + pose/target MLP branches, policy/value heads |
| `workflows/skrl/train.py` / `play.py` | Training / inference entry points |
