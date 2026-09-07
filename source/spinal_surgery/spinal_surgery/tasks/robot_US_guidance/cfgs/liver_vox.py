import nibabel as nib, numpy as np

PATIENTS = ['s0030', 's0004', 's0006', 's0010', 's0012', 's0015', 's0024', 's0028', 's0029', 's0038']
LIVER_LABEL = 5   # matches LIVER_LABEL_ID in robotic_US_guidance_liver.py:245
SKIN_LABEL = 12   # matches us_cfg.yaml surface_to_label_map: {'skin': 12, ...}
EFFECTIVE_MAX_DEPTH_VOX = 90  # current physical visibility budget (100 nominal - 10 radius margin)

for pid in PATIENTS:
    path = f"/home/yue/ws/sonogym/SonoGym/source/spinal_surgery/spinal_surgery/assets/data/HumanModels/selected_dataset_stl/{pid}/combined_label_map.nii.gz"
    lm = nib.load(path).get_fdata()
    X, Y, Z = lm.shape

    liver = (lm == LIVER_LABEL)
    skin = (lm == SKIN_LABEL)

    y_idx = np.arange(Y).reshape(1, Y, 1)
    # skin surface per (x,z) column: HIGHEST y where skin exists (shallow = high y,
    # same convention as construct_highest_y_array / the erosion code)
    skin_y = np.where(skin, y_idx, -1).max(axis=1)                  # (X, Z)
    # deepest liver voxel per column: LOWEST y where liver exists
    liver_y_min = np.where(liver, y_idx, 10**9).min(axis=1)         # (X, Z)

    has_liver = liver.any(axis=1)
    has_both = has_liver & (skin_y >= 0) & (liver_y_min < 10**9)
    if not has_both.any():
        print(f"{pid}: no columns with both skin and liver found, skipping")
        continue

    depth_from_skin = (skin_y - liver_y_min)[has_both]  # how deep the FARTHEST liver point is, per column
    frac_beyond_budget = float((depth_from_skin > EFFECTIVE_MAX_DEPTH_VOX).mean())

    print(f"{pid}: liver's deepest point per column ranges {depth_from_skin.min():.0f}-{depth_from_skin.max():.0f} "
          f"voxels below skin (mean {depth_from_skin.mean():.1f}) | "
          f"{frac_beyond_budget*100:.1f}% of (x,z) columns have liver extending past the "
          f"{EFFECTIVE_MAX_DEPTH_VOX}-voxel visibility budget (genuinely unreachable there, not a training issue)")
