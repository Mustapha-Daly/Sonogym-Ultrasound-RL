"""
Liver fraction XZ map for s0030.
For each (x, z) position on the patient surface, computes the fraction of
the Y-column (depth axis) that contains liver (label 5).
Produces a 2D heatmap showing where the probe should go for best liver visibility.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LABEL_MAP_PATH = (
    "/home/yue/ws/sonogym/SonoGym/source/spinal_surgery/spinal_surgery/"
    "assets/data/HumanModels/selected_dataset_stl/s0030/combined_label_map.nii.gz"
)

LIVER_ID   = 5
BONE_IDS   = [13, 15]       # ribs / vertebra
COSTAL_ID  = 117            # costal cartilage (used to mark intercostal overlap)
TARGET_CENTER = (206, 226)  # (x, z) of our target sphere center

# XZ sweep range — intercostal overlap zone from check_patient_ranges.py
X_MIN, X_MAX = 116, 251
Z_MIN, Z_MAX = 147, 295
STEP = 2   # sample every 2 voxels for speed

print("Loading label map …")
vol = nib.load(LABEL_MAP_PATH).get_fdata().astype(np.int32)
print(f"Volume shape: {vol.shape}  (X×Y×Z)")

xs = np.arange(X_MIN, X_MAX, STEP)
zs = np.arange(Z_MIN, Z_MAX, STEP)

liver_map  = np.zeros((len(xs), len(zs)), dtype=np.float32)
bone_map   = np.zeros_like(liver_map)
shadow_map = np.zeros_like(liver_map)   # bone blocking = above bone, no liver below

for xi, x in enumerate(xs):
    for zi, z in enumerate(zs):
        col = vol[x, :, z]
        total = np.sum(col > 0)
        if total == 0:
            continue
        liver_map[xi, zi] = np.sum(col == LIVER_ID) / total

print("Done. Plotting …")

THRESH = 0.10  # minimum liver fraction to count as "liver present"

fig, ax = plt.subplots(figsize=(8, 7))
fig.suptitle("s0030 — Liver fraction at each (x, z) probe position", fontsize=13)

im = ax.imshow(
    liver_map.T,
    origin="lower",
    aspect="auto",
    extent=[X_MIN, X_MAX, Z_MIN, Z_MAX],
    cmap="YlGn",
    vmin=0, vmax=liver_map.max(),
)
plt.colorbar(im, ax=ax, label="Liver fraction of Y-column")
ax.set_xlabel("X voxel")
ax.set_ylabel("Z voxel")

# bounding box of positions where liver fraction > THRESH
liver_present = liver_map > THRESH
xi_vals = np.where(liver_present.any(axis=1))[0]
zi_vals = np.where(liver_present.any(axis=0))[0]
x_lo, x_hi = xs[xi_vals[0]],  xs[xi_vals[-1]]
z_lo, z_hi = zs[zi_vals[0]],  zs[zi_vals[-1]]

rect = mpatches.Rectangle(
    (x_lo, z_lo), x_hi - x_lo, z_hi - z_lo,
    linewidth=2, edgecolor="red", facecolor="none",
    label=f"Liver zone x:[{x_lo},{x_hi}] z:[{z_lo},{z_hi}]"
)
ax.add_patch(rect)
ax.plot(*TARGET_CENTER, "r*", markersize=14, label=f"Target center {TARGET_CENTER}")
ax.legend(fontsize=9)

plt.tight_layout()
out_path = "/home/yue/ws/sonogym/SonoGym/liver_xz_map.png"
plt.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")

# ── Liver zone range ────────────────────────────────────────────────────────
print(f"\nLiver present (>{THRESH*100:.0f}% of Y-column) at:")
print(f"  X range : {x_lo} – {x_hi}")
print(f"  Z range : {z_lo} – {z_hi}")

# ── Stats around target center ───────────────────────────────────────────────
tx, tz = TARGET_CENTER
xi_t = np.argmin(np.abs(xs - tx))
zi_t = np.argmin(np.abs(zs - tz))
print(f"\nAt target center ({tx}, {tz}): liver fraction = {liver_map[xi_t, zi_t]*100:.1f}%")

# ── Top-10 positions ─────────────────────────────────────────────────────────
flat = liver_map.flatten()
top_idx = np.argsort(flat)[::-1][:10]
print("\nTop-10 (x, z) by liver fraction:")
print(f"{'x':>6}  {'z':>6}  {'liver%':>8}")
for idx in top_idx:
    xi, zi = np.unravel_index(idx, liver_map.shape)
    print(f"{xs[xi]:6d}  {zs[zi]:6d}  {liver_map[xi,zi]*100:7.1f}%")
