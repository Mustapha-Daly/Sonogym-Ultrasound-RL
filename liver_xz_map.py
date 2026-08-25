"""Plot the XZ liver footprint for one patient and print its bounds + center.

Usage:
    python liver_xz_map.py
    python liver_xz_map.py s0030
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


PATIENT_ID = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SONOGYM_PATIENT_ID", "s0015")
LIVER_ID = 5
LABEL_MAP_PATH = (
    "/home/yue/ws/sonogym/SonoGym/source/spinal_surgery/spinal_surgery/"
    f"assets/data/HumanModels/selected_dataset_stl/{PATIENT_ID}/combined_label_map.nii.gz"
)

print(f"Loading label map for {PATIENT_ID} ...")
vol = nib.load(LABEL_MAP_PATH).get_fdata().astype(np.int32)
liver = vol == LIVER_ID

if not liver.any():
    raise RuntimeError(f"No liver voxels (label {LIVER_ID}) found for {PATIENT_ID}")

x_idx, y_idx, z_idx = np.where(liver)
x_min, x_max = int(x_idx.min()), int(x_idx.max())
y_min, y_max = int(y_idx.min()), int(y_idx.max())
z_min, z_max = int(z_idx.min()), int(z_idx.max())
# center = CENTROID (mean of the actual liver voxels), NOT the bbox midpoint.
# For the diagonal liver the two differ: bbox midpoint would give ~[183,·,220],
# the centroid gives the hand-verified [206,175,226]. Y is the DEPTH (→ 175).
center_x = int(round(float(x_idx.mean())))
center_y = int(round(float(y_idx.mean())))   # DEPTH — this is where 175 comes from
center_z = int(round(float(z_idx.mean())))

print(f"X range        : {x_min} - {x_max}")
print(f"Y range (depth): {y_min} - {y_max}")
print(f"Z range        : {z_min} - {z_max}")
print(f"Center (centroid) [x, DEPTH, z]: [{center_x}, {center_y}, {center_z}]")

# Liver fraction of each Y-column: at (x,z), what fraction of the DEPTH axis is
# liver. Dark = a probe there sees lots of liver (the real "good scan" positions).
liver_frac = liver.mean(axis=1).astype(np.float32)  # (X, Z), values 0..1

# "Liver zone" = bbox of columns with a meaningful amount of liver in depth.
# Raise ZONE_THRESH to tighten the box around the dense core.
ZONE_THRESH = 0.15
zx, zz = np.where(liver_frac >= ZONE_THRESH)
zone_x_min, zone_x_max = int(zx.min()), int(zx.max())
zone_z_min, zone_z_max = int(zz.min()), int(zz.max())
print(f"Liver zone (frac>={ZONE_THRESH})  x:[{zone_x_min},{zone_x_max}]  z:[{zone_z_min},{zone_z_max}]")

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(liver_frac.T, origin="lower", cmap="YlGn", aspect="auto")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Liver fraction of Y-column")
ax.set_title(f"{PATIENT_ID} — Liver fraction at each (x, z) probe position")
ax.set_xlabel("X voxel")
ax.set_ylabel("Z voxel")

# zoom to the liver region (+ margin)
m = 20
ax.set_xlim(x_min - m, x_max + m)
ax.set_ylim(z_min - m, z_max + m)

ax.add_patch(plt.Rectangle(
    (zone_x_min, zone_z_min), zone_x_max - zone_x_min, zone_z_max - zone_z_min,
    fill=False, edgecolor="red", linewidth=2,
    label=f"Liver zone x:[{zone_x_min},{zone_x_max}] z:[{zone_z_min},{zone_z_max}]",
))
ax.plot(center_x, center_z, "r*", markersize=16,
        label=f"Target center ({center_x}, {center_z})")
ax.legend(loc="lower left")

out_path = f"/home/yue/ws/sonogym/SonoGym/liver_xz_map_{PATIENT_ID}.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved -> {out_path}")
