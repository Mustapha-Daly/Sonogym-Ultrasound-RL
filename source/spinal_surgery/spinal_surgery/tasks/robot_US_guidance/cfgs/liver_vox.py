import nibabel as nib, numpy as np
lm = nib.load("/home/yue/ws/sonogym/SonoGym/source/spinal_surgery/spinal_surgery/assets/data/HumanModels/selected_dataset_stl/s0030/combined_label_map.nii.gz").get_fdata()
xz = (lm == 5).any(axis=1)            # collapse depth(Y) → (X,Z) liver footprint
xs, zs = np.where(xz)
print("liver X:", xs.min(), xs.max(), " Z:", zs.min(), zs.max())
print("liver at (141,192)?", bool(xz[141, 192]))
for z in range(zs.min(), zs.max()+1, 10):   # real x-extent per z (shows the taper)
    col = np.where(xz[:, z])[0]
    if len(col): print(f"z={z}: x {col.min()}-{col.max()}")
