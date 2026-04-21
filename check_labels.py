import nibabel as nib
import numpy as np

path = "/home/yue/ws/sonogym/SonoGym/source/spinal_surgery/spinal_surgery/assets/data/HumanModels/selected_dataset_stl/s0024/combined_label_map.nii.gz"

img = nib.load(path)
data = img.get_fdata().astype(int)

labels, counts = np.unique(data, return_counts=True)

print("Labels present in the volume:\n")

for l, c in zip(labels, counts):
    print(f"Label {l}: {c} voxels")
