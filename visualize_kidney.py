import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

path = "/home/yue/ws/sonogym/SonoGym/source/spinal_surgery/spinal_surgery/assets/data/HumanModels/selected_dataset_stl/s0024/combined_label_map.nii.gz"

img = nib.load(path)
data = img.get_fdata().astype(int)

kidney_mask = np.isin(data, range(52, 69))

slice_index = data.shape[2] // 2
plt.imshow(kidney_mask[:, :, slice_index], cmap="hot")
plt.title("Kidney voxels (labels 52–68)")
plt.axis("off")
plt.show()
