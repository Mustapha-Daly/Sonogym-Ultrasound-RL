import nibabel as nib
import numpy as np
from ruamel.yaml import YAML

path = "/home/yue/ws/sonogym/SonoGym/source/spinal_surgery/spinal_surgery/assets/data/HumanModels/selected_dataset_stl/s0010/combined_label_map.nii.gz"
yaml_path = "/home/yue/ws/sonogym/SonoGym/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml"

data = nib.load(path).get_fdata().astype(int)
converted = data.copy()

label_convert_map = YAML().load(open(yaml_path, "r"))

for key, value in label_convert_map.items():
    converted[converted == int(key)] = int(value)

raw_liver_label = 5

print("Raw liver voxels:", np.sum(data == raw_liver_label))
print("Converted labels present:", np.unique(converted))
print("Converted voxel count still labeled 5:", np.sum(converted == 5))
print("Converted voxel count labeled 2:", np.sum(converted == 2))

