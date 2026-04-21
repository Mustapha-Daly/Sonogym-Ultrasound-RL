import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

path = "/home/yue/ws/sonogym/SonoGym/source/spinal_surgery/spinal_surgery/assets/data/HumanModels/selected_dataset_stl/s0024/combined_label_map.nii.gz"

data = nib.load(path).get_fdata().astype(int)
mask = np.isin(data, range(52,69))

z = data.shape[2]//2
fig, ax = plt.subplots()

def draw():
    ax.clear()
    ax.imshow(data[:,:,z], cmap="gray")
    ax.imshow(mask[:,:,z], cmap="autumn", alpha=0.6)
    ax.set_title(f"Slice {z}")
    ax.axis("off")
    fig.canvas.draw()

def key(e):
    global z
    if e.key=="right": z=min(z+1,data.shape[2]-1)
    if e.key=="left":  z=max(z-1,0)
    draw()

fig.canvas.mpl_connect("key_press_event", key)
draw()
plt.show()
