
'''

organ_id_map.py is doing a best-effort match between:

the organ name from each .stl filename
the numeric label values inside combined_label_map.nii.gz

The mapping is not perfect, but it should be good enough for this purposes


'''
from pathlib import Path
from collections import Counter

import nibabel as nib
import numpy as np
import trimesh
from ruamel.yaml import YAML

PATIENT_DIR = Path("/home/yue/ws/sonogym/SonoGym/source/spinal_surgery/spinal_surgery/assets/data/HumanModels/selected_dataset_stl/s0015")
LABELMAP_PATH = PATIENT_DIR / "combined_label_map.nii.gz"
CONVERSION_YAML = Path("/home/yue/ws/sonogym/SonoGym/source/spinal_surgery/spinal_surgery/lab/sensors/cfgs/label_conversion.yaml")

IGNORE_PREFIXES = (
    "combined_",
    "standard_",
)
IGNORE_EXACT = {
    "body_highest_y_array",
    "body_lowest_y_array",
    "body_surface_normal_array",
    "body_surface_normal_array_highest",
}
IGNORE_SUFFIXES = (
    "_traj_L4",
)

def should_skip(stem: str) -> bool:
    if stem in IGNORE_EXACT:
        return True
    if any(stem.startswith(p) for p in IGNORE_PREFIXES):
        return True
    if any(stem.endswith(s) for s in IGNORE_SUFFIXES):
        return True
    return False

def sample_labels_from_points(points_vox: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    vox = np.rint(points_vox).astype(np.int64)

    valid = (
        (vox[:, 0] >= 0) & (vox[:, 0] < label_map.shape[0]) &
        (vox[:, 1] >= 0) & (vox[:, 1] < label_map.shape[1]) &
        (vox[:, 2] >= 0) & (vox[:, 2] < label_map.shape[2])
    )
    vox = vox[valid]
    if len(vox) == 0:
        return np.array([], dtype=np.int64)

    labels = label_map[vox[:, 0], vox[:, 1], vox[:, 2]]
    labels = labels[labels != 0]
    return labels.astype(np.int64)

def dominant_label_for_mesh(mesh: trimesh.Trimesh, label_map: np.ndarray):
    if mesh.vertices.shape[0] == 0:
        return None, []

    surface_pts, _ = trimesh.sample.sample_surface(mesh, 4000)
    centroid = mesh.centroid.reshape(1, 3)

    inner_pts = centroid + 0.92 * (surface_pts - centroid)
    centroid_pts = np.repeat(centroid, 200, axis=0)

    pts = np.vstack([surface_pts, inner_pts, centroid_pts])
    labels = sample_labels_from_points(pts, label_map)

    if labels.size == 0:
        return None, []

    counts = Counter(labels.tolist()).most_common(5)
    return counts[0][0], counts

def pick_organ_label(top_counts, ignore_labels={0, 120}):
    for label, count in top_counts:
        if label not in ignore_labels:
            return label
    return None

def main():
    label_map = nib.load(str(LABELMAP_PATH)).get_fdata().astype(np.int64)

    convert_map = YAML().load(CONVERSION_YAML.read_text())
    convert_map = {int(k): int(v) for k, v in convert_map.items()}

    stl_files = sorted(PATIENT_DIR.glob("*.stl"))

    print(f"Patient folder: {PATIENT_DIR}")
    print(f"Label map: {LABELMAP_PATH.name}")
    print()

    for stl_path in stl_files:
        stem = stl_path.stem
        if should_skip(stem):
            continue

        try:
            mesh = trimesh.load_mesh(stl_path, process=False)
            raw_id, top_counts = dominant_label_for_mesh(mesh, label_map)
        except Exception as e:
            print(f"{stem:35s} raw=? converted=? ERROR: {e}")
            continue

        if raw_id is None:
            print(f"{stem:35s} raw=? converted=? top=[]")
            continue

        organ_raw_id = pick_organ_label(top_counts)
        if organ_raw_id is None:
            organ_raw_id = raw_id

        converted_id = convert_map.get(organ_raw_id, organ_raw_id)
        print(
            f"{stem:35s} raw={organ_raw_id:<4d} converted={converted_id:<4d} top={top_counts}"
        )

if __name__ == "__main__":
    main()
