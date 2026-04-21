from isaaclab.utils.math import (
    subtract_frame_transforms,
    transform_points,
)

#import pyvista as pv
try:
    import pyvista as pv
except ImportError as e:
    raise ImportError(
        "pyvista is required for human_frame_viewer.\n"
        "Install it using Isaac Sim python:\n"
        "  ./_isaac_sim/python.sh -m pip install pyvista"
    ) from e
import torch
import numpy as np


class HumanFrameViewer:
    """
    Human frame viewer for ultrasound guidance.

    Key fix:
    - Do NOT use self.visualize as a boolean flag because it collides with visualize() methods
      defined in subclasses (LabelImgSlicer / USSlicer). Use self.enable_visualization instead.
    """

    def __init__(
        self,
        label_maps,
        num_envs,
        device,
        label_res=0.0015,
        height_img=0.1,
        visualize=True,
        plane_axes={"h": [0, 0, 1], "w": [1, 0, 0]},
    ):
        self.device = device
        self.num_envs = int(num_envs)
        self.label_res = float(label_res)
        self.height_img = float(height_img)
        self.plane_axes = plane_axes

        # IMPORTANT: avoid name collision with methods called "visualize"
        self.enable_visualization = bool(visualize)

        self.label_maps = [
            torch.tensor(lm, dtype=torch.uint8, device=device) for lm in label_maps
        ]
        self.n_human_types = len(self.label_maps)

        # Axes
        self.w_axis = np.asarray(self.plane_axes["w"], dtype=np.float32)
        self.h_axis = np.asarray(self.plane_axes["h"], dtype=np.float32)

        self.w_axis_tensor = torch.tensor(self.w_axis, device=device)
        self.h_axis_tensor = torch.tensor(self.h_axis, device=device)

        self.np_linear_space = np.linspace(0.0, 10.0, 5).reshape((-1, 1))

        # Visualization objects
        self.p_list = []
        self.w_markers_list = []
        self.h_markers_list = []

        if not self.enable_visualization:
            return

        # Create one plotter per human type (as in your original intention)
        for i in range(self.n_human_types):
            p = pv.Plotter()
            p.add_volume(self.label_maps[i].cpu().numpy(), opacity=0.01)
            p.show_axes()
            p.show(interactive_update=True)
            self.p_list.append(p)

        # Add markers for each env (attach all to plotter 0 for simplicity)
        # If you want per-human-type plotters, you can map env->human_type.
        for _ in range(self.num_envs):
            ee_w_points = (
                self.np_linear_space * self.w_axis.reshape((1, -1))
                + (self.height_img / self.label_res) * self.h_axis.reshape((1, -1))
            )
            ee_h_points = (
                (self.np_linear_space + self.height_img / self.label_res)
                * self.h_axis.reshape((1, -1))
            )

            w_pv = pv.PolyData(ee_w_points)
            h_pv = pv.PolyData(ee_h_points)

            self.w_markers_list.append(w_pv)
            self.h_markers_list.append(h_pv)

            self.p_list[0].add_mesh(w_pv, color="red")
            self.p_list[0].add_mesh(h_pv, color="green")

    # ------------------------------------------------------------------

    def ee_pose_to_human_pcd(self, human_to_ee_pos, human_to_ee_quat):
        lin = torch.linspace(
            0.0, 10.0, 5, device=human_to_ee_pos.device
        ).reshape((-1, 1))

        ee_w = (
            lin * self.w_axis_tensor.reshape((1, -1))
            + (self.height_img / self.label_res) * self.h_axis_tensor.reshape((1, -1))
        )

        ee_h = (
            (lin + self.height_img / self.label_res)
            * self.h_axis_tensor.reshape((1, -1))
        )

        human_w = transform_points(
            ee_w, human_to_ee_pos / self.label_res, human_to_ee_quat
        )
        human_h = transform_points(
            ee_h, human_to_ee_pos / self.label_res, human_to_ee_quat
        )

        return human_w, human_h

    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_env_dim(pts: np.ndarray) -> np.ndarray:
        if pts.ndim == 2:
            return pts[None, :, :]
        return pts

    # ------------------------------------------------------------------

    def update_plotter(
        self,
        world_human_pos,
        world_human_quat,
        world_ee_pos,
        world_ee_quat,
    ):
        if not self.enable_visualization:
            return

        human_ee_pos, human_ee_quat = subtract_frame_transforms(
            world_human_pos,
            world_human_quat,
            world_ee_pos,
            world_ee_quat,
        )

        human_w, human_h = self.ee_pose_to_human_pcd(
            human_ee_pos, human_ee_quat
        )

        human_w = self._ensure_env_dim(human_w.cpu().numpy())
        human_h = self._ensure_env_dim(human_h.cpu().numpy())

        for i in range(self.num_envs):
            self.w_markers_list[i].points = human_w[i]
            self.h_markers_list[i].points = human_h[i]

        for p in self.p_list:
            p.update()

