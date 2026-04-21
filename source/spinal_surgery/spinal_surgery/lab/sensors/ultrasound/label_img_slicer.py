from spinal_surgery.lab.kinematics.surface_motion_planner import SurfaceMotionPlanner
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from isaaclab.utils.math import (
    transform_points,
    subtract_frame_transforms,
    matrix_from_quat,
)


class LabelImgSlicer(SurfaceMotionPlanner):
    """
    Label/CT slicer with auto-switch between:
      - planar (linear probe) slicing
      - convex (fan) slicing with scan-converted Cartesian image (fan-shaped mask)
    """

    def __init__(
        self,
        label_maps,
        ct_maps,
        human_list,
        num_envs,
        x_z_range,
        init_x_z_x_angle,
        device,
        label_convert_map,
        img_size,
        img_res,
        img_thickness=1,
        roll_adj=0.0,
        label_res=0.0015,
        max_distance=0.12,   #0.015,  # [m]
        body_label=120,
        height=0.1,
        height_img=0.16,
        visualize=True,
        plane_axes={"h": [0, 0, 1], "w": [1, 0, 0]},
    ):
        super().__init__(
            label_maps,
            human_list,
            num_envs,
            x_z_range,
            init_x_z_x_angle,
            device,
            roll_adj,
            label_res,
            body_label,
            height,
            height_img,
            visualize,
            plane_axes,
        )

        self.img_size = img_size  # [W, H] in your convention
        self.img_res = img_res
        self.img_thickness = img_thickness
        self.max_distance = max_distance
        self.img_real_size = [img_size[0] * img_res, img_size[1] * img_res]
        self.height_img = height_img

        # CT volumes
        self.ct_maps = [
            torch.tensor(ct_map, dtype=torch.int32, device=device) for ct_map in ct_maps
        ]

        # output tensors (N, W, H, E)
        self.label_img_tensor = torch.zeros(
            (self.num_envs, self.img_size[0], self.img_size[1], self.img_thickness),
            dtype=torch.uint8,
            device=self.device,
        )
        self.ct_img_tensor = torch.zeros(
            (self.num_envs, self.img_size[0], self.img_size[1], self.img_thickness),
            dtype=torch.int32,
            device=self.device,
        )

        # -----------------------------
        # Planar grid (original)
        # -----------------------------
        self.x_grid, self.z_grid, self.y_grid = torch.meshgrid(
            torch.arange(self.img_size[0], device=self.device) - self.img_size[0] // 2,
            torch.arange(self.img_size[1], device=self.device),
            torch.arange(self.img_thickness, device=self.device) - self.img_thickness // 2,
            indexing="ij",
        )
        self.img_coords = (
            torch.stack([self.x_grid, self.y_grid, self.z_grid], dim=-1)
            .reshape((-1, 3))
            .float()
            * img_res
        )  # (P, 3) meters in EE frame

        # smoothing kernel (kept)
        self.kernel = self.gaussian_kernel()

        # bounds for clamping (per env -> per human type)
        self.env_to_human_inds = (
            torch.arange(self.num_envs, device=self.device) % self.n_human_types
        )
        self.label_img_shapes = [label_map.shape for label_map in self.label_maps]
        self.label_img_shapes = torch.tensor(self.label_img_shapes, device=self.device)
        self.coords_upper_bound = (
            self.label_img_shapes[self.env_to_human_inds, :] - 1
        ).reshape((-1, 1, 3))

        # -----------------------------
        # Convex parameters (safe defaults; USSlicer can overwrite)
        # -----------------------------
        self.probe_type = getattr(self, "probe_type", "linear")  # "linear" | "convex"
        self.fan_angle_deg = getattr(self, "fan_angle_deg", 60.0)  # degrees
        self.max_depth = getattr(self, "max_depth", 0.12)  # meters

        # cache holder used by USSlicer.slice_rand_maps
        self.human_img_coords = None

    # ---------------------------------------------------------
    # Shared helper: EE points (meters) -> human voxel coords
    # ---------------------------------------------------------
    def get_human_img_coords(
        self,
        img_coords,  # (P,3) meters in EE frame
        world_to_human_pos,
        world_to_human_quat,
        world_to_ee_pos,
        world_to_ee_quat,
    ):
        human_to_ee_pos, human_to_ee_quat = subtract_frame_transforms(
            world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat
        )
        human_to_ee_rot = matrix_from_quat(human_to_ee_quat)

        # keep normal consistent (your original logic)
        normal_drcts = human_to_ee_rot[:, :, 2]

        # image plane origin offset from EE along normal (you used this originally)
        human_to_img_pos = human_to_ee_pos + self.height_img * normal_drcts

        # transform points into human frame (meters)
        human_pts = transform_points(img_coords, human_to_img_pos, human_to_ee_quat)  # (N,P,3)

        # meters -> voxel coords
        human_vox = human_pts / self.label_res

        # clamp
        human_vox = torch.clamp(
            human_vox,
            torch.zeros_like(human_vox, device=self.device),
            max=self.coords_upper_bound,
        )
        return human_vox

    # ---------------------------------------------------------
    # Planar slicing (corrected: computes coords first)
    # ---------------------------------------------------------
    def slice_label_img_planar(
        self, world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat
    ):
        # IMPORTANT: compute planar coords here
        human_vox = self.get_human_img_coords(
            self.img_coords,
            world_to_human_pos,
            world_to_human_quat,
            world_to_ee_pos,
            world_to_ee_quat,
        )
        # IMPORTANT: keep for USSlicer.slice_rand_maps
        self.human_img_coords = human_vox

        W, H, E = self.img_size[0], self.img_size[1], self.img_thickness

        for i in range(self.n_human_types):
            inds = slice(i, None, self.n_human_types)

            #coords = human_vox[inds].long()
            coords = torch.round(human_vox[inds]).long()
            sampled_label = self.label_maps[i][coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]]
            sampled_ct = self.ct_maps[i][coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]]

            self.label_img_tensor[inds] = sampled_label.reshape(-1, W, H, E).to(torch.uint8)
            self.ct_img_tensor[inds] = sampled_ct.reshape(-1, W, H, E)
            
        for b in range(self.label_img_tensor.shape[0]):
            for e in range(self.label_img_tensor.shape[3]):
                slice_2d = self.label_img_tensor[b, :, :, e]
                self.label_img_tensor[b, :, :, e] = self.bilateral_filter_pytorch(slice_2d)

        self.check_collision(self.label_img_tensor, self.ct_img_tensor)

    # ---------------------------------------------------------
    # Convex slicing (scan-converted Cartesian fan)
    # ---------------------------------------------------------
    def slice_label_img_convex(
        self, world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat
    ):
        device = self.device
        W = self.img_size[0]
        H = self.img_size[1]
        E = self.img_thickness

        fan_angle = float(self.fan_angle_deg) * np.pi / 180.0  # rad
        max_depth = float(self.max_depth)  # meters

        # Cartesian image grid (x,z) where z is depth [0..max_depth]
        half_width = max_depth * np.tan(fan_angle / 2.0)  # meters
        x_lin = torch.linspace(-half_width, half_width, W, device=device)  # (W,)
        probe_radius = 0.04  # meters (adjust)
        z_lin = torch.linspace(0.0, max_depth, H, device=device)

        x_grid, z_grid = torch.meshgrid(x_lin, z_lin, indexing="ij")  # (W,H)
        z_shifted = z_grid + probe_radius

        theta = torch.atan2(x_grid, z_grid + 1e-9)  # (W,H)
        r = torch.sqrt(x_grid * x_grid + z_shifted * z_shifted)           # (W,H)

        fan_mask_2d = (torch.abs(theta) <= (fan_angle / 2.0)) & (r >= probe_radius) & (r <= probe_radius + max_depth)  # (W,H)

        # probe-frame points (x,y,z) meters: z forward(depth), x lateral, y elevation
        if E > 1:
            e = (torch.arange(E, device=device) - E // 2).float() * self.img_res  # meters
            x3 = x_grid.unsqueeze(-1).expand(W, H, E)
            z3 = z_grid.unsqueeze(-1).expand(W, H, E)
            y3 = e.view(1, 1, E).expand(W, H, E)
            probe_pts = torch.stack([x3, y3, z3], dim=-1).reshape(-1, 3)  # (P,3)
            fan_mask = fan_mask_2d.unsqueeze(-1).expand(W, H, E).reshape(-1)  # (P,)
        else:
            y0 = torch.zeros_like(x_grid)
            probe_pts = torch.stack([x_grid, y0, z_grid], dim=-1).reshape(-1, 3)  # (P,3)
            fan_mask = fan_mask_2d.reshape(-1)  # (P,)

        # Transform to human voxel coords
        human_vox = self.get_human_img_coords(
            probe_pts,
            world_to_human_pos,
            world_to_human_quat,
            world_to_ee_pos,
            world_to_ee_quat,
        )  # (N,P,3)

        # IMPORTANT: keep for USSlicer.slice_rand_maps
        self.human_img_coords = human_vox

        for i in range(self.n_human_types):
            inds = slice(i, None, self.n_human_types)
            #coords = human_vox[inds].long()
            coords = torch.round(human_vox[inds]).long()

            sampled_label = self.label_maps[i][coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]]
            sampled_ct = self.ct_maps[i][coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]]

            # Apply fan mask: outside sector -> 0
            sampled_label = sampled_label * fan_mask.unsqueeze(0).to(sampled_label.dtype)
            sampled_ct = sampled_ct * fan_mask.unsqueeze(0).to(sampled_ct.dtype)

            if E > 1:
                self.label_img_tensor[inds] = sampled_label.reshape(-1, W, H, E).to(torch.uint8)
                self.ct_img_tensor[inds] = sampled_ct.reshape(-1, W, H, E)
            else:
                self.label_img_tensor[inds, :, :, 0] = sampled_label.reshape(-1, W, H).to(torch.uint8)
                self.ct_img_tensor[inds, :, :, 0] = sampled_ct.reshape(-1, W, H)
                
        for b in range(self.label_img_tensor.shape[0]):
            for e in range(self.label_img_tensor.shape[3]):
                slice_2d = self.label_img_tensor[b, :, :, e]
                self.label_img_tensor[b, :, :, e] = self.bilateral_filter_pytorch(slice_2d)
                
        self.check_collision(self.label_img_tensor, self.ct_img_tensor)

    # ---------------------------------------------------------
    # Public API used by USSlicer.slice_US
    # ---------------------------------------------------------
    def slice_label_img(
        self, world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat
    ):

        # refresh if USSlicer overwrote after init
        self.probe_type = getattr(self, "probe_type", "linear")
        self.fan_angle_deg = getattr(self, "fan_angle_deg", 60.0)
        self.max_depth = getattr(self, "max_depth", float(self.max_depth))

        if self.probe_type == "convex":
            self.slice_label_img_convex(
                world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat
            )
        else:
            self.slice_label_img_planar(
                world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat
            )

    # ---------------------------------------------------------
    # Collision / distance utilities
    # ---------------------------------------------------------
    def get_distances_from_label_img(self, label_img_tensor):
        # label_img_tensor: (B,W,H,E)
        first_nonzero = torch.argmax((label_img_tensor > 0).int(), dim=2)  # (B,W,E)
        return torch.amin(first_nonzero, dim=(1, 2))  # (B,)

    def check_collision(self, label_img_tensor, ct_img_tensor):
        first_nonzero = self.get_distances_from_label_img(label_img_tensor)
        self.no_collide = first_nonzero > self.max_distance / self.label_res
        label_img_tensor[self.no_collide] = 0
        ct_img_tensor[self.no_collide] = 0

    # ---------------------------------------------------------
    # Smoothing helpers (kept)
    # ---------------------------------------------------------
    def gaussian_kernel(self, size=9, sigma=5.0):
        x = torch.arange(size).float() - size // 2
        y = x[:, None]
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel.view(1, 1, size, size).to(self.device)

    def bilateral_filter_pytorch(self, seg_tensor):
        """
        seg_tensor: (W, H)
        returns:    (W, H)
        """
        seg_tensor = seg_tensor.to(torch.int64)
        smoothed_seg = seg_tensor.clone()

        unique_labels = torch.unique(seg_tensor)

        for label in unique_labels:
            if int(label.item()) == 0:
                continue

            # --- FIX: make it 4D ---
            mask = (seg_tensor == label).float().unsqueeze(0).unsqueeze(0)  # (1,1,W,H)

            smoothed = F.conv2d(mask, self.kernel, padding=self.kernel.shape[-1] // 2)

            smoothed = smoothed[0, 0]  # back to (W,H)

            smoothed_seg[smoothed > 0.5] = label

        return smoothed_seg.to(torch.uint8)
    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    def visualize(self, key, first_n=10):
        first_n = min(first_n, self.num_envs)

        if key == "seg":
            combined_img = self.label_img_tensor[:first_n, :, :, 0].reshape(
                (first_n * self.img_size[0], self.img_size[1])
            )
            combined_img_np = combined_img.cpu().numpy()

            plt.figure(1, figsize=(first_n * 2, 3))
            plt.clf()
            #denom = np.max(combined_img_np) + 1e-8
            img = combined_img_np
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            plt.imshow(img.T, cmap="gray")
            #plt.imshow((combined_img_np.T / denom * 255).astype(np.uint8), cmap="gray")
            plt.pause(0.0001)

        if key == "CT":
            combined_ct = self.ct_img_tensor[:first_n, :, :, 0].reshape(
                (first_n * self.img_size[0], self.img_size[1])
            )
            combined_ct_np = combined_ct.cpu().numpy()

            plt.figure(2, figsize=(first_n * 2, 3))
            plt.clf()
            denom = np.max(combined_ct_np) + 1e-8
            plt.imshow((combined_ct_np.T / denom * 255).astype(np.uint8), cmap="gray")
            plt.pause(0.0001)

        return

