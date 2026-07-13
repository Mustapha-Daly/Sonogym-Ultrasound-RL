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
        self.probe_radius = getattr(self, "probe_radius", 0.04)

        # Fan warp grid — built once lazily in _build_fan_warp_for_labels()
        self._fan_warp_grid    = None   # (H, W, 2) grid_sample coords
        self._fan_warp_mask_hw = None   # (H, W)    fan boolean mask
        self._convex_fan_mask_2d = None # alias kept for backward compat

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
        return_oob: bool = False,
    ):
        human_to_ee_pos, human_to_ee_quat = subtract_frame_transforms(
            world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat
        )
        human_to_ee_rot = matrix_from_quat(human_to_ee_quat)

        normal_drcts = human_to_ee_rot[:, :, 2]
        human_to_img_pos = human_to_ee_pos + self.height_img * normal_drcts

        human_pts = transform_points(img_coords, human_to_img_pos, human_to_ee_quat)  # (N,P,3)
        human_vox = human_pts / self.label_res

        if return_oob:
            # Detect which points fall outside the volume BEFORE clamping
            oob = (
                (human_vox < 0).any(dim=-1)                            # (N,P)
                | (human_vox > self.coords_upper_bound).any(dim=-1)
            )

        human_vox = torch.clamp(
            human_vox,
            torch.zeros_like(human_vox, device=self.device),
            max=self.coords_upper_bound,
        )

        if return_oob:
            return human_vox, oob
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
        self.last_sampled_coords_per_type = {}  # reset each step; one entry per human type

        for i in range(self.n_human_types):
            inds = slice(i, None, self.n_human_types)

            #coords = human_vox[inds].long()
            coords_f = torch.round(human_vox[inds])
            if torch.isnan(coords_f).any():
                print(f"[WARN] NaN in slicer coords (human_type={i}) — clamping to 0. Check IK/quaternion validity.")
                coords_f = torch.nan_to_num(coords_f, nan=0.0)
            coords = coords_f.long()
            # reshape to (B_i, W, H, E, 3) — B_i = envs belonging to this human type
            self.last_sampled_coords = coords.reshape(-1, W, H, E, 3)
            self.last_sampled_coords_per_type[i] = self.last_sampled_coords
            sampled_label = self.label_maps[i][coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]]
            sampled_ct = self.ct_maps[i][coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]]

            self.label_img_tensor[inds] = sampled_label.reshape(-1, W, H, E).to(torch.uint8)
            self.ct_img_tensor[inds] = sampled_ct.reshape(-1, W, H, E)
            
        for b in range(self.label_img_tensor.shape[0]):
            for e in range(self.label_img_tensor.shape[3]):
                slice_2d = self.label_img_tensor[b, :, :, e]
                #self.label_img_tensor[b, :, :, e] = self.bilateral_filter_pytorch(slice_2d)

        # check_collision is skipped here for the convex path; the fan mask
        # applied in slice_label_img already zeros everything outside the sector.
        # Calling it would blank the entire image when the probe lifts off the body,
        # causing the fan window to disappear or flicker.
        if getattr(self, "probe_type", "linear") != "convex":
            self.check_collision(self.label_img_tensor, self.ct_img_tensor)

    # ---------------------------------------------------------
    # NEW FUNCTION
    # Fan warp grid — built ONCE, applied every frame as a 2-D post-process
    # Identical philosophy to the original apply_convex_geometry:
    #   step 1: stable flat 3-D sampling (slice_label_img_planar)
    #   step 2: 2-D fan warp via grid_sample — zero probe-pose dependency
    # ---------------------------------------------------------
    def _build_fan_warp_for_labels(self):
        """
        Cartesian fan output + explicit sector mask 
        Output canvas (H rows × W cols) spans:
          rows  h → probe-frame depth  z = h * max_depth / (H-1)
          cols  w → probe-frame lateral x = (w - W//2) * x_step

        This canvas is then masked to keep only the sector defined by the
        probe geometry.  The warp grid depends purely on (H, W, fan_angle,
        probe_radius, max_depth) — zero probe-pose dependency.
        """
        device = self.device
        W = self.img_size[0]
        H = self.img_size[1]

        half_fan_rad = float(self.fan_angle_deg) * 0.5 * np.pi / 180.0
        probe_r      = float(self.probe_radius)
        max_d        = float(self.max_depth)
        near_field_m = float(
            getattr(self, "us_cfg", {}).get("probe", {}).get("near_field", 0.01)
            if hasattr(self, "us_cfg") else 0.01
        )
        inner_arc_r = probe_r + near_field_m   # distance from virtual apex → top arc
        outer_arc_r = probe_r + max_d          # distance from virtual apex → bottom arc

        # ── Cartesian output grid in probe frame ──────────────────────────
        # Lateral extent at the bottom arc; same formula as _build_convex_grid.
        x_half = outer_arc_r * np.sin(half_fan_rad)
        x_out  = torch.linspace(-x_half, x_half,  W, device=device)   # (W,)
        z_out  = torch.linspace(0.0,     max_d,    H, device=device)   # (H,)
        Z_out, X_out = torch.meshgrid(z_out, x_out, indexing="ij")     # (H, W)

        # ── Sector mask ───────────────────────────────────────────────────
        Z_shifted = Z_out + probe_r
        r_px      = torch.sqrt(X_out**2 + Z_shifted**2)
        theta_px  = torch.atan2(X_out, Z_shifted)
        fan_mask  = (
            (theta_px.abs() <= half_fan_rad)
            & (r_px >= inner_arc_r)
            & (r_px <= outer_arc_r)
        )  # (H, W) bool — purely geometric, probe-pose independent

        # ── Map fan output (x, z) → flat input normalised coords ─────────
        # Flat input (from slice_label_img_planar):
        #   col w_in → x = (w_in - W//2) * img_res   ∈ [-(W//2)*res, +(W//2)*res]
        #   row h_in → z = h_in * img_res             ∈ [0, (H-1)*res]
        # grid_sample align_corners=True: coord -1 → pixel 0, +1 → pixel N-1
        x_norm = (X_out / ((W // 2) * self.img_res)).clamp(-1.0, 1.0)
        z_norm = (2.0 * Z_out / ((H - 1) * self.img_res) - 1.0).clamp(-1.0, 1.0)

        grid = torch.stack([x_norm, z_norm], dim=-1)  # (H, W, 2)

        self._fan_warp_grid      = grid      # (H, W, 2)
        self._fan_warp_mask_hw   = fan_mask  # (H, W) bool
        self._convex_fan_mask_2d = fan_mask  # backward-compat alias

    def _apply_fan_warp(self):
        """
        Warp label_img_tensor and ct_img_tensor into the fan shape — single
        batched grid_sample call covering all environments and elevations at once.

        Tensor layout:
          label_img_tensor : (B, W, H, E)   W=lateral, H=depth, E=elevation
          fan_warp_grid    : (H_out, W_out, 2)  fan Cartesian output
          fan_warp_mask_hw : (H_out, W_out)
        """
        B, W, H, E = self.label_img_tensor.shape
        n = B * E  # flatten batch × elevation for a single grid_sample call

        # grid_sample needs input (n, 1, H_in, W_in)  and grid (n, H_out, W_out, 2)
        # label_img_tensor (B,W,H,E) → permute → (B,E,H,W) → reshape → (n,1,H,W)
        lbl = self.label_img_tensor.permute(0, 3, 2, 1).float()   # (B, E, H, W)
        lbl = lbl.reshape(n, 1, H, W)                              # (n, 1, H, W)

        grid_n = self._fan_warp_grid.unsqueeze(0).expand(n, -1, -1, -1)  # (n, H, W, 2)

        warped = F.grid_sample(lbl, grid_n, mode="nearest",
                               padding_mode="zeros", align_corners=True)  # (n, 1, H, W)
        warped = warped.squeeze(1)                                          # (n, H, W)
        warped = warped * self._fan_warp_mask_hw.float().unsqueeze(0)      # (n, H, W)

        # Keep the convex window visually stable under large probe tilts:
        # inside each fan column, fill shallow zero-valued gaps with the first
        # valid label in that column. If a whole fan column is empty, fall back
        # to the configured skin label so the fan outline stays intact.
        skin_label = int(
            getattr(self, "us_cfg", {}).get("surface_to_label_map", {}).get("skin", 12)
            if hasattr(self, "us_cfg")
            else 12
        )
        fan_mask_hw = self._fan_warp_mask_hw
        for i in range(n):
            for col in range(W):
                mask_col = fan_mask_hw[:, col]
                if not torch.any(mask_col):
                    continue

                label_col = warped[i, :, col]
                valid_col = mask_col & (label_col > 0)

                if torch.any(valid_col):
                    first_valid = int(torch.argmax(valid_col.to(torch.int32)).item())
                    fill_mask = mask_col.clone()
                    fill_mask[first_valid:] = False
                    if torch.any(fill_mask):
                        fill_value = label_col[first_valid]
                        label_col[fill_mask] = fill_value
                        warped[i, :, col] = label_col
                else:
                    label_col[mask_col] = skin_label
                    warped[i, :, col] = label_col

        warped_labels_hw = warped
        # reshape back: (n,H,W) → (B,E,H,W) → permute → (B,W,H,E)
        warped = warped_labels_hw.reshape(B, E, H, W).permute(0, 3, 2, 1)  # (B, W, H, E)
        self.label_img_tensor.copy_(warped.to(torch.uint8))

        # same for CT
        ct = self.ct_img_tensor.permute(0, 3, 2, 1).float().reshape(n, 1, H, W)
        ct_w = F.grid_sample(ct, grid_n, mode="nearest",
                             padding_mode="zeros", align_corners=True).squeeze(1)
        ct_w = ct_w * self._fan_warp_mask_hw.float().unsqueeze(0)
        for i in range(n):
            for col in range(W):
                mask_col = fan_mask_hw[:, col]
                if not torch.any(mask_col):
                    continue

                label_col = warped_labels_hw[i, :, col]
                valid_col = mask_col & (label_col > 0)
                if not torch.any(valid_col):
                    continue

                first_valid = int(torch.argmax(valid_col.to(torch.int32)).item())
                fill_mask = mask_col.clone()
                fill_mask[first_valid:] = False
                if torch.any(fill_mask):
                    ct_col = ct_w[i, :, col]
                    ct_col[fill_mask] = ct_col[first_valid]
                    ct_w[i, :, col] = ct_col
        ct_w = ct_w.reshape(B, E, H, W).permute(0, 3, 2, 1)
        self.ct_img_tensor.copy_(ct_w.to(torch.int32))

    # ---------------------------------------------------------
    # Public API used by USSlicer.slice_US
    # ---------------------------------------------------------
    def slice_label_img(
        self, world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat
    ):
        # Step 1: stable flat 3-D sampling — identical to the linear-probe path,
        # always fills the full (W, H) rectangle, never affected by probe pose.
        self.slice_label_img_planar(
            world_to_human_pos, world_to_human_quat, world_to_ee_pos, world_to_ee_quat
        )

        # Step 2 (convex only): 2-D fan warp — purely geometric, built once,
        # zero probe-pose dependency.  Same principle as apply_convex_geometry.
        if getattr(self, "probe_type", "linear") == "convex":
            if self._fan_warp_grid is None:
                self._build_fan_warp_for_labels()
            self._apply_fan_warp()

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
            plt.imshow(img.T[::-1], cmap="gray")
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
