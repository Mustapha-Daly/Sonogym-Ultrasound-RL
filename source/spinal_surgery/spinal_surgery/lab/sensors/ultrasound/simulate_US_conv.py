import numpy as np
import torch
import torch.nn.functional as F


class USSimulatorConv:
    """
    Conv-based ultrasound simulator.

    Key fixes:
    1) Store self.us_cfg so apply_convex_geometry can read fan_angle/depth params.
    2) grid_sample on CUDA does NOT support Byte/Int -> always cast input to float before grid_sample.
       If the input is a label map (integer), cast back after warping (nearest mode).
    3) Use a geometrically consistent grid for convex (sector) geometry:
       - output image is (H, W) where H is depth axis, W is lateral axis (angle).
       - grid samples from a *rectangular* input image (same H,W) assuming:
         x in [-1, 1] maps left-right of the probe
         z in [0, 1] maps shallow->deep (top->bottom)
    """

    def __init__(self, us_cfg, device) -> None:
        # ---- REQUIRED for your previous AttributeError ----
        self.us_cfg = us_cfg
        self.device = device

        system_params = us_cfg["system_params"]
        label_to_params_dict = us_cfg["label_to_ac_params_dict"]
        self.kernel_size = tuple(us_cfg["kernel_size"])
        self.E_S_ratio = us_cfg["E_S_ratio"]

        self.f = system_params["frequency"]
        self.I0 = system_params["I0"]
        self.e = system_params["element_size"]
        self.sx_E = system_params["sx_E"]
        self.sy_E = system_params["sy_E"]
        self.sx_B = system_params["sx_B"]
        self.sy_B = system_params["sy_B"]
        self.beta = system_params["TGC_beta"]
        self.beta_edge = system_params["TGC_edge"]

        self.n_I = system_params["noise_I"]
        self.n_mu0 = system_params["noise_mu0"]
        self.n_mu1 = system_params["noise_mu1"]
        self.n_s0 = system_params["noise_s0"]
        self.n_f = system_params["noise_f"]

        # Label -> acoustic params
        self.label_to_params_dict = label_to_params_dict
        for key, item in self.label_to_params_dict.items():
            self.label_to_params_dict[key] = torch.tensor(
                [
                    item["alpha"],
                    item["z"],
                    item["mu0"],
                    item["mu1"],
                    item["s0"],
                    item["Al"],
                    item["fl"],
                ],
                device=self.device,
            )

        self.PSF_E = self.compute_PSF_kernel(self.sx_E, self.sy_E)
        self.PSF_B = self.compute_PSF_kernel(self.sx_B, self.sy_B)

        self.if_large_scale_speckle = us_cfg["large_scale_speckle"]
        self.l_size = us_cfg["large_scale_resolution"]
        if self.if_large_scale_speckle:
            self.large_rand_param_map = torch.zeros((1, 1, 1, 3), device=self.device)

        self.param_map = torch.zeros((1, 1, 1, 3), device=self.device)

        # Optional convex params (safe defaults)
        # fan_angle in degrees; max_depth controls the sector mask (in normalized depth [0,1])
        self.fan_angle_deg = float(us_cfg.get("fan_angle", 80.0))
        self.max_depth = float(us_cfg.get("max_depth", 1.0))  # normalized [0..1] mask only

    def compute_PSF_kernel(self, sx, sy):
        center = (torch.tensor(self.kernel_size, device=self.device) - 1) / 2
        x_inds = torch.arange(0, self.kernel_size[0], device=self.device)
        y_inds = torch.arange(0, self.kernel_size[1], device=self.device)
        x_grid, y_grid = torch.meshgrid(x_inds, y_inds, indexing="ij")

        x = x_grid - center[0]
        y = y_grid - center[1]
        density_x = torch.exp(-0.5 * x**2 / sx**2) * torch.cos(2 * torch.pi * self.f * x)
        density_y = torch.exp(-0.5 * y**2 / sy**2)

        PSF_kernel = (density_x * density_y).reshape((1, 1) + self.kernel_size)
        return PSF_kernel

    def assign_params_map(self, label_img: torch.Tensor):
        if not label_img.shape == self.param_map.shape[:-1]:
            self.param_map = torch.zeros(label_img.shape + (7,), device=label_img.device)

        labels = torch.unique(label_img)
        for i in range(labels.shape[0]):
            label = int(labels[i].item())
            label_items = label_img == label
            if label in self.label_to_params_dict:
                self.param_map[label_items, :] = self.label_to_params_dict[label][:]
            else:
                # unknown label -> leave zeros (background-like)
                self.param_map[label_items, :] = 0.0

        return self.param_map

    def compute_attenuation_map(self, alpha_map: torch.Tensor):
        alpha_l_map = torch.cumsum(alpha_map, dim=1) * self.e
        return torch.exp(-alpha_l_map * self.f)

    def compute_edge_map(self, label_img: torch.Tensor):
        edge_map = torch.zeros(label_img.shape, device=label_img.device)
        pad_img = F.pad(label_img, (1, 1, 1, 1), "reflect")
        label_up = pad_img[:, :-2, 1:-1]
        label_down = pad_img[:, 1:-1, 1:-1]
        if_edge = torch.logical_not(label_up == label_down)
        edge_map[if_edge] = 1
        return edge_map

    def compute_ct_edge_map(self, ct_img: torch.Tensor):
        edge_map = torch.zeros(ct_img.shape, device=ct_img.device)
        pad_img = F.pad(ct_img, (1, 1, 1, 1), "reflect")
        label_up = pad_img[:, :-2, 1:-1]
        label_down = pad_img[:, 1:-1, 1:-1]
        if_edge = (label_up - label_down).abs() > (label_down.abs() * 0.3)
        edge_map[if_edge] = 1
        return edge_map

    def compute_image_gradient(self, img: torch.Tensor):
        pad_img = F.pad(img, (1, 1, 1, 1), "reflect")
        img_xm = pad_img[:, :-2, 1:-1]
        img_ym = pad_img[:, 1:-1, :-2]
        img_yp = pad_img[:, 1:-1, 2:]

        grad_x = img - img_xm
        grad_ym = img - img_ym
        grad_yp = img_yp - img
        grad_y = 0.5 * (grad_ym + grad_yp)
        return torch.stack([grad_x, grad_y], dim=-1)

    def compute_cos_map(self, img_grad: torch.Tensor):
        return img_grad[:, :, :, 0] / (torch.linalg.norm(img_grad, dim=-1) + 1e-5)

    def generate_noise_map(self, label_img: torch.Tensor):
        r0_map = torch.normal(
            torch.zeros(label_img.shape, device=self.device),
            torch.ones(label_img.shape, device=self.device),
        )
        r1_map = torch.normal(
            torch.zeros(label_img.shape, device=self.device),
            torch.ones(label_img.shape, device=self.device),
        )
        n_map = r0_map * self.n_s0 + self.n_mu0
        n_map_zero = torch.logical_not(r1_map <= self.n_mu1)
        n_map[n_map_zero] = 0

        beta_map = self.beta * torch.ones(label_img.shape, device=self.device)
        beta_l_map = torch.cumsum(beta_map, dim=1) * self.e
        TGC_map = torch.exp(beta_l_map * self.n_f)
        return n_map * TGC_map * self.n_I

    # ------------------------------------------------------------------
    # Correct convex geometry warp (sector) with dtype-safe grid_sample
    def apply_convex_geometry(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (N, H, W) - rectangular Cartesian slice (H=depth, W=lateral)
        returns: (N, H, W) - convex-probe fan shape.

        r0 is the normalised probe radius (probe face = arc at distance r0 from the
        virtual apex above the image).  r0=0 → sharp point (phased-array).
        r0=2 → probe face is ~67% of bottom width (typical convex-probe look).
        Set us_cfg["probe_radius_norm"] to override the default.
        """
        assert img.ndim == 3, f"expected (N,H,W), got {tuple(img.shape)}"
        N, H, W = img.shape
        device = img.device

        probe_cfg    = self.us_cfg.get("probe", {})
        fan_angle    = float(probe_cfg.get("fan_angle", self.fan_angle_deg))
        # r0: normalised radius — top_width / bottom_width = r0 / (r0 + 1)
        # Default 2.0 → top ≈ 67 % of bottom, matches a typical curvilinear probe image.
        r0           = float(self.us_cfg.get("probe_radius_norm",
                             probe_cfg.get("radius_norm", 2.0)))

        half_fan_rad = fan_angle * 0.5 * (torch.pi / 180.0)
        sin_half     = float(torch.sin(torch.tensor(half_fan_rad, device=device)))

        # Lateral extent at the bottom arc: (r0 + 1) * sin(half_fan)
        x_max = (r0 + 1.0) * sin_half

        # Cartesian output canvas: x ∈ [-x_max, +x_max], z ∈ [0, 1]
        x_coords = torch.linspace(-x_max, x_max, W, device=device)
        z_coords = torch.linspace(0.0, 1.0,       H, device=device)
        Z, X = torch.meshgrid(z_coords, x_coords, indexing="ij")  # (H, W)

        # Distance from virtual apex (located r0 above the probe face, i.e. z = -r0)
        R_apex = torch.sqrt(X**2 + (Z + r0)**2)
        Theta  = torch.atan2(X,    Z + r0)

        # Fan mask: probe-face arc (r0) → max-depth arc (r0+1), within ±half_fan_rad
        in_fan = (R_apex >= r0) & (R_apex <= r0 + 1.0) & (Theta.abs() <= half_fan_rad)

        # Sample from flat input at the same Cartesian (x, z) position
        x_norm = (X / (x_max + 1e-6)).clamp(-1.0, 1.0)   # lateral → [-1, 1]
        z_norm = (2.0 * Z - 1.0).clamp(-1.0, 1.0)         # depth   → [-1, 1]

        grid = torch.stack([x_norm, z_norm], dim=-1).unsqueeze(0).expand(N, -1, -1, -1)

        in_dtype = img.dtype
        out = F.grid_sample(
            img.float().unsqueeze(1),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(1)  # (N, H, W)

        out = out * in_fan.float().unsqueeze(0)

        if in_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            out = out.round().to(in_dtype)

        return out

    # ------------------------------------------------------------------
    # US simulation
    # ------------------------------------------------------------------
    def simulate_US_image(self, label_img: torch.Tensor, if_noise=True):
        """
        label_img: (N,H,W) integer labels
        returns: (N,H,W) float US intensities (sector-warped)
        """
        self.I0_map = torch.ones(label_img.shape, device=self.device) * self.I0

        params_map = self.assign_params_map(label_img)
        alpha_map = params_map[:, :, :, 0]
        z_map = params_map[:, :, :, 1]
        T_params_map = params_map[:, :, :, 2:5]

        atten_map = self.compute_attenuation_map(alpha_map)
        edge_map = self.compute_edge_map(label_img)

        edge_grad = self.compute_image_gradient(edge_map)
        cos_map = self.compute_cos_map(edge_grad)

        pad_z_map = F.pad(z_map, (1, 1, 1, 1), mode="reflect")
        z2_map = z_map
        z1_map = pad_z_map[:, :-2, 1:-1]

        E_map = self.I0_map * atten_map
        E_map = E_map * edge_map
        E_map = E_map * (z1_map - z2_map) ** 2 / (z1_map + z2_map + 1e-5) ** 2
        E_map *= cos_map
        E_map = E_map[:, None, :, :]
        E_map = F.conv2d(E_map, self.PSF_E, stride=1, padding="same")[:, 0, :, :]

        # Random speckle
        T0_map = torch.normal(torch.zeros(label_img.shape, device=self.device),
                              torch.ones(label_img.shape, device=self.device))
        T1_map = torch.normal(torch.zeros(label_img.shape, device=self.device),
                              torch.ones(label_img.shape, device=self.device))

        S_map = T0_map * T_params_map[:, :, :, 2] + T_params_map[:, :, :, 0]
        S_map_zero = torch.logical_not(T1_map <= T_params_map[:, :, :, 1])
        S_map[S_map_zero] = 0

        # Large scale speckle (unchanged logic)
        if self.if_large_scale_speckle:
            Vl_map = torch.normal(
                torch.zeros((label_img.shape[0], self.l_size, self.l_size), device=self.device),
                torch.ones((label_img.shape[0], self.l_size, self.l_size), device=self.device),
            )
            Al_map = params_map[:, :, :, 5]
            fl_map = params_map[:, :, :, 6].clamp_min(1e-6)

            inds_n, inds_h, inds_w = torch.meshgrid(
                torch.arange(0, label_img.shape[0], device=self.device),
                torch.arange(0, label_img.shape[1], device=self.device),
                torch.arange(0, label_img.shape[2], device=self.device),
                indexing="ij",
            )
            inds = torch.stack([inds_n, inds_h, inds_w], dim=-1)  # (n,H,W,3)
            inds_lower = (inds / fl_map[:, :, :, None]).long()
            inds_lower[..., 1] = inds_lower[..., 1].clamp(0, self.l_size - 1)
            inds_lower[..., 2] = inds_lower[..., 2].clamp(0, self.l_size - 1)

            S_map = S_map * (1 + Al_map * Vl_map[
                inds_lower[:, :, :, 0],
                inds_lower[:, :, :, 1],
                inds_lower[:, :, :, 2],
            ])

        S_map = S_map[:, None, :, :]
        B_map = self.I0_map * atten_map * F.conv2d(S_map, self.PSF_B, padding="same")[:, 0, :, :]

        US = self.E_S_ratio * E_map + B_map

        if if_noise:
            US = US + self.generate_noise_map(label_img=label_img)

        # Sector warp for *intensity* output
        US = self.apply_convex_geometry(US)
        return US

    def simulate_US_image_given_rand_map(
        self,
        label_img: torch.Tensor,
        T0_img: torch.Tensor,
        T1_img: torch.Tensor,
        Vl_img: torch.Tensor,
        if_noise=True,
        if_ct=False,
        ct_img=None,
    ):
        """
        label_img: (N,H,W) integer labels
        returns: (N,H,W) float US intensities (NOT sector-warped here by default)
        """
        self.I0_map = torch.ones(label_img.shape, device=label_img.device) * self.I0

        params_map = self.assign_params_map(label_img)
        alpha_map = params_map[:, :, :, 0]

        if if_ct:
            z_map = ct_img * 1e2
        else:
            z_map = params_map[:, :, :, 1]

        T_params_map = params_map[:, :, :, 2:5]

        atten_map = self.compute_attenuation_map(alpha_map - self.beta)
        atten_map_E = self.compute_attenuation_map(alpha_map - self.beta_edge)

        edge_map = self.compute_ct_edge_map(ct_img) if if_ct else self.compute_edge_map(label_img)

        edge_grad = self.compute_image_gradient(edge_map)
        cos_map = self.compute_cos_map(edge_grad)

        pad_z_map = F.pad(z_map, (1, 1, 1, 1), mode="reflect")
        z2_map = z_map
        z1_map = pad_z_map[:, :-2, 1:-1]

        E_map = self.I0_map * atten_map_E
        E_map = E_map * (z1_map - z2_map) ** 2 / (z1_map + z2_map + 1e-5) ** 2
        if not if_ct:
            E_map = E_map * edge_map
            E_map *= cos_map
        E_map = E_map[:, None, :, :]
        E_map = E_map.clamp_max(0.1)
        E_map = F.conv2d(E_map, self.PSF_E, stride=1, padding="same")[:, 0, :, :]

        T0_map = T0_img
        T1_map = T1_img
        S_map = T0_map * T_params_map[:, :, :, 2] + T_params_map[:, :, :, 0]
        S_map_zero = torch.logical_not(T1_map <= T_params_map[:, :, :, 1])
        S_map[S_map_zero] = 0

        if self.if_large_scale_speckle:
            Vl_map = Vl_img
            Al_map = params_map[:, :, :, 5]
            fl_map = params_map[:, :, :, 6].clamp_min(1e-6)

            inds_n, inds_h, inds_w = torch.meshgrid(
                torch.arange(0, label_img.shape[0], device=self.device),
                torch.arange(0, label_img.shape[1], device=self.device),
                torch.arange(0, label_img.shape[2], device=self.device),
                indexing="ij",
            )
            inds = torch.stack([inds_n, inds_h, inds_w], dim=-1)
            inds_lower = (inds / fl_map[:, :, :, None]).long()
            S_map = S_map * (1 + Al_map * Vl_map[
                inds_lower[:, :, :, 0],
                inds_lower[:, :, :, 1].clamp(0, Vl_map.shape[1] - 1),
                inds_lower[:, :, :, 2].clamp(0, Vl_map.shape[2] - 1),
            ])

        S_map = S_map[:, None, :, :]
        B_map = self.I0_map * atten_map * F.conv2d(S_map, self.PSF_B, padding="same")[:, 0, :, :]

        US = self.E_S_ratio * E_map + B_map

        if if_noise:
            US = US + self.generate_noise_map(label_img=label_img)

        # Apply convex (fan) geometry warp
        US = self.apply_convex_geometry(US)

        return US

