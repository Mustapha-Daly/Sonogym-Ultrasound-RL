import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from monai.networks.nets.unet import UNet

from spinal_surgery import PROJECT_DIR


class USSimulatorNetwork:
    """
    Network-based US simulator.

    Key fixes vs. your broken version:
    - No nested function definitions inside __init__.
    - No references to `env` or `us_img` inside __init__.
    - Robust device handling (cpu/cuda/cuda:0).
    - Stores multiple models; selects model periodically (model_change_interval).
    - Histogram-matching state initialized safely.
    """

    def __init__(self, us_model_cfg: dict, device: str | torch.device):
        # -------------------------
        # Device normalization
        # -------------------------
        if isinstance(device, torch.device):
            self.device = device
        else:
            # accept "cpu", "cuda", "cuda:0", etc.
            if str(device).startswith("cuda") and torch.cuda.is_available():
                self.device = torch.device(str(device))
            else:
                self.device = torch.device("cpu")

        # -------------------------
        # Save cfg + basic params
        # -------------------------
        self.cfg = us_model_cfg
        self.CT_cfg = us_model_cfg["CT"]
        self.label_res = us_model_cfg["label_res"]
        self.image_size = self.CT_cfg["size"]

        # intervals
        self.k = int(us_model_cfg.get("reset_hist_interval", 1))
        self.model_k = int(us_model_cfg.get("model_change_interval", 1))
        self.step = 0

        # will be set later
        self.test_min = None
        self.test_max = None
        self.test_cdf = None

        # -------------------------
        # Load one or more models
        # -------------------------
        model_paths = us_model_cfg["model_path"]
        if isinstance(model_paths, str):
            model_paths = [model_paths]

        self.model_list: list[torch.nn.Module] = []

        # map_location for torch.load should be CPU unless you explicitly want GPU weights
        # (but since we move model to self.device, CPU map_location is safe and common).
        map_location = torch.device("cpu")

        for rel_path in model_paths:
            abs_path = os.path.join(PROJECT_DIR, rel_path)

            model = UNet(
                spatial_dims=us_model_cfg["model"]["spatial_dims"],
                in_channels=us_model_cfg["model"]["in_channels"],
                out_channels=us_model_cfg["model"]["out_channels"],
                channels=us_model_cfg["model"]["channels"],
                strides=us_model_cfg["model"]["strides"],
                num_res_units=us_model_cfg["model"]["num_res_units"],
                dropout=us_model_cfg["model"]["dropout"],
                act=("leakyrelu", {"negative_slope": 0.2}),
            )

            state_dict = torch.load(abs_path, map_location=map_location)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            self.model_list.append(model)

        if len(self.model_list) == 0:
            raise RuntimeError("No models loaded. Check us_model_cfg['model_path'].")

        # active model
        self.model = self.model_list[0]

        # -------------------------
        # Build histogram from training samples
        # -------------------------
        self.num_bins = int(self.cfg["num_bins"])
        self.construct_train_data_histogram()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def simulate_US_image(self, ct_img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Simulate US image from CT image.
        Input:
          ct_img_tensor: (B, 1, H, W)   (B can be num_envs or num_envs*something)
        Output:
          us_img_tensor: (B, 1, H, W)   resized back to original H,W
        """
        if not torch.is_tensor(ct_img_tensor):
            raise TypeError("ct_img_tensor must be a torch.Tensor")

        # Ensure correct device
        ct_img_tensor = ct_img_tensor.to(self.device)

        # 1) intensity clamp + normalize to [0,1]
        ct = torch.clamp(ct_img_tensor, self.CT_cfg["range"][0], self.CT_cfg["range"][1])
        ct = (ct - self.CT_cfg["range"][0]) / (self.CT_cfg["range"][1] - self.CT_cfg["range"][0] + 1e-8)

        # 2) resize to configured network input size
        ct = F.interpolate(
            ct,
            size=(self.image_size[0], self.image_size[1]),
            mode="nearest-exact",
        )

        # 3) downsample + upsample (your original augmentation)
        ct = F.interpolate(
            ct,
            size=(self.image_size[0] // 4, self.image_size[1] // 4),
            mode="nearest-exact",
        )
        ct = F.interpolate(
            ct,
            size=(self.image_size[0], self.image_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        # 4) histogram match
        ct = self.test_match_train(ct)

        # 5) run network
        us = self.model(ct)

        # 6) resize back to original size
        us_img_tensor = F.interpolate(
            us,
            size=(ct_img_tensor.shape[-2], ct_img_tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        return us_img_tensor.detach()

    # Optional alias (if other code expects .simulate(...))
    @torch.no_grad()
    def simulate(self, ct_img_tensor: torch.Tensor) -> torch.Tensor:
        return self.simulate_US_image(ct_img_tensor)

    # -------------------------------------------------------------------------
    # Histogram matching helpers
    # -------------------------------------------------------------------------
    def read_img_folder(self, folder_path: str) -> None:
        """Read all images in a folder and store them as a tensor on self.device."""
        transform = transforms.ToTensor()
        image_tensors = []

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Train data sample folder not found: {folder_path}")

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpeg", ".jpg")):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert("L")
                img_tensor = transform(img)  # (1,H,W) in [0,1]
                image_tensors.append(img_tensor)

        if len(image_tensors) == 0:
            raise RuntimeError(f"No PNG/JPEG images found in: {folder_path}")

        self.train_samples = torch.stack(image_tensors, dim=0).to(self.device)  # (N,1,H,W)

    def construct_train_data_histogram(self) -> None:
        """Compute source histogram and CDF from training sample images."""
        source_path = os.path.join(PROJECT_DIR, self.cfg["train_data_sample_path"])
        self.read_img_folder(source_path)

        source_flat = self.train_samples.flatten()
        self.src_min, self.src_max = source_flat.min(), source_flat.max()
        source_norm = (source_flat - self.src_min) / (self.src_max - self.src_min + 1e-8)

        # bin edges/centers
        self.bin_edges = torch.linspace(0.0, 1.0, steps=self.num_bins + 1, device=self.device)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0

        # histogram + CDF
        self.source_hist = torch.histc(source_norm, bins=self.num_bins, min=0.0, max=1.0)
        self.source_cdf = torch.cumsum(self.source_hist, dim=0)
        self.source_cdf = self.source_cdf / (self.source_cdf[-1] + 1e-8)

    def test_match_train(self, test: torch.Tensor) -> torch.Tensor:
        """
        Histogram matching (vectorized).
        Input `test`: (B,1,H,W) float tensor on self.device.
        Output: same shape, histogram-matched into train-sample intensity distribution.
        """
        # switch model periodically
        if self.model_k > 0 and (self.step % self.model_k == 0) and (len(self.model_list) > 1):
            m = np.random.randint(0, len(self.model_list))
            self.model = self.model_list[m]

        # reset test range periodically
        if self.k > 0 and (self.step % self.k == 0):
            self.test_min = test.min()
            self.test_max = test.max()

        # safety in case k was 0 or not set
        if self.test_min is None or self.test_max is None:
            self.test_min = test.min()
            self.test_max = test.max()

        test_norm = (test - self.test_min) / (self.test_max - self.test_min + 1e-8)
        test_flat = test_norm.flatten()

        # update test CDF periodically
        if self.k > 0 and (self.step % self.k == 0) or (self.test_cdf is None):
            test_hist = torch.histc(test_flat, bins=self.num_bins, min=0.0, max=1.0)
            self.test_cdf = torch.cumsum(test_hist, dim=0)
            self.test_cdf = self.test_cdf / (self.test_cdf[-1] + 1e-8)

        # bucketize test values into bins
        test_bins = torch.bucketize(test_flat, self.bin_edges[:-1], right=False)
        test_bins = torch.clamp(test_bins, 1, self.num_bins) - 1  # [0, num_bins-1]

        # map test CDF values to source CDF
        mapping_indices = torch.searchsorted(self.source_cdf, self.test_cdf)
        mapping_indices = torch.clamp(mapping_indices, 0, self.num_bins - 1)

        lut = self.bin_centers[mapping_indices]
        matched_norm = lut[test_bins]

        # denormalize back into training sample intensity range
        matched = matched_norm * (self.src_max - self.src_min) + self.src_min

        self.step += 1
        return matched.reshape_as(test)

