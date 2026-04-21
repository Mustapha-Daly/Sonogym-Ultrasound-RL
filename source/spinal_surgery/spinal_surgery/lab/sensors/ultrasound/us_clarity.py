import torch

def us_clarity_score(us_img: torch.Tensor) -> torch.Tensor:
    """
    us_img: (N, C, H, W) or (N, 1, H, W)
    returns: (N,) clarity score
    """

    # Gradient magnitude → structure visibility
    dx = torch.abs(us_img[..., :, 1:] - us_img[..., :, :-1])
    dy = torch.abs(us_img[..., 1:, :] - us_img[..., :-1, :])

    grad_energy = dx.mean(dim=(-2, -1)) + dy.mean(dim=(-2, -1))

    return grad_energy.squeeze()
