import torch


def make_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    num_channels = x.shape[-3]
    patches = torch.squeeze(
        x.unfold(1, num_channels, num_channels)
            .unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size),
        dim=1,
    )
    return patches
