import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader

from ViT.utils import make_patches


def main():

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    train_data = datasets.CIFAR10(
        root="../datasets/cifar10",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    image, _ = next(iter(train_loader))

    plt.imshow(np.transpose(image[0], (1, 2, 0)))
    plt.show()

    patches = make_patches(image, patch_size=4)
    patches = patches[0]

    images = torch.flatten(patches, start_dim=0, end_dim=1).numpy()

    M, N = patches.shape[:2]

    fig, axs = plt.subplots(M, N)

    for ax, im in zip(axs.reshape(-1,), images):
        ax.imshow(np.transpose(im, (1, 2, 0)))
        ax.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
