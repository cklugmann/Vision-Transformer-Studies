import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader

import torch.optim as optim

import torch.nn as nn

from ViT.transformer.transformer import VisionTransformer


def compute_accuracy(logits, labels):
    pred_labels = torch.argmax(logits, dim=1)
    return torch.sum(pred_labels == labels) / len(labels)


def main():
    train_data = datasets.CIFAR10(
        root="datasets/cifar10",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    train_loader = DataLoader(train_data, batch_size=16, shuffle=False)
    image, label = next(iter(train_loader))

    im_size = image.shape[-1]
    patch_size = 4
    embedding_dim = 64
    # Number of classes
    output_dim = 10

    vit = VisionTransformer(
        im_size,
        patch_size,
        embedding_dim,
        output_dim,
        num_heads=4,
        num_encoder_layers=8,
    )

    print("Number of parameters", sum(p.numel() for p in vit.parameters()))

    optimizer = optim.SGD(vit.parameters(), lr=1e-2, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(16):
        running_loss = 0.0
        running_accuracy = 0.0
        for step, batch in enumerate(train_loader):
            images, labels = batch
            logits = vit(images)

            optimizer.zero_grad()
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            accuracy = compute_accuracy(logits, labels)

            running_loss = 1 / (step + 1) * (step * running_loss + loss.item())
            running_accuracy = (
                1 / (step + 1) * (step * running_accuracy + accuracy.item())
            )

            print(
                "\rBatch {} - step {} - loss {:.3f} - accuracy {:.3f}".format(
                    epoch + 1, step + 1, running_loss, running_accuracy
                ),
                end="",
            )

        print("")


if __name__ == "__main__":
    main()
