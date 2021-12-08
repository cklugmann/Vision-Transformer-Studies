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

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    train_data = datasets.CIFAR10(
        root="../datasets/cifar10",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False)
    image, label = next(iter(train_loader))

    im_size = image.shape[-1]
    patch_size = 4
    embedding_dim = 64
    # Number of classes
    output_dim = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit = VisionTransformer(
        im_size,
        patch_size,
        embedding_dim,
        output_dim,
        num_heads=4,
        num_encoder_layers=8,
    )
    vit.to(device)

    print("Number of parameters", sum(p.numel() for p in vit.parameters()))

    optimizer = optim.Adam(vit.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    def running_mean(current_value, new_value, step):
        return 1 / (step + 1) * (step * current_value + new_value)

    for epoch in range(1000):
        mean_loss, mean_accuracy = 0.0, 0.0
        for step, batch in enumerate(train_loader):
            images, labels = [x.to(device) for x in batch]
            logits = vit(images)

            optimizer.zero_grad()

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            accuracy = compute_accuracy(logits, labels)

            mean_loss = running_mean(mean_loss, loss.item(), step)
            mean_accuracy = running_mean(mean_accuracy, accuracy.item(), step)

            print(
                "\rBatch {} - step {} - loss {:.3f} - accuracy {:.3f}".format(
                    epoch + 1, step + 1, mean_loss, mean_accuracy
                ),
                end="",
            )

        print("")


if __name__ == "__main__":
    main()
