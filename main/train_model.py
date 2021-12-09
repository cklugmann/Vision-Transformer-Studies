import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim

import torch.nn as nn

from ViT.transformer.transformer import VisionTransformer


def running_mean(current_value, new_value, step):
    return 1 / (step + 1) * (step * current_value + new_value)


def compute_accuracy(logits, labels):
    pred_labels = torch.argmax(logits, dim=1)
    return torch.sum(pred_labels == labels) / len(labels)


def eval_model(model, data_loader, criterion, device=torch.device("cpu")):
    model.eval()
    mean_loss, mean_accuracy = 0.0, 0.0
    for step, batch in enumerate(data_loader):
        images, labels = [x.to(device) for x in batch]

        with torch.no_grad():
            logits = model(images)
            loss = criterion(logits, labels)

        accuracy = compute_accuracy(logits, labels)

        mean_loss = running_mean(mean_loss, loss.item(), step)
        mean_accuracy = running_mean(mean_accuracy, accuracy.item(), step)

        print(
            "\r\tEval - step {} - loss {:.3f} - accuracy {:.3f}".format(
                step + 1, mean_loss, mean_accuracy
            ),
            end="",
        )
    print("")
    model.train()

    return mean_loss, mean_accuracy


def main():

    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_data = datasets.CIFAR10(
        root="../datasets/cifar10", train=True, download=True, transform=transforms
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False)

    test_data = datasets.CIFAR10(
        root="../datasets/cifar10",
        train=False,
        download=True,
        transform=transforms,
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    image, label = next(iter(train_loader))

    im_size = image.shape[-1]
    patch_size = 4
    embedding_dim = 32
    # Number of classes
    output_dim = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit = VisionTransformer(
        im_size,
        patch_size,
        embedding_dim,
        output_dim,
        num_heads=2,
        num_encoder_layers=1,
    )
    vit.to(device)

    print("Number of parameters", sum(p.numel() for p in vit.parameters()))

    optimizer = optim.Adam(vit.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir="../logs")

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

        writer.add_scalar("Loss/train", mean_loss, epoch)
        writer.add_scalar("Accuracy/train", mean_accuracy, epoch)

        mean_loss_test, mean_accuracy_test = eval_model(vit, test_loader, criterion, device=device)
        writer.add_scalar("Loss/test", mean_loss_test, epoch)
        writer.add_scalar("Accuracy/test", mean_accuracy_test, epoch)


if __name__ == "__main__":
    main()
