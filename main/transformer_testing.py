import torch

from ViT.transformer.transformer import VisionTransformer


def main():
    im_size = 32
    patch_size = 4
    embedding_dim = 16
    output_dim = 10

    x = torch.rand(4, 3, im_size, im_size)

    vit = VisionTransformer(
        im_size, patch_size, embedding_dim, output_dim,
        num_heads=8, num_encoder_layers=16
    )
    vit.eval()

    with torch.no_grad():
        y = vit(x)

    print(y)
    print(sum(p.numel() for p in vit.parameters()))


if __name__ == "__main__":
    main()
