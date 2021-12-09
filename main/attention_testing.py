import torch

from ViT.transformer import MultiHeadAttention


def main():
    sequence_length = 16
    input_dim = 64

    mha = MultiHeadAttention(input_dim=input_dim, num_heads=8)
    mha.eval()

    x = torch.rand(4, sequence_length, input_dim)

    with torch.no_grad():
        scores = mha(x)

    print(scores.shape)
    print(sum(p.numel() for p in mha.parameters()))


if __name__ == "__main__":
    main()
