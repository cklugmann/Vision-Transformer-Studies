import torch
import torch.nn as nn

from ViT.utils import make_patches
from ViT.transformer import MultiHeadAttention


class PositionWiseFFN(nn.Module):
    def __init__(self, num_features: int, num_layers: int = 1):
        super().__init__()
        layers = list()
        for layer in range(num_layers):
            layers.append(
                nn.Linear(in_features=num_features, out_features=num_features)
            )
            layers.append(nn.ReLU())
        self.ffn = nn.Sequential(*layers)

    def __call__(self, x, *args, **kwargs):
        """
        :param x: A torch tensor of shape (B, N, D).
        :return: A torch tensor of the same shape, i.e. (B, N, D).
        """
        return self.ffn(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 1):
        super().__init__()
        self.mha = MultiHeadAttention(
            input_dim=hidden_dim, output_dim=hidden_dim, num_heads=num_heads
        )
        self.norm1 = torch.nn.LayerNorm(normalized_shape=hidden_dim)
        self.norm2 = torch.nn.LayerNorm(normalized_shape=hidden_dim)
        self.ffn = PositionWiseFFN(num_features=self.mha.output_dim)

    def get_output_dim(self):
        return self.mha.output_dim

    def __call__(self, x, *args, **kwargs):
        """
        :param x: A torch tensor of shape (B, N, D)
            where B is the batch size, N the sequence length and D the number of features.
        """
        z = self.norm1(x + self.mha(x))
        z = self.norm2(z + self.ffn(z))
        return z


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_encoder_layers: int,
    ):
        super().__init__()
        blocks = list()
        for _ in range(num_encoder_layers):
            blocks.append(TransformerBlock(hidden_dim=hidden_dim, num_heads=num_heads))
        self.block_transform = nn.Sequential(*blocks)

    def __call__(self, x, *args, **kwargs):
        z = self.block_transform(x)
        return z


class VisionTransformer(nn.Module):
    def __init__(
        self,
        im_size,
        patch_size: int,
        embedding_dim: int,
        output_dim: int,
        num_heads: int = 1,
        num_encoder_layers: int = 1,
    ):
        super().__init__()

        # Assume RGB images of size im_size x im_size

        self.im_size = im_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        num_patches = self.get_num_patches()
        num_features = 3 * patch_size * patch_size

        # Embedding weights
        self.embedding_weights = nn.Parameter(torch.rand(num_features, embedding_dim))

        # Learnable class token
        self.class_token = nn.Parameter(torch.rand(1, self.embedding_dim))

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(
            torch.rand(num_patches + 1, embedding_dim)
        )

        self.transformer_encoder = TransformerEncoder(
            hidden_dim=self.embedding_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
        )

        self.fc = nn.Linear(in_features=embedding_dim, out_features=output_dim)

    def get_num_patches(self):
        patches_per_dim = self.im_size // self.patch_size
        return patches_per_dim * patches_per_dim

    def __call__(self, x, *args, **kwargs):
        """
        :param x: A torch tensor of RGB images (B, 3, K, K) (assume square images)
        """
        patches = make_patches(x, self.patch_size)

        # Flatten patches
        flattened_patches = torch.flatten(patches, start_dim=-3)

        # Make flat list of flattened patches
        flattened_patches = torch.flatten(flattened_patches, start_dim=1, end_dim=2)

        patch_embeddings = torch.einsum(
            "ji, ...j -> ...i", self.embedding_weights, flattened_patches
        )

        # Concat with class token embedding
        batch_size, sequence_length = patch_embeddings.shape[:2]
        class_token = torch.unsqueeze(self.class_token, dim=0).repeat(batch_size, 1, 1)
        patch_embeddings = torch.cat([class_token, patch_embeddings], dim=1)

        # Add positional encoding
        patch_embeddings = patch_embeddings + torch.unsqueeze(
            self.positional_encoding, dim=0
        )

        z = self.transformer_encoder(patch_embeddings)

        # Take first item in output sequence as the head for classification
        z_0 = z[..., 0, :]

        logits = self.fc(z_0)

        return logits
