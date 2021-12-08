from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, output_dim: Optional[int] = None, num_heads: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim // num_heads
        self.num_heads = num_heads

        self.weight_query = nn.Parameter(torch.rand(self.num_heads, self.input_dim, self.output_dim))
        self.weight_key = nn.Parameter(torch.rand(self.num_heads, self.input_dim, self.output_dim))
        self.weight_value = nn.Parameter(torch.rand(self.num_heads, self.input_dim, self.output_dim))

        self.weight_concat = nn.Parameter(torch.rand(self.output_dim, self.num_heads * self.output_dim))

    @staticmethod
    def multi_head_dot(weights, inputs):
        return torch.einsum("...ikl, ...jk -> ...ijl", weights, inputs)

    def __call__(self, x, *args, **kwargs):
        q = MultiHeadAttention.multi_head_dot(self.weight_query, x)
        k = MultiHeadAttention.multi_head_dot(self.weight_key, x)
        v = MultiHeadAttention.multi_head_dot(self.weight_value, x)

        # How well do the queries match the keys?
        scores = torch.div(
            torch.einsum("...hlm, ...hnm -> ...hln", q, k),
            torch.sqrt(torch.tensor(self.output_dim)),
        ).softmax(dim=-1)

        # Weighting the values with scores
        z = torch.einsum(
            "...ik, ...kj -> ...ij", scores, v
        )

        # Swap number of heads and sequence length axes
        z = z.transpose(1, 2)

        # Flatten the multiple outputs
        z = torch.flatten(z, start_dim=2)

        # Final projection
        z = torch.einsum("ij, ...j -> ...i", self.weight_concat, z)

        return z
