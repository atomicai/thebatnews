from typing import List

import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    A feed forward neural network of variable depth and width.
    """

    def __init__(self, layer_dims: List[int], **kwargs):
        # Todo: Consider having just one input argument
        super(FeedForwardBlock, self).__init__()
        self.layer_dims = layer_dims
        # If read from config the input will be string
        n_layers = len(layer_dims) - 1
        layers_all = []
        # TODO: IS this needed?
        self.output_size = layer_dims[-1]

        for i in range(n_layers):
            size_in = layer_dims[i]
            size_out = layer_dims[i + 1]
            layer = nn.Linear(size_in, size_out)
            layers_all.append(layer)
        self.feed_forward = nn.Sequential(*layers_all)

    def forward(self, X: torch.Tensor):
        logits = self.feed_forward(X)
        return logits


__all__ = ["FeedForwardBlock"]
