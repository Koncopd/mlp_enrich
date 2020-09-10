import torch
import torch.nn as nn

class MaskedLinear(nn.Linear):
    def __init__(self, mask, bias=True):
        # mask should have the same dimensions as the transposed linear weight
        # n_input x n_output_nodes
        in_features = mask.shape[0]
        out_features = mask.shape[1]

        super().__init__(in_features, out_features, bias)

        self.mask = mask.t()

    def forward(self, input):
        return nn.functional.linear(input, self.weight*self.mask, self.bias)
