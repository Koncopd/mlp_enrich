import torch
import torch.nn as nn

class MaskedLinear(nn.Linear):
    def __init__(self, mask, bias=True):
        # mask should have the same dimensions as the transposed linear weight
        # n_input x n_output_nodes
        in_features = mask.shape[0]
        out_features = mask.shape[1]

        super().__init__(in_features, out_features, bias)

        self.register_buffer('mask', mask.t())

    def forward(self, input):
        return nn.functional.linear(input, self.weight*self.mask, self.bias)


std_layer = lambda mask: nn.Sequential(MaskedLinear(mask),
                                       nn.BatchNorm1d(mask.shape[1]),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))


class EnrichClassifier(nn.Module):
    def __init__(self, pathways_mask, n_labels, divide_nodes=(5, 2, 2), min_nodes=(4, 3, 2)):
        super().__init__()

        layer_1_sizes = pathways_mask.sum(0) // divide_nodes[0]
        layer_1_sizes[layer_1_sizes < min_nodes[0]] = min_nodes[0]
        layer_1_sizes = layer_1_sizes.type(torch.LongTensor)

        layer_1_mask = pathways_mask.repeat_interleave(layer_1_sizes, dim=1)

        layer_2_sizes = layer_1_sizes // divide_nodes[1]
        layer_2_sizes[layer_2_sizes < min_nodes[1]] = min_nodes[1]

        layer_2_mask = torch.block_diag(*(torch.ones((size, 1)) for size in layer_1_sizes))
        layer_2_mask = layer_2_mask.repeat_interleave(layer_2_sizes, dim=1)

        layer_3_sizes = layer_2_sizes // divide_nodes[2]
        layer_3_sizes[layer_3_sizes < min_nodes[2]] = min_nodes[2]

        layer_3_mask = torch.block_diag(*(torch.ones((size, 1)) for size in layer_2_sizes))
        layer_3_mask = layer_3_mask.repeat_interleave(layer_3_sizes, dim=1)

        final_mask = torch.block_diag(*(torch.ones((size, 1)) for size in layer_3_sizes))

        self.score_pathways = nn.Sequential(std_layer(layer_1_mask),
                                            std_layer(layer_2_mask),
                                            std_layer(layer_3_mask),
                                            MaskedLinear(final_mask),
                                            nn.ReLU())

        self.classify = nn.Linear(final_mask.shape[1], n_labels)

        self.normalize = nn.Softmax(dim=1)

    def forward(self, x):
        scores = self.score_pathways(x)
        return self.classify(scores)

    def get_scores(self, x):
        return self.score_pathways(x).data

    def get_probs(self, x):
        return self.normalize(self.forward(x))

    def get_relevance(self):
        return self.classify.weight.data

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)
