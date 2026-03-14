from torch import nn


class LinearClassificationNet(nn.Module):
    """
    A fully-connected single-layer linear NN for classification.
    """

    def __init__(self, input_size=11, output_size=2):
        super().__init__()
        self.layer1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.layer1(x.float())
