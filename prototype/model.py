import torch.nn as nn


class MLP(nn.Module):
    """Vanilla MLP: 784 → 400 → 400 → 10, ReLU activations, no batchnorm, no dropout."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 10),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
