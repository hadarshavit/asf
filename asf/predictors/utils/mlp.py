try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_mlp(
    input_size: int,
    output_size: int,
    hidden_sizes: list[int] = [128, 64],
    dropout: float = 0.0,
    output_activation: str = None,
    compile: bool = False,
):
    layers = [torch.nn.Linear(input_size, hidden_sizes[0]), torch.nn.ReLU()]

    for i in range(len(hidden_sizes) - 1):
        layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        layers.append(torch.nn.ReLU())

    layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))

    if output_activation is not None:
        layers.append(output_activation)

    model = torch.nn.Sequential(*layers)

    if compile:
        model = torch.compile(model)
    return model


class ExpActivation(torch.nn.Module):
    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, x):
        return torch.exp(x)
