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
    activation_cls: type[torch.nn.Module] = torch.nn.ReLU,
    output_activation: torch.nn.Module = None,
    compile: bool = False,
    use_batchnorm: bool = True,
):
    layers = [
        torch.nn.Linear(input_size, hidden_sizes[0]),
        activation_cls(),
    ]
    if use_batchnorm:
        layers.append(torch.nn.BatchNorm1d(hidden_sizes[0]))

    for i in range(len(hidden_sizes) - 1):
        layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        layers.append(activation_cls())
        if use_batchnorm:
            layers.append(torch.nn.BatchNorm1d(hidden_sizes[i + 1]))

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
