from __future__ import annotations

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
):
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install it with: pip install torch"
        )
    layers = [torch.nn.Linear(input_size, hidden_sizes[0]), torch.nn.ReLU()]

    for i in range(len(hidden_sizes) - 1):
        layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        layers.append(torch.nn.ReLU())

    layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))

    model = torch.nn.Sequential(*layers)

    return model
