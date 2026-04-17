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
    """
    Build an MLP matching the ZAP paper's batch_mlp architecture.
    
    Architecture: For each hidden layer:
        Linear -> Dropout -> ReLU
    Followed by final Linear output layer.
    
    Reference: https://arxiv.org/pdf/2206.08476
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install it with: pip install torch"
        )
    
    layers = []
    prev_size = input_size
    
    for hidden_size in hidden_sizes:
        layers.append(torch.nn.Linear(prev_size, hidden_size))
        layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.ReLU())
        prev_size = hidden_size
    
    layers.append(torch.nn.Linear(prev_size, output_size))

    model = torch.nn.Sequential(*layers)

    return model

