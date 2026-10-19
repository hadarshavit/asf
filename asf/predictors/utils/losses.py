from typing import Any

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def wmse(input: Any, target: Any, weights: Any) -> Any:
    return torch.mean(
        weights * torch.nn.functional.mse_loss(input, target, reduction="none")
    )


def mse(y_pred: Any, y_pred_s: Any, y_pred_l: Any, yc: Any, ys: Any, yl: Any) -> Any:
    return (
        torch.nn.functional.mse_loss(y_pred, yc)
        + torch.nn.functional.mse_loss(y_pred_s, ys)
        + torch.nn.functional.mse_loss(y_pred_l, yl)
    )


def bpr_loss(
    y_pred: Any, y_pred_s: Any, y_pred_l: Any, yc: Any, ys: Any, yl: Any
) -> Any:
    return torch.nn.functional.binary_cross_entropy(
        torch.cat(
            [
                torch.sigmoid(y_pred - y_pred_s),
                torch.sigmoid(y_pred_l - y_pred),
                torch.sigmoid(y_pred_l - y_pred_s),
            ],
            0,
        ),
        torch.ones(3 * yc.shape[0], 1).to(yc.device),
    )


def tml_loss(
    y_pred: Any,
    y_pred_s: Any,
    y_pred_l: Any,
    yc: Any,
    ys: Any,
    yl: Any,
    margin: Any = 1.0,
    p: Any = 2,
) -> Any:
    return torch.nn.functional.triplet_margin_loss(
        y_pred, y_pred_s, y_pred_l, margin=margin, p=p
    )
