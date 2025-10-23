try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def wmse(input, target, weights):
    return torch.mean(
        weights * torch.nn.functional.mse_loss(input, target, reduction="none")
    )


def mse(y_pred, y_pred_s, y_pred_l, yc, ys, yl):
    return (
        torch.nn.functional.mse_loss(y_pred, yc)
        + torch.nn.functional.mse_loss(y_pred_s, ys)
        + torch.nn.functional.mse_loss(y_pred_l, yl)
    )


def bpr_loss(y_pred, y_pred_s, y_pred_l, yc, ys, yl):
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


def tml_loss(y_pred, y_pred_s, y_pred_l, yc, ys, yl, margin=1.0, p=2):
    return torch.nn.functional.triplet_margin_loss(
        y_pred, y_pred_s, y_pred_l, margin=margin, p=p
    )


@torch.jit.script
def lognorm_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    s = y_pred[:, 0]
    s = torch.reshape(s, [-1, 1])

    scale = y_pred[:, 1]
    scale = torch.reshape(scale, [-1, 1])
    log_scale = torch.log(scale)
    log_true = torch.log(y_true)

    # Compute logged lh (removed constants)
    help1 = log_true - log_scale
    help1 = 0.5 * torch.pow(help1 / s, 2)

    # add terms (not multiplying them)
    lh = -torch.log(s) - log_true - help1

    return -lh.mean()


@torch.jit.script
def invgauss_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    mu = y_pred[:, 0]
    mu = torch.reshape(mu, [-1, 1])

    scale = y_pred[:, 1]
    scale = torch.reshape(scale, [-1, 1])

    tmp_true = torch.zeros_like(y_true)
    y_true = tmp_true + y_true

    # Compute logged lh (removed constants)
    help1 = 0.5 * torch.log(scale)
    help2 = 3.0 / 2.0 * torch.log(y_true)

    tmp = y_true / scale

    help3 = torch.pow(tmp - mu, 2)
    lower = 2 * tmp * torch.pow(mu, 2)
    help3 = help3 / lower

    # add terms (not multiplying them)
    lh = help1 - help2 - help3

    return -lh.mean()


@torch.jit.script
def exp_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    scale = y_pred[:, 0]
    scale = torch.reshape(scale, [-1, 1])
    scale = 1 / scale

    log_scale = torch.log(scale)

    # Compute logged lh (removed constants)
    lh = log_scale - y_true * scale

    return -lh.mean()


# def weibull_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
#     shape = y_pred[:, 0]
#     shape = torch.reshape(shape, [-1, 1])

#     scale = y_pred[:, 1]
#     scale = torch.reshape(scale, [-1, 1])


#     return -lh
