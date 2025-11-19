try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

LOGNORM_N_PARAMS = 2  # mu, sigma
EXP_N_PARAMS = 1  # scale
WEIBULL_N_PARAMS = 2  # c, scale
INVGAUSS_N_PARAMS = 2  # mu, lambda
GAMMA_N_PARAMS = 2  # k, theta
CAUCHY_N_PARAMS = 1  # loc, scale
LEVY_N_PARAMS = 1  # scale
BETA_N_PARAMS = 3  # alpha, beta
BETAPRIME_N_PARAMS = 3  # alpha, beta, scale
LOMAX_N_PARAMS = 2  # alpha, scale
NORM_N_PARAMS = 2  # mu, sigma


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
    y_true = torch.reshape(y_true, [-1, 1])
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
    lh = -torch.log(s) - log_true - help1 - 0.5 * torch.log(2.0 * torch.pi)

    return -lh.mean()


@torch.jit.script
def invgauss_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    mu = y_pred[:, 0]
    mu = torch.reshape(mu, [-1, 1])

    scale = y_pred[:, 1]
    scale = torch.reshape(scale, [-1, 1])

    help1 = torch.log(scale)  # corresponds to sqrt(lambda) term
    help2 = 1.5 * torch.log(y_true)  # corresponds to x
    help3 = scale * torch.pow(y_true - mu, 2) / (2 * y_true * torch.pow(mu, 2))
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


def weibull_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    c = y_pred[:, 0]  # shape parameter
    c = torch.reshape(c, [-1, 1])

    scale = y_pred[:, 1]  # scale parameter
    scale = torch.reshape(scale, [-1, 1])

    log_c = torch.log(c)
    log_scale = torch.log(scale)
    log_true = torch.log(y_true)

    help1 = torch.pow(y_true / scale, c)
    
    lh = (log_c - c * log_scale + (c - 1) * log_true - help1)
    
    return -lh.mean()


@torch.jit.script
def normal_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    # y_pred[:,0] = mu (real), y_pred[:,1] = sigma (>0)
    mu = y_pred[:, 0]
    mu = torch.reshape(mu, [-1, 1])
    sigma = y_pred[:, 1]
    sigma = torch.reshape(sigma, [-1, 1])
    # logpdf = -0.5*log(2*pi) - log(sigma) - (y-mu)^2/(2*sigma^2)
    diff = y_true - mu
    logpdf = -0.5 * torch.log(2.0 * torch.pi) - torch.log(sigma) - torch.pow(diff, 2) / (2.0 * torch.pow(sigma, 2))
    return -logpdf.mean()


@torch.jit.script
def gamma_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    # y_pred[:,0] = k (shape>0), y_pred[:,1] = theta (scale>0)
    k = y_pred[:, 0]
    k = torch.reshape(k, [-1, 1])
    theta = y_pred[:, 1]
    theta = torch.reshape(theta, [-1, 1])
    # logpdf = (k-1)*log(x) - x/theta - k*log(theta) - log(Gamma(k))
    log_x = torch.log(y_true)
    lgamma_k = torch.lgamma(k)
    logpdf = (k - 1.0) * log_x - y_true / theta - k * torch.log(theta) - lgamma_k
    return -logpdf.mean()


@torch.jit.script
def cauchy_loss(y_true: torch.Tensor, y_pred: torch.Tensor):

    scale = y_pred[:, 0]
    scale = torch.reshape(scale, [-1, 1])
    # logpdf = -log(pi) - log(scale) - log(1 + ((x-loc)/scale)^2)
    z = (y_true) / scale
    logpdf = -torch.log(torch.pi) - torch.log(scale) - torch.log(1.0 + torch.pow(z, 2))
    return -logpdf.mean()


@torch.jit.script
def levy_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    # y_pred[:,0] = loc (real), y_pred[:,1] = scale (>0) ; support y_true > loc

    scale = y_pred[:, 0]
    scale = torch.reshape(scale, [-1, 1])
    shifted = y_true
    # To avoid log of non-positive due to numerical issues, clamp shifted
    shifted = torch.clamp(shifted, min=1e-12)
    # logpdf = 0.5*log(scale) - 0.5*log(2*pi) - scale/(2*shifted) - 1.5*log(shifted)
    logpdf = 0.5 * torch.log(scale) - 0.5 * torch.log(torch.tensor(2.0 * torch.pi)) - scale / (2.0 * shifted) - 1.5 * torch.log(shifted)
    return -logpdf.mean()


@torch.jit.script
def beta_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    # y_pred[:,0] = scale (>0), y_pred[:,1] = alpha (>0), y_pred[:,2] = beta (>0)
    # Support: y_true in (0, scale)
    scale = y_pred[:, 0]
    scale = torch.reshape(scale, [-1, 1])
    alpha = y_pred[:, 1]
    alpha = torch.reshape(alpha, [-1, 1])
    beta = y_pred[:, 2]
    beta = torch.reshape(beta, [-1, 1])

    # Transform to standard Beta support via x = y/scale
    eps = 1e-12
    x = y_true / scale
    x = torch.clamp(x, eps, 1.0 - eps)

    # logpdf_beta_scaled(y; a,b,scale) = -log(scale) + logpdf_beta(x; a,b)
    logpdf = (
        -torch.log(scale)
        + (alpha - 1.0) * torch.log(x)
        + (beta - 1.0) * torch.log(1.0 - x)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + torch.lgamma(alpha + beta)
    )
    return -logpdf.mean()


@torch.jit.script
def betaprime_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    # y_pred[:,0] = scale (>0), y_pred[:,1] = alpha (>0), y_pred[:,2] = beta (>0)
    # Support: y_true > 0
    scale = y_pred[:, 0]
    scale = torch.reshape(scale, [-1, 1])
    alpha = y_pred[:, 1]
    alpha = torch.reshape(alpha, [-1, 1])
    beta = y_pred[:, 2]
    beta = torch.reshape(beta, [-1, 1])

    eps = 1e-12
    x = y_true / scale
    x = torch.clamp(x, min=eps)

    # logpdf_scaled(x*scale) = -log(scale) + (alpha-1)*log(x) - (alpha+beta)*log(1+x) - lgamma(alpha) - lgamma(beta) + lgamma(alpha+beta)
    logpdf = (
        -torch.log(scale)
        + (alpha - 1.0) * torch.log(x)
        - (alpha + beta) * torch.log1p(x)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + torch.lgamma(alpha + beta)
    )
    return -logpdf.mean()


@torch.jit.script
def lomax_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
    # y_pred[:,0] = alpha (>0), y_pred[:,1] = scale (>0); support y_true >= 0
    alpha = y_pred[:, 0]
    alpha = torch.reshape(alpha, [-1, 1])
    scale = y_pred[:, 1]
    scale = torch.reshape(scale, [-1, 1])

    z = y_true / scale
    z = torch.clamp(z, min=0.0)
    # logpdf = log(alpha) - log(scale) - (alpha+1)*log(1+z)
    logpdf = torch.log(alpha) - torch.log(scale) - (alpha + 1.0) * torch.log1p(z)
    return -logpdf.mean()
