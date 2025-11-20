import math

import numpy as np
import pytest
import torch
from scipy import stats

from asf.predictors.utils.losses import (
    exp_loss,
    invgauss_loss,
    lognorm_loss,
    weibull_loss,
    normal_loss,
    gamma_loss,
    cauchy_loss,
    levy_loss,
    beta_loss,
    betaprime_loss,
    lomax_loss,
)


def _rand_pos(shape, low=0.1, high=3.0):
    return (high - low) * torch.rand(shape) + low


def test_lognorm_loss_matches_scipy():
    """Test that lognorm_loss matches scipy.stats.lognorm up to constants."""
    torch.manual_seed(42)
    np.random.seed(42)

    N = 256
    y_true = _rand_pos((N, 1), low=0.1, high=5.0)

    # Our parameterization: y_pred[:,0] = s (sigma>0), y_pred[:,1] = scale>0 with mu=log(scale)
    s = _rand_pos((N, 1), low=0.2, high=1.5)
    scale = _rand_pos((N, 1), low=0.5, high=3.0)
    y_pred = torch.cat([s, scale], dim=1)

    ours = float(lognorm_loss(y_true, y_pred))

    # scipy.stats.lognorm parameterization: lognorm(s, loc=0, scale=scale)
    # where s is the shape parameter (sigma) and scale relates to the median
    # scipy's scale is exp(mu), which matches our scale parameter
    y_true_np = y_true.numpy().flatten()
    s_np = s.numpy().flatten()
    scale_np = scale.numpy().flatten()

    # Compute negative log-likelihood using scipy
    scipy_nll = 0.0
    for i in range(N):
        logpdf = stats.lognorm.logpdf(y_true_np[i], s=s_np[i], scale=scale_np[i])
        scipy_nll -= logpdf
    scipy_nll /= N

    assert abs((scipy_nll - ours)) < 1e-5


def test_exp_loss_matches_scipy():
    """Test that exp_loss matches scipy.stats.expon exactly."""
    torch.manual_seed(42)
    np.random.seed(42)

    N = 256
    y_true = _rand_pos((N, 1), low=0.0, high=5.0)
    scale_param = _rand_pos((N, 1), low=0.2, high=3.0)

    ours = float(exp_loss(y_true, scale_param))

    # Our implementation: takes scale_param as input, then computes rate = 1/scale_param
    # Then uses: log(rate) - y * rate = -log(scale_param) - y / scale_param
    # scipy.stats.expon uses scale parameter directly (scale = 1/rate)
    y_true_np = y_true.numpy().flatten()
    scale_np = scale_param.numpy().flatten()

    # Compute negative log-likelihood using scipy
    scipy_nll = 0.0
    for i in range(N):
        # Our scale_param is the input, and we compute 1/scale_param as the rate
        # scipy's scale = 1/rate, so scipy_scale = scale_param
        scipy_scale = scale_np[i]
        logpdf = stats.expon.logpdf(y_true_np[i], scale=scipy_scale)
        scipy_nll -= logpdf
    scipy_nll /= N

    assert abs(scipy_nll - ours) < 1e-5


def test_invgauss_loss_matches_scipy():
    """Test that invgauss_loss matches scipy.stats.invgauss up to constants."""
    torch.manual_seed(42)
    np.random.seed(42)

    N = 256
    y_true = _rand_pos((N, 1), low=0.1, high=5.0)

    # Our parameterization: y_pred[:,0] = mu (>0), y_pred[:,1] = scale = lambda (>0)
    # Looking at the code:
    # help1 = log(scale)  <- this is log(lambda), but standard form has 0.5*log(lambda)
    # help3 = scale * (y-mu)^2 / (2*y*mu^2)  <- this is lambda * (y-mu)^2 / (2*y*mu^2)
    # So scale = lambda directly
    # Our log-likelihood omits: 0.5*log(lambda) term and 0.5*log(2*pi) constant
    mu = _rand_pos((N, 1), low=0.2, high=3.0)
    lam = _rand_pos((N, 1), low=0.5, high=4.0)  # lambda parameter
    y_pred = torch.cat([mu, lam], dim=1)

    ours = float(invgauss_loss(y_true, y_pred))

    # scipy.stats.invgauss(mu, loc=0, scale) gives inverse Gaussian with:
    # - mean = scale * mu
    # - For standard form with mean=M and shape=lambda: use invgauss(mu=M/lambda, scale=lambda)
    y_true_np = y_true.numpy().flatten()
    mu_np = mu.numpy().flatten()
    lam_np = lam.numpy().flatten()

    # Compute negative log-likelihood using scipy
    scipy_nll = 0.0
    for i in range(N):
        logpdf = stats.invgauss.logpdf(
            y_true_np[i], mu=mu_np[i], scale=lam_np[i]
        )
        scipy_nll -= logpdf
    scipy_nll /= N

    diff = scipy_nll - ours

    assert abs(diff) < 1e-5


def test_weibull_loss_matches_scipy():
    """Test that weibull_loss matches scipy.stats.weibull_min."""
    torch.manual_seed(42)
    np.random.seed(42)

    N = 256
    y_true = _rand_pos((N, 1), low=0.1, high=5.0)

    # Our parameterization: y_pred[:,0] = c (shape>0), y_pred[:,1] = scale (scale>0)
    c = _rand_pos((N, 1), low=0.5, high=3.0)
    scale = _rand_pos((N, 1), low=0.5, high=3.0)
    y_pred = torch.cat([c, scale], dim=1)

    ours = float(weibull_loss(y_true, y_pred))

    # scipy.stats.weibull_min parameterization: weibull_min(c, loc=0, scale=scale)
    # where c is the shape parameter and scale is the scale parameter
    y_true_np = y_true.numpy().flatten()
    c_np = c.numpy().flatten()
    scale_np = scale.numpy().flatten()

    # Compute negative log-likelihood using scipy
    scipy_nll = 0.0
    for i in range(N):
        logpdf = stats.weibull_min.logpdf(y_true_np[i], c=c_np[i], scale=scale_np[i])
        scipy_nll -= logpdf
    scipy_nll /= N

    # Should match exactly as both use the same formula
    assert abs(scipy_nll - ours) < 1e-5


def test_normal_loss_matches_scipy():
    """Test that normal_loss matches scipy.stats.norm."""
    torch.manual_seed(42)
    np.random.seed(42)

    N = 256
    # Sample y_true in a symmetric interval
    y_true = 4.0 * torch.rand((N, 1)) - 2.0  # in [-2,2]
    mu = 4.0 * torch.rand((N, 1)) - 2.0
    sigma = _rand_pos((N, 1), low=0.2, high=2.0)
    y_pred = torch.cat([mu, sigma], dim=1)

    ours = float(normal_loss(y_true, y_pred))

    y_true_np = y_true.numpy().flatten()
    mu_np = mu.numpy().flatten()
    sigma_np = sigma.numpy().flatten()

    scipy_nll = 0.0
    for i in range(N):
        logpdf = stats.norm.logpdf(y_true_np[i], loc=mu_np[i], scale=sigma_np[i])
        scipy_nll -= logpdf
    scipy_nll /= N

    assert abs(scipy_nll - ours) < 1e-5


def test_gamma_loss_matches_scipy():
    """Test that gamma_loss matches scipy.stats.gamma."""
    torch.manual_seed(42)
    np.random.seed(42)

    N = 256
    y_true = _rand_pos((N, 1), low=0.05, high=5.0)
    k = _rand_pos((N, 1), low=0.5, high=5.0)  # shape
    theta = _rand_pos((N, 1), low=0.3, high=3.0)  # scale
    y_pred = torch.cat([k, theta], dim=1)

    ours = float(gamma_loss(y_true, y_pred))

    y_true_np = y_true.numpy().flatten()
    k_np = k.numpy().flatten()
    theta_np = theta.numpy().flatten()

    scipy_nll = 0.0
    for i in range(N):
        logpdf = stats.gamma.logpdf(y_true_np[i], a=k_np[i], scale=theta_np[i])
        scipy_nll -= logpdf
    scipy_nll /= N

    assert abs(scipy_nll - ours) < 1e-5


def test_cauchy_loss_matches_scipy():
    """Test that cauchy_loss matches scipy.stats.cauchy."""
    torch.manual_seed(42)
    np.random.seed(42)

    N = 256
    y_true = 6.0 * torch.rand((N, 1)) - 3.0  # in [-3,3]
    loc = torch.zeros((N, 1))
    scale = _rand_pos((N, 1), low=0.2, high=3.0)
    y_pred = torch.cat([ scale], dim=1)

    ours = float(cauchy_loss(y_true, y_pred))

    y_true_np = y_true.numpy().flatten()
    loc_np = loc.numpy().flatten()
    scale_np = scale.numpy().flatten()

    scipy_nll = 0.0
    for i in range(N):
        logpdf = stats.cauchy.logpdf(y_true_np[i], loc=loc_np[i], scale=scale_np[i])
        scipy_nll -= logpdf
    scipy_nll /= N

    assert abs(scipy_nll - ours) < 1e-5


def test_levy_loss_matches_scipy():
    """Test that levy_loss matches scipy.stats.levy."""
    torch.manual_seed(42)
    np.random.seed(42)

    N = 256
    loc = torch.zeros((N, 1))
    # Ensure y_true > loc
    delta = _rand_pos((N, 1), low=0.1, high=5.0)
    y_true = loc + delta
    scale = _rand_pos((N, 1), low=0.1, high=3.0)
    y_pred = torch.cat([scale], dim=1)

    ours = float(levy_loss(y_true, y_pred))

    y_true_np = y_true.numpy().flatten()
    loc_np = loc.numpy().flatten()
    scale_np = scale.numpy().flatten()

    scipy_nll = 0.0
    for i in range(N):
        logpdf = stats.levy.logpdf(y_true_np[i], loc=loc_np[i], scale=scale_np[i])
        scipy_nll -= logpdf
    scipy_nll /= N

    assert abs(scipy_nll - ours) < 1e-5


def test_beta_loss_matches_scipy():
    """Test that beta_loss matches scipy.stats.beta."""
    torch.manual_seed(42)
    np.random.seed(42)

    N = 256
    # Sample parameters and y_true consistent with support y in (0, scale)
    alpha = _rand_pos((N, 1), low=0.5, high=5.0)
    beta = _rand_pos((N, 1), low=0.5, high=5.0)
    scale = _rand_pos((N, 1), low=0.1, high=3.0)
    # Sample u in (0,1) away from boundaries, then y = scale * u
    u = torch.clamp(torch.rand((N, 1)), 1e-6, 1.0 - 1e-6)
    y_true = scale * u
    y_pred = torch.cat([scale, alpha, beta], dim=1)

    ours = float(beta_loss(y_true, y_pred))

    y_true_np = y_true.numpy().flatten()
    a_np = alpha.numpy().flatten()
    b_np = beta.numpy().flatten()
    scale_np = scale.numpy().flatten()

    scipy_nll = 0.0
    for i in range(N):
        logpdf = stats.beta.logpdf(y_true_np[i], a=a_np[i], b=b_np[i], scale=scale_np[i])
        scipy_nll -= logpdf
    scipy_nll /= N

    assert abs(scipy_nll - ours) < 1e-5


def test_betaprime_loss_matches_scipy():
    """Test that betaprime_loss matches scipy.stats.betaprime with scaling."""
    torch.manual_seed(42)
    np.random.seed(42)

    N = 256
    y_true = _rand_pos((N, 1), low=0.05, high=5.0)
    alpha = _rand_pos((N, 1), low=0.5, high=5.0)
    beta = _rand_pos((N, 1), low=0.5, high=5.0)
    scale = _rand_pos((N, 1), low=0.2, high=3.0)
    y_pred = torch.cat([scale, alpha, beta], dim=1)

    ours = float(betaprime_loss(y_true, y_pred))

    y_true_np = y_true.numpy().flatten()
    alpha_np = alpha.numpy().flatten()
    beta_np = beta.numpy().flatten()
    scale_np = scale.numpy().flatten()

    scipy_nll = 0.0
    for i in range(N):
        z = y_true_np[i] / scale_np[i]
        logpdf_std = stats.betaprime.logpdf(z, a=alpha_np[i], b=beta_np[i])
        scipy_nll -= logpdf_std
    scipy_nll /= N

    assert abs(scipy_nll - ours) < 1e-5


def test_lomax_loss_matches_scipy():
    """Test that lomax_loss matches scipy.stats.lomax exactly."""
    torch.manual_seed(42)
    np.random.seed(42)

    N = 256
    y_true = _rand_pos((N, 1), low=0.0, high=5.0)  # support x>=0
    alpha = _rand_pos((N, 1), low=0.5, high=5.0)  # shape
    scale = _rand_pos((N, 1), low=0.2, high=3.0)  # scale
    y_pred = torch.cat([alpha, scale], dim=1)

    ours = float(lomax_loss(y_true, y_pred))

    y_true_np = y_true.numpy().flatten()
    alpha_np = alpha.numpy().flatten()
    scale_np = scale.numpy().flatten()

    scipy_nll = 0.0
    for i in range(N):
        logpdf = stats.lomax.logpdf(y_true_np[i], c=alpha_np[i], scale=scale_np[i])
        scipy_nll -= logpdf
    scipy_nll /= N

    assert abs(scipy_nll - ours) < 1e-5

