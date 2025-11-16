import math

import pytest
import torch

from asf.predictors.utils.losses import lognorm_loss, exp_loss, invgauss_loss


def _rand_pos(shape, low=0.1, high=3.0):
    return (high - low) * torch.rand(shape) + low


def test_lognorm_loss_matches_torch_up_to_constant():
    torch.manual_seed(0)

    N = 256
    y_true = _rand_pos((N, 1), low=0.1, high=5.0)

    # Our parameterization: y_pred[:,0] = s (sigma>0), y_pred[:,1] = scale>0 with mu=log(scale)
    s = _rand_pos((N, 1), low=0.2, high=1.5)
    scale = _rand_pos((N, 1), low=0.5, high=3.0)
    y_pred = torch.cat([s, scale], dim=1)

    ours = float(lognorm_loss(y_true, y_pred))

    loc = torch.log(scale).squeeze(-1)
    ref = torch.distributions.LogNormal(loc, s.squeeze(-1))
    torch_ll = float(-ref.log_prob(y_true.squeeze(-1)).mean())

    # Our loss omits the constant 0.5*log(2*pi)
    expected_diff = 0.5 * math.log(2.0 * math.pi)
    assert abs((torch_ll - ours) - expected_diff) < 1e-4


def test_exp_loss_matches_torch_exact():
    torch.manual_seed(0)

    N = 256
    y_true = _rand_pos((N, 1), low=0.0, high=5.0)
    scale_param = _rand_pos((N, 1), low=0.2, high=3.0)

    ours = float(exp_loss(y_true, scale_param))

    rate = 1.0 / scale_param.squeeze(-1)
    ref = torch.distributions.Exponential(rate)
    torch_ll = float(-ref.log_prob(y_true.squeeze(-1)).mean())

    assert abs(torch_ll - ours) < 1e-6


@pytest.mark.skipif(
    not hasattr(torch.distributions, "InverseGaussian"),
    reason="PyTorch InverseGaussian not available in this environment",
)
def test_invgauss_loss_matches_torch_with_param_mapping():
    torch.manual_seed(0)

    N = 256
    y_true = _rand_pos((N, 1), low=0.1, high=5.0)

    # Our parameterization: y_pred[:,0] = mu (>0), y_pred[:,1] = scale (= lambda >0)
    mu = _rand_pos((N, 1), low=0.2, high=3.0)
    lam = _rand_pos((N, 1), low=0.5, high=4.0)
    y_pred = torch.cat([mu, lam], dim=1)

    ours = float(invgauss_loss(y_true, y_pred))

    # Mapping to torch: mean = mu * lam, scale = lam
    mean_param = (mu * lam).squeeze(-1)
    scale_param = lam.squeeze(-1)

    ref = torch.distributions.InverseGaussian(mean_param, scale_param)
    torch_ll = float(-ref.log_prob(y_true.squeeze(-1)).mean())

    # Our implementation retains all non-constant terms, so it should match exactly
    assert abs(torch_ll - ours) < 1e-5
