"""Tests for predictor loss functions."""

import pytest
from asf.predictors.utils.losses import wmse, mse, bpr_loss, tml_loss

torch = pytest.importorskip("torch")


class TestLossFunctions:
    """Tests for loss functions."""

    def test_wmse(self):
        """Test weighted MSE loss."""
        input_tensor = torch.tensor([[1.0], [2.0], [3.0]])
        target = torch.tensor([[1.5], [2.5], [3.5]])
        weights = torch.tensor([[1.0], [2.0], [1.0]])

        loss = wmse(input_tensor, target, weights)

        assert torch.is_tensor(loss)
        assert loss.item() > 0

    def test_wmse_uniform_weights(self):
        """Test WMSE with uniform weights equals MSE."""
        input_tensor = torch.tensor([[1.0], [2.0], [3.0]])
        target = torch.tensor([[1.5], [2.5], [3.5]])
        weights = torch.ones(3, 1)

        wmse_loss = wmse(input_tensor, target, weights)
        mse_loss = torch.nn.functional.mse_loss(input_tensor, target)

        assert torch.isclose(wmse_loss, mse_loss, rtol=1e-5)

    def test_mse_triplet(self):
        """Test MSE for triplet predictions."""
        y_pred = torch.tensor([[1.0], [2.0]])
        y_pred_s = torch.tensor([[0.5], [1.5]])
        y_pred_l = torch.tensor([[1.5], [2.5]])
        yc = torch.tensor([[1.0], [2.0]])
        ys = torch.tensor([[0.5], [1.5]])
        yl = torch.tensor([[1.5], [2.5]])

        loss = mse(y_pred, y_pred_s, y_pred_l, yc, ys, yl)

        assert torch.is_tensor(loss)
        assert loss.item() == 0.0  # Perfect predictions

    def test_mse_triplet_with_error(self):
        """Test MSE for triplet with prediction error."""
        y_pred = torch.tensor([[1.0], [2.0]])
        y_pred_s = torch.tensor([[0.5], [1.5]])
        y_pred_l = torch.tensor([[1.5], [2.5]])
        yc = torch.tensor([[2.0], [3.0]])  # Different targets
        ys = torch.tensor([[1.5], [2.5]])
        yl = torch.tensor([[2.5], [3.5]])

        loss = mse(y_pred, y_pred_s, y_pred_l, yc, ys, yl)

        assert torch.is_tensor(loss)
        assert loss.item() > 0

    def test_bpr_loss(self):
        """Test Bayesian Personalized Ranking loss."""
        y_pred = torch.tensor([[1.0], [2.0]])
        y_pred_s = torch.tensor([[0.5], [1.5]])  # Smaller
        y_pred_l = torch.tensor([[1.5], [2.5]])  # Larger
        yc = torch.tensor([[1.0], [2.0]])
        ys = torch.tensor([[0.5], [1.5]])
        yl = torch.tensor([[1.5], [2.5]])

        loss = bpr_loss(y_pred, y_pred_s, y_pred_l, yc, ys, yl)

        assert torch.is_tensor(loss)
        assert loss.item() >= 0

    def test_tml_loss(self):
        """Test triplet margin loss."""
        y_pred = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
        y_pred_s = torch.tensor([[0.5, 1.5], [1.5, 2.5]])  # Similar (smaller)
        y_pred_l = torch.tensor([[2.0, 3.0], [3.0, 4.0]])  # Dissimilar (larger)
        yc = torch.tensor([[1.0], [2.0]])
        ys = torch.tensor([[0.5], [1.5]])
        yl = torch.tensor([[1.5], [2.5]])

        loss = tml_loss(y_pred, y_pred_s, y_pred_l, yc, ys, yl)

        assert torch.is_tensor(loss)
        assert loss.item() >= 0

    def test_tml_loss_custom_margin(self):
        """Test triplet margin loss with custom margin."""
        y_pred = torch.tensor([[1.0, 2.0]])
        y_pred_s = torch.tensor([[0.5, 1.5]])
        y_pred_l = torch.tensor([[2.0, 3.0]])
        yc = torch.tensor([[1.0]])
        ys = torch.tensor([[0.5]])
        yl = torch.tensor([[1.5]])

        loss = tml_loss(y_pred, y_pred_s, y_pred_l, yc, ys, yl, margin=2.0, p=1)

        assert torch.is_tensor(loss)
