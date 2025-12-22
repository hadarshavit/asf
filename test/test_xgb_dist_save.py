import numpy as np
import xgboost as xgb
from asf.epm.xgb_dist import XGBDistNet


def test_xgbdist_save_and_load(tmp_path):
    # tiny dataset
    X = np.random.rand(20, 4).astype('float32')
    y = (np.random.rand(20, 1)*2).astype('float32')

    m = XGBDistNet()
    # Fit a tiny model so that save_model works
    m.model = xgb.XGBRegressor(objective=m.objective, num_target=m.n_loss_params)
    m.model.fit(X, y)

    model_path = str(tmp_path / "test.model")
    m.save(model_path)

    # Load back
    m2 = XGBDistNet.load(model_path)
    assert isinstance(m2, XGBDistNet)
    assert hasattr(m2, 'start_values')
    assert m2.start_values.shape == m.start_values.shape
