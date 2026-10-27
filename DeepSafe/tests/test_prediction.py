from backend.services.model_service import PredictionService, ModelUnavailable
import pytest

def test_missing_checkpoint_fails_explicitly(tmp_path):
    svc=PredictionService(tmp_path/"missing.pth",tmp_path/"metadata.json",tmp_path/"out")
    assert not svc.available()
    with pytest.raises(ModelUnavailable,match="train the model first"):
        svc.load()
