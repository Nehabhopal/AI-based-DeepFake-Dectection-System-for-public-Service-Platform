import torch
from ml.model import build_model
from ml.gradcam import generate_gradcam
from PIL import Image

def test_gradcam_creates_real_output_files(tmp_path):
    model=build_model().eval()
    tensor=torch.randn(1,3,224,224)
    heat=tmp_path/"heat.jpg"; overlay=tmp_path/"overlay.jpg"
    generate_gradcam(model,tensor,Image.new("RGB",(128,128)),0,heat,overlay,torch.device("cpu"))
    assert heat.is_file() and heat.stat().st_size>0
    assert overlay.is_file() and overlay.stat().st_size>0
