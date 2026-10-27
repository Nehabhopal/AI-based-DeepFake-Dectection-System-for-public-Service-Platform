from io import BytesIO
from werkzeug.datastructures import FileStorage
import pytest
from backend.services.image_utils import validate_image, ImageValidationError
from PIL import Image

def test_rejects_unsupported_extension():
    f=FileStorage(stream=BytesIO(b"hello"),filename="bad.exe")
    with pytest.raises(ImageValidationError): validate_image(f,1024)

def test_rejects_corrupted_image():
    f=FileStorage(stream=BytesIO(b"not an image"),filename="x.png")
    with pytest.raises(ImageValidationError): validate_image(f,1024)

def test_accepts_png():
    buf=BytesIO(); Image.new("RGB",(20,20)).save(buf,format="PNG"); buf.seek(0)
    f=FileStorage(stream=buf,filename="x.png",content_type="image/png")
    assert validate_image(f,1024).startswith(b"\x89PNG")
