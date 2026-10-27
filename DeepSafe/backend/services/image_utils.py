from pathlib import Path
from PIL import Image, UnidentifiedImageError
import io
import cv2

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

class ImageValidationError(ValueError):
    pass

def validate_image(file_storage, max_bytes):
    if file_storage is None or not file_storage.filename:
        raise ImageValidationError("Choose an image file to analyze.")
    ext = Path(file_storage.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError("Unsupported image type. Use JPG, JPEG, PNG, or WEBP.")
    raw = file_storage.read(max_bytes + 1)
    file_storage.stream.seek(0)
    if not raw:
        raise ImageValidationError("The uploaded file is empty.")
    if len(raw) > max_bytes:
        raise ImageValidationError(f"Image exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw))
        if image.format not in ALLOWED_FORMATS:
            raise ImageValidationError("File content is not a supported image format.")
        image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, ImageValidationError):
            raise
        raise ImageValidationError("The image is corrupted or cannot be decoded.") from exc
    return raw

def image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if min(h, w) < 80:
        quality = "LOW"
    elif cv2.Laplacian(gray, cv2.CV_64F).var() < 35:
        quality = "LOW"
    else:
        quality = "GOOD"
    return quality, w, h

def detect_face(image):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0, len(faces)
