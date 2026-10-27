import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from ml.model import build_model
from ml.gradcam import generate_gradcam

class ModelUnavailable(RuntimeError):
    pass

class PredictionService:
    def __init__(self, model_path: Path, metadata_path: Path, output_dir: Path):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metadata = self._metadata()

    def _metadata(self):
        if self.metadata_path.exists():
            try:
                return json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
        return {"model_name": "ResNet18", "framework": "PyTorch", "version": "1.0.0",
                "classes": {"0": "REAL", "1": "DEEPFAKE"}, "input_size": 224,
                "status": "NOT_TRAINED"}

    def available(self):
        return self.model_path.is_file()

    def load(self):
        if not self.available():
            raise ModelUnavailable("Trained model is not available. Please train the model first: python ml/train.py")
        if self.model is None:
            self.model = build_model(num_classes=2)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
            if isinstance(checkpoint, dict) and "metadata" in checkpoint:
                self.metadata.update(checkpoint["metadata"])
        return self.model

    def predict(self, image_path: Path, output_prefix: str):
        model = self.load()
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        idx = int(np.argmax(probs))
        class_names = ["REAL", "DEEPFAKE"]
        prediction = class_names[idx]
        heat_path = self.output_dir / f"{output_prefix}_gradcam.jpg"
        overlay_path = self.output_dir / f"{output_prefix}_overlay.jpg"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        generate_gradcam(model, tensor, image, idx, heat_path, overlay_path, self.device)
        return {
            "prediction": prediction, "confidence": float(probs[idx] * 100),
            "real_probability": float(probs[0] * 100), "deepfake_probability": float(probs[1] * 100),
            "gradcam_path": heat_path, "overlay_path": overlay_path,
            "model_version": str(self.metadata.get("version", "1.0.0"))
        }
