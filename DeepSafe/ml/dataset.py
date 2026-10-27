from pathlib import Path
from torchvision import datasets, transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def build_transforms(train=False):
    ops = [transforms.Resize((160,160))]
    if train:
        ops += [transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1)]
    ops += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(ops)

class RealFirstImageFolder(datasets.ImageFolder):
    # Explicit mapping is important: ImageFolder's default alphabetical order
    # would place DEEPFAKE before REAL and silently invert the model labels.
    def find_classes(self, directory):
        classes = ["REAL", "DEEPFAKE"]
        found = {p.name for p in Path(directory).iterdir() if p.is_dir()}
        if set(classes) != found:
            raise ValueError(f"{directory} must contain exactly REAL and DEEPFAKE folders; found {sorted(found)}")
        return classes, {"REAL": 0, "DEEPFAKE": 1}

def load_split(root, split, train=False):
    path = Path(root) / split
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset split folder is missing: {path}")
    dataset = RealFirstImageFolder(path, transform=build_transforms(train))
    if not dataset:
        raise ValueError(f"No readable images found under {path}")
    return dataset

def count_classes(dataset):
    return {name: sum(1 for _, label in dataset.samples if dataset.classes[label] == name) for name in dataset.classes}