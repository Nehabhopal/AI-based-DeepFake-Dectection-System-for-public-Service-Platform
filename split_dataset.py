
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent / "DeepSafe" / "dataset_rvf10k"
random.seed(42)

for label in ["REAL", "DEEPFAKE"]:
    source = ROOT / "val" / label
    test_dir = ROOT / "test" / label
    test_dir.mkdir(parents=True, exist_ok=True)

    images = [
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in
        {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ]

    random.shuffle(images)
    test_images = images[:300]

    for image in test_images:
        shutil.move(str(image), str(test_dir / image.name))

    print(
        f"{label}: moved {len(test_images)} images to test; "
        f"{len(images) - len(test_images)} remain in validation"
    )

print("Test split created successfully.")
