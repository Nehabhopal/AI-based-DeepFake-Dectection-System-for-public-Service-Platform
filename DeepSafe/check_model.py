
import torch
from collections import Counter
from torch.utils.data import DataLoader

from ml.dataset import load_split
from ml.model import build_model

DATA_DIR = "./dataset_rvf10k"
MODEL_PATH = "./models/deepfake_resnet18_best.pth"

device = torch.device("cpu")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model = build_model(2)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_ds = load_split(DATA_DIR, "test", train=False)

test_loader = DataLoader(
    test_ds,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

actual_counts = Counter()
predicted_counts = Counter()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        actual_counts.update(labels.tolist())
        predicted_counts.update(predictions.tolist())

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print("Class mapping:", test_ds.class_to_idx)
print("Actual counts:", actual_counts)
print("Predicted counts:", predicted_counts)
print(f"Test accuracy: {correct / total:.2%}")
print(f"Correct predictions: {correct}/{total}")
