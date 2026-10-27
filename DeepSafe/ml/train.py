
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ml.dataset import load_split
from ml.model import build_model


ROOT = Path(__file__).resolve().parents[1]


def balanced_subset(dataset, per_class, seed=42):
    """Randomly select a balanced subset from both classes."""
    rng = random.Random(seed)
    selected = []

    for label in (0, 1):
        class_indices = [
            index
            for index, (_, sample_label) in enumerate(dataset.samples)
            if sample_label == label
        ]

        if len(class_indices) < per_class:
            raise ValueError(
                f"Class {label} has only {len(class_indices)} images."
            )

        selected.extend(rng.sample(class_indices, per_class))

    rng.shuffle(selected)
    return Subset(dataset, selected)


def count_classes(dataset):
    """Count classes correctly for both full datasets and Subsets."""
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset

    if isinstance(dataset, Subset):
        labels = [base_dataset.samples[i][1] for i in dataset.indices]
    else:
        labels = [label for _, label in dataset.samples]

    return {
        name: sum(
            1 for label in labels
            if base_dataset.classes[label] == name
        )
        for name in base_dataset.classes
    }


def run_evaluation(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_images += labels.size(0)

    return {
        "loss": total_loss / total_images,
        "accuracy": total_correct / total_images,
        "images": total_images,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a ResNet18 deepfake detector."
    )
    parser.add_argument("--data", default=str(ROOT / "dataset_rvf10k"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use a small balanced subset to verify the training pipeline."
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load three separate dataset splits.
    train_ds = load_split(args.data, "train", train=True)
    val_ds = load_split(args.data, "val", train=False)
    test_ds = load_split(args.data, "test", train=False)

    # Small balanced subsets are ONLY for checking that the pipeline works.
    if args.quick_test:
        train_ds = balanced_subset(train_ds, 20, args.seed)
        val_ds = balanced_subset(val_ds, 10, args.seed)

    for name, dataset in [
        ("train", train_ds),
        ("val", val_ds),
        ("test", test_ds),
    ]:
        print(f"{name} counts: {count_classes(dataset)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = build_model(2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr
    )

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    output_dir = ROOT / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "deepfake_resnet18_best.pth"
    metadata_path = models_dir / "model_metadata.json"

    best_val_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_images = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_images += labels.size(0)

        train_loss = total_loss / total_images
        train_accuracy = total_correct / total_images

        validation = run_evaluation(
            model, val_loader, criterion, device
        )

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
        }
        history.append(record)
        print(record, flush=True)

        # Select the model using validation loss, not test results.
        if validation["loss"] < best_val_loss:
            best_val_loss = validation["loss"]

            metadata = {
                "model_name": "ResNet18",
                "framework": "PyTorch",
                "version": "1.0.0",
                "classes": {"0": "REAL", "1": "DEEPFAKE"},
                "input_size": 224,
                "status": "TRAINED",
                "training_date": datetime.now().isoformat(timespec="seconds"),
                "device": str(device),
                "seed": args.seed,
                "epochs_requested": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "quick_test_mode": args.quick_test,
                "dataset_counts": {
                    "train": count_classes(train_ds),
                    "val": count_classes(val_ds),
                    "test": count_classes(test_ds),
                },
            }

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata": metadata,
                },
                model_path,
            )

            metadata_path.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

    history_path = output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    # Reload the best validation checkpoint before final test evaluation.
    checkpoint = torch.load(
        model_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_results = run_evaluation(
        model, test_loader, criterion, device
    )

    print("\nFinal held-out test results:")
    print(f"Test images: {test_results['images']}")
    print(f"Test loss: {test_results['loss']:.4f}")
    print(f"Test accuracy: {test_results['accuracy']:.2%}")

    test_path = output_dir / "test_results.json"
    test_path.write_text(
        json.dumps(test_results, indent=2),
        encoding="utf-8",
    )

    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Test results saved to: {test_path}")
    print("Training and evaluation finished.")


if __name__ == "__main__":
    main()
