#!/usr/bin/env python3
"""Image classification: preserve sample identity through evaluation.

Required packages: torch, torchvision, scikit-learn, numpy, Pillow, matplotlib.
Use a compatible torch / torchvision pair for your operating system.

macOS / Linux examples:
  python examples-en.py
  python examples-en.py --demo --device cpu --output-dir runs/demo-01
  python examples-en.py --data-root ./chest_xray/train --epochs 3 --device auto

With no data argument, this program generates small synthetic RGB images in a
temporary directory. This checks the processing flow, not X-ray performance.
With --output-dir, the synthetic files and reports remain in that new directory.
The chosen output directory must not already exist. Existing data is read only.

For real data, --data-root must contain one folder per class. The program splits
that folder by image into training and validation sets. It does not infer patient
IDs or guarantee patient-level independence. It does not evaluate a held-out test
set, save the best model, or download data / pretrained weights.

The numbers 222, 48, 12, and 765 below are a separate arithmetic example supplied
with the article. They are not results produced by training this example model.
"""

import argparse
import csv
import json
import random
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class TinyCNN(nn.Module):
    """A small model for the example: (B, 3, 64, 64) -> (B, K) logits."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 x 64 -> 32 x 32
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 x 32 -> 16 x 16
        )
        self.classifier = nn.Linear(16 * 16 * 16, num_classes)

    def forward(self, images):
        features = self.features(images)
        flattened = torch.flatten(features, start_dim=1)
        # CrossEntropyLoss consumes logits; no Softmax is needed here.
        return self.classifier(flattened)


def choose_device(requested):
    """Prefer CUDA, then Apple MPS, then CPU when 'auto' is selected."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this Python environment.")
    return torch.device(requested)


def seed_everything(seed):
    """Set local random seeds; identical results across devices are not promised."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_demo_images(root, seed):
    """Create 24 blue and 24 red images, with small random pixel differences."""
    rng = np.random.default_rng(seed)
    for name, rgb in [("BLUE", [35, 45, 210]), ("RED", [210, 45, 35])]:
        class_dir = root / name
        class_dir.mkdir(parents=True, exist_ok=False)
        for sample_index in range(24):
            noise = rng.integers(-20, 21, size=(48, 60, 3))
            pixels = np.clip(np.array(rgb) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(class_dir / f"sample-{sample_index:02d}.png")


def make_transform():
    """Both views use deterministic RGB preprocessing in this baseline."""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # uint8 RGB -> float tensor (3, 64, 64), range [0, 1]
    ])


def prepare_data(root, batch_size, seed):
    """Inspect metadata, split original indices, and then create sample views."""
    metadata = ImageFolder(root=root)
    if len(metadata.classes) < 2:
        raise ValueError("At least two class folders are required.")

    # Counting classes needs stored labels, not image pixels.
    class_counts = {name: 0 for name in metadata.classes}
    for label in metadata.targets:
        class_counts[metadata.classes[label]] += 1
    assert sum(class_counts.values()) == len(metadata)
    print("class_to_idx:", metadata.class_to_idx)
    print("class_counts:", class_counts)
    print("first sample (path, label):", metadata.samples[0])

    # The indices refer to this exact ordered file-and-label list.
    original_indices = list(range(len(metadata)))
    train_idx, val_idx = train_test_split(
        original_indices,
        test_size=0.25,
        stratify=metadata.targets,
        random_state=seed,
    )
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx) | set(val_idx) == set(original_indices)

    # Separate ImageFolder objects let us configure transforms independently.
    train_base = ImageFolder(root=root, transform=make_transform())
    val_base = ImageFolder(root=root, transform=make_transform())
    assert metadata.samples == train_base.samples == val_base.samples
    assert metadata.class_to_idx == train_base.class_to_idx == val_base.class_to_idx
    train_dataset = Subset(train_base, train_idx)
    val_dataset = Subset(val_base, val_idx)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=0, generator=generator,
    )
    # These settings preserve Subset order, which link_results relies on.
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=0,
    )
    print("samples: total / train / validation:",
          len(metadata), len(train_dataset), len(val_dataset))
    print("validation batches:", len(val_loader))
    images, labels = next(iter(val_loader))
    print("batch shapes:", tuple(images.shape), tuple(labels.shape))
    print("batch dtypes:", images.dtype, labels.dtype)
    assert images.shape[0] == labels.shape[0]
    assert images.shape[1:] == (3, 64, 64)
    assert labels.dtype == torch.int64
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, device):
    """Return the sample-weighted mean of ordinary, unweighted CE losses."""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    loss_sum, total = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        # The final small batch must contribute according to its sample count.
        loss_sum += loss.item() * len(labels)
        total += len(labels)
    return loss_sum / total


def evaluate(model, loader, device):
    """Collect paired labels and predictions in the validation Subset's order."""
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("File mapping requires sequential sampling and drop_last=False.")
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            preds = logits.argmax(dim=1)
            loss_sum += loss.item() * len(labels)
            correct += (preds == labels).sum().item()
            total += len(labels)
            # Extend adds individual sample values, not one list per batch.
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    assert total > 0
    assert len(y_true) == len(y_pred) == total == len(loader.dataset)
    stats = {"loss": loss_sum / total, "accuracy": correct / total, "samples": total}
    return stats, y_true, y_pred


def link_results(val_dataset, y_true, y_pred):
    """Result index -> Subset's original index -> file path and class names."""
    val_base = val_dataset.dataset
    assert len(y_true) == len(y_pred) == len(val_dataset)
    rows = []
    for result_index, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        origin_idx = val_dataset.indices[result_index]
        img_path, stored_label = val_base.samples[origin_idx]
        # This checks labels, not identity among different images of one class.
        # Identity is preserved by the loader settings and paired collection above.
        assert true_label == stored_label
        rows.append({
            "result_index": result_index,
            "original_index": origin_idx,
            "file_path": img_path,
            "true_label": true_label,
            "pred_label": pred_label,
            "true_name": val_base.classes[true_label],
            "pred_name": val_base.classes[pred_label],
            "correct": true_label == pred_label,
        })
    return rows


def make_report(y_true, y_pred, class_names):
    """Keep every known class in a fixed order, including absent predictions."""
    class_ids = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    assert cm.sum() == len(y_true) == len(y_pred)
    report = classification_report(
        y_true, y_pred, labels=class_ids, target_names=class_names,
        digits=4, zero_division=0,
    )
    # zero_division is a display policy; absence of data is not measured failure.
    print("confusion matrix (rows=true, columns=predicted):\n", cm)
    print(report)
    return cm, report


def arithmetic_example():
    """Reconstruct the article's illustrative matrix, separately from inference."""
    demo_true = [0] * (222 + 48) + [1] * (12 + 765)
    demo_pred = [0] * 222 + [1] * 48 + [0] * 12 + [1] * 765
    print("\nArithmetic example only: not this model's measured performance")
    cm, _ = make_report(demo_true, demo_pred, ["NORMAL", "PNEUMONIA"])
    assert cm.tolist() == [[222, 48], [12, 765]]
    assert np.isclose(accuracy_score(demo_true, demo_pred), 987 / 1047)


def save_outputs(output_dir, metadata, history, stats, rows, cm, report):
    """Write traceable results. Real source images are never modified."""
    summary = {**metadata, "history": history, "validation": stats,
               "confusion_matrix": cm.tolist()}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "classification-report.txt").write_text(report, encoding="utf-8")
    columns = list(rows[0])
    errors = [row for row in rows if not row["correct"]]
    for filename, records in [("predictions.csv", rows), ("errors.csv", errors)]:
        with (output_dir / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)

    if errors:
        # The title contains both names. This displays the original file, not
        # necessarily the exact transformed tensor that was passed to the model.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        first = errors[0]
        with Image.open(first["file_path"]) as image:
            display_image = image.convert("RGB")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(display_image)
        ax.set_title(f"True: {first['true_name']} | Pred: {first['pred_name']}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "first-error.png", dpi=140)
        plt.close(fig)
        display_image.close()
    print("saved reports:", output_dir)


def run(root, args, output_dir, demo_mode):
    seed_everything(args.seed)
    device = choose_device(args.device)
    print("mode:", "SYNTHETIC DEMO (not X-ray performance)" if demo_mode else "REAL DATA")
    print("device:", device)
    train_loader, val_loader = prepare_data(root, args.batch_size, args.seed)
    val_base = val_loader.dataset.dataset
    model = TinyCNN(num_classes=len(val_base.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        stats, y_true, y_pred = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_loss": stats["loss"], "val_accuracy": stats["accuracy"]})
        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f}, "
              f"val_loss={stats['loss']:.4f}, val_accuracy={stats['accuracy']:.2%}")

    # An epoch count of zero deliberately evaluates random, untrained weights.
    if args.epochs == 0:
        print("No training: evaluating random initial weights.")
        stats, y_true, y_pred = evaluate(model, val_loader, device)
    rows = link_results(val_loader.dataset, y_true, y_pred)
    cm, report = make_report(y_true, y_pred, val_base.classes)
    assert np.isclose(cm.diagonal().sum() / cm.sum(), stats["accuracy"])
    errors = [row for row in rows if not row["correct"]]
    print("misclassified samples:", len(errors))
    if errors:
        print("first error:", json.dumps(errors[0], ensure_ascii=False))
    if output_dir is not None:
        metadata = {"synthetic_demo": demo_mode, "device": str(device),
                    "seed": args.seed, "data_root": str(root),
                    "class_to_idx": val_base.class_to_idx,
                    "preprocessing": "RGB -> Resize(64,64) -> ToTensor; no Normalize",
                    "split": "image-level stratified split, validation fraction 0.25"}
        save_outputs(output_dir, metadata, history, stats, rows, cm, report)
    arithmetic_example()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true", help="use synthetic images (default)")
    source.add_argument("--data-root", type=Path, help="parent of the class folders")
    parser.add_argument("--epochs", type=int, default=1, help="training epochs; 0 evaluates untrained weights")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, help="new directory for reports and optional demo files")
    args = parser.parse_args()
    if args.epochs < 0 or args.batch_size <= 0:
        parser.error("epochs must be nonnegative and batch-size must be positive")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
    root = args.data_root.expanduser().resolve() if args.data_root else None
    if root is not None and not root.is_dir():
        parser.error("data-root must be an existing directory")
    if output_dir is not None:
        # An output folder inside the class parent could be read as another class.
        if root is not None and output_dir.is_relative_to(root):
            parser.error("output-dir must be outside data-root")
        output_dir.mkdir(parents=True, exist_ok=False)

    if root is not None:
        run(root, args, output_dir, demo_mode=False)
    elif output_dir is not None:
        root = output_dir / "demo-data"
        make_demo_images(root, args.seed)
        run(root, args, output_dir, demo_mode=True)
    else:
        with tempfile.TemporaryDirectory(prefix="image-evaluation-demo-") as temporary:
            root = Path(temporary) / "demo-data"
            make_demo_images(root, args.seed)
            run(root, args, None, demo_mode=True)
            print("Temporary demo files are removed after this run.")


if __name__ == "__main__":
    main()
