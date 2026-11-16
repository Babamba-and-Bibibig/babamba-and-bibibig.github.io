#!/usr/bin/env python3
"""Chest X-ray classification: inspect, split, train, validate, then test.

Required packages: torch, torchvision, scikit-learn, numpy, Pillow.
The data directory must contain train/NORMAL, train/PNEUMONIA,
val/NORMAL, val/PNEUMONIA, and test/ with the same class folders.

Examples (macOS / Linux):
  python examples-en.py --self-check
  python examples-en.py --data-dir ./chest_xray --model custom --epochs 1
  python examples-en.py --data-dir ./chest_xray --model partial --epochs 3
  python examples-en.py --data-dir ./chest_xray --evaluate-test runs/partial-best.pt

Training never reads the test folder. The final command loads an already selected
checkpoint and evaluates test once, without retraining or choosing a new model.
An existing checkpoint is not overwritten; choose another --output path.

The image-level split does not establish patient independence. The self-check
uses generated RGB images and random weights, not clinical images or real scores.
ResNet training downloads official pretrained weights if they are not cached.
The checkpoint records the experiment; exact interrupted-run resume is not
implemented, and a seed does not guarantee equality across devices or versions.
"""

import argparse
import copy
import json
import random
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2


class CustomCNN(nn.Module):
    """Input (B, 3, 224, 224); output two raw class scores per image."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, images):
        features = self.pool(torch.relu(self.conv1(images)))
        features = self.pool(torch.relu(self.conv2(features)))
        features = torch.flatten(features, start_dim=1)
        return self.fc(features)  # CrossEntropyLoss expects logits, not Softmax.


def seed_everything(seed):
    """Set initialization seeds; the split and loader also have their own seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
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


def inspect_and_split(data_dir, seed):
    """Read path/label metadata. Do not decode pixels or inspect test."""
    originals = [ImageFolder(data_dir / name) for name in ("train", "val")]
    class_to_idx = originals[0].class_to_idx
    if class_to_idx != originals[1].class_to_idx:
        raise ValueError("The original train/val class mappings differ.")
    if set(class_to_idx) != {"NORMAL", "PNEUMONIA"}:
        raise ValueError("Expected exactly the NORMAL and PNEUMONIA folders.")

    # These lists use exactly the same concatenation order as ConcatDataset.
    samples = originals[0].samples + originals[1].samples
    all_targets = originals[0].targets + originals[1].targets
    train_idx, val_idx = train_test_split(
        range(len(samples)), test_size=0.2, stratify=all_targets, random_state=seed
    )
    train_idx, val_idx = list(train_idx), list(val_idx)
    assert len(samples) == len(all_targets)
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx) | set(val_idx) == set(range(len(samples)))
    return originals, samples, all_targets, train_idx, val_idx, class_to_idx


def make_transform(model_kind, weights_name=None):
    if model_kind == "custom":
        return v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
    # This is a deterministic evaluation preset, also used for this baseline's training.
    weights = ResNet18_Weights[weights_name] if weights_name is not None else ResNet18_Weights.DEFAULT
    return weights.transforms()


def make_views(data_dir, originals, train_idx, val_idx, model_kind):
    """Give the new train/val split separate datasets and transform objects."""
    def make_view(indices):
        transform = make_transform(model_kind)
        folders = [
            ImageFolder(data_dir / name, transform=transform)
            for name in ("train", "val")
        ]
        for new, original in zip(folders, originals):
            # Equal numeric indices only identify equal samples if file order is equal.
            assert new.samples == original.samples
            assert new.class_to_idx == original.class_to_idx
        return Subset(ConcatDataset(folders), indices)

    train_set = make_view(train_idx)
    val_set = make_view(val_idx)
    assert train_set.dataset is not val_set.dataset
    return train_set, val_set


def build_model(model_kind, seed, pretrained=True):
    """Each call starts an independent experiment, including a fresh two-class head."""
    seed_everything(seed)
    if model_kind == "custom":
        return CustomCNN(2)
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    for name, parameter in model.named_parameters():
        if model_kind == "frozen":
            trainable = name.startswith("fc.")
        elif model_kind == "partial":
            trainable = name.startswith("layer4.") or name.startswith("fc.")
        else:
            trainable = True
        parameter.requires_grad_(trainable)
    return model


def set_training_mode(model, model_kind, bn_policy):
    """Module mode and trainable parameters are independent controls."""
    if model_kind == "custom":
        model.train()
    elif model_kind == "full" and bn_policy == "adapt":
        model.train()
    else:
        # eval() fixes running statistics; it does not disable gradient recording.
        model.eval()
        model.fc.train()
        if model_kind == "partial" and bn_policy == "adapt":
            model.layer4.train()
    # Frozen always keeps its backbone in eval, regardless of the CLI policy.


def check_optimizer(model, optimizer):
    """Reject an optimizer that still points at a different model object."""
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual_list = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert expected and expected == set(actual_list)
    assert len(actual_list) == len(set(actual_list))


def train_one_epoch(model, loader, optimizer, device, model_kind, bn_policy):
    set_training_mode(model, model_kind, bn_policy)
    check_optimizer(model, optimizer)
    loss_fn = nn.CrossEntropyLoss()  # No class weights or ignored targets in this example.
    loss_sum, sample_count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("Training produced a non-finite loss.")
        loss.backward()
        optimizer.step()
        # Undo the batch mean so a small final batch is not over-weighted.
        loss_sum += loss.item() * labels.numel()
        sample_count += labels.numel()
    if sample_count == 0:
        raise ValueError("The training loader is empty.")
    return loss_sum / sample_count


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loss_sum, sample_count = 0.0, 0
    all_labels, all_preds = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("Evaluation produced a non-finite loss.")
        loss_sum += loss.item() * labels.numel()
        sample_count += labels.numel()
        # Both lists receive the same batch order, preserving image-label pairing.
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
    if sample_count == 0:
        raise ValueError("The evaluation loader is empty.")
    class_ids = list(range(len(class_names)))
    return {
        "loss": loss_sum / sample_count,
        "sample_count": sample_count,
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(
            all_labels, all_preds, labels=class_ids, average="macro", zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            all_labels, all_preds, labels=class_ids
        ).tolist(),
        "classification_report": classification_report(
            all_labels, all_preds, labels=class_ids, target_names=class_names,
            output_dict=True, zero_division=0,
        ),
    }


def class_names_in_order(class_to_idx):
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def print_report(metrics, class_names):
    print(f"loss={metrics['loss']:.6f} accuracy={metrics['accuracy']:.4%} "
          f"macro_f1={metrics['macro_f1']:.4f}")
    print("class            precision    recall        F1   support")
    for name in class_names:
        row = metrics["classification_report"][name]
        print(f"{name:16} {row['precision']:.4f}    {row['recall']:.4f}    "
              f"{row['f1-score']:.4f}    {int(row['support'])}")
    print("Confusion matrix: rows=true labels, columns=predicted labels")
    print("Class order:", class_names)
    print(np.asarray(metrics["confusion_matrix"]))


def run_training(args):
    data_dir = args.data_dir.expanduser().resolve()
    output = (args.output or Path("runs") / f"{args.model}-best.pt").expanduser()
    if output.exists():
        raise FileExistsError(f"Checkpoint exists: {output}. Choose another --output path.")
    device = choose_device(args.device)
    originals, samples, targets, train_idx, val_idx, mapping = inspect_and_split(
        data_dir, args.seed
    )
    class_names = class_names_in_order(mapping)
    train_set, val_set = make_views(data_dir, originals, train_idx, val_idx, args.model)
    print("Class mapping:", mapping)
    for name, indices in (("train", train_idx), ("validation", val_idx)):
        print(name, len(indices), dict(Counter(targets[i] for i in indices)))
    print("Device:", device)
    print("Image-level split: patient overlap has not been verified.")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    # Inspect a dataset sample without consuming the shuffled loader's generator.
    image, label = train_set[0]
    assert isinstance(image, torch.Tensor) and image.shape == (3, 224, 224)
    assert image.dtype == torch.float32 and label in mapping.values()
    assert torch.isfinite(image).all()
    if args.model == "custom":
        assert 0.0 <= image.min().item() <= image.max().item() <= 1.0

    model = build_model(args.model, args.seed).to(device)
    default_lr = {"custom": 1e-3, "frozen": 1e-3, "partial": 1e-4, "full": 1e-5}
    learning_rate = args.lr if args.lr is not None else default_lr[args.model]
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )
    check_optimizer(model, optimizer)
    effective_bn_policy = "fixed" if args.model == "frozen" else args.bn_policy
    print("Learning rate:", learning_rate, "BatchNorm policy:", effective_bn_policy)
    best_f1, best_state, best_optimizer, best_metrics, best_epoch = -1.0, None, None, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, args.model, effective_bn_policy
        )
        metrics = evaluate(model, val_loader, device, class_names)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation": metrics})
        print(f"epoch={epoch} train_loss={train_loss:.6f} "
              f"val_loss={metrics['loss']:.6f} val_accuracy={metrics['accuracy']:.4%} "
              f"val_macro_f1={metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            # Make independent CPU copies; state_dict() alone would retain shared state.
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            best_metrics = copy.deepcopy(metrics)

    checkpoint = {
        "model_state": best_state,
        "optimizer_state": best_optimizer,
        "model_kind": args.model,
        "seed": args.seed,
        "selected_epoch": best_epoch,
        "selection_metric": "validation_macro_f1",
        "class_to_idx": mapping,
        "bn_policy": effective_bn_policy,
        "learning_rate": learning_rate,
        "pretrained_weights": None if args.model == "custom" else ResNet18_Weights.DEFAULT.name,
        "transform": repr(make_transform(args.model)),
        "samples": [(str(Path(path).relative_to(data_dir)), label) for path, label in samples],
        "train_indices": train_idx,
        "validation_indices": val_idx,
        "validation": best_metrics,
        "history": history,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "versions": {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                     "scikit-learn": str(sklearn.__version__)},
        "patient_split_verified": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print("Selected epoch:", best_epoch)
    print_report(best_metrics, class_names)
    print("Saved checkpoint:", output.resolve())
    print("Test was not evaluated. Select a model before using --evaluate-test CHECKPOINT.")


def evaluate_saved_test(args):
    """Load a selected checkpoint and evaluate test once; do not retrain."""
    device = choose_device(args.device)
    checkpoint = torch.load(args.evaluate_test.expanduser(), map_location="cpu", weights_only=True)
    model_kind = checkpoint["model_kind"]
    mapping = checkpoint["class_to_idx"]
    # Weights are already in the checkpoint; no pretrained download is needed.
    model = build_model(model_kind, checkpoint["seed"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    test_set = ImageFolder(
        args.data_dir.expanduser().resolve() / "test",
        transform=make_transform(model_kind, checkpoint["pretrained_weights"]),
    )
    if test_set.class_to_idx != mapping:
        raise ValueError("Test class mapping differs from the saved training mapping.")
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    class_names = class_names_in_order(mapping)
    metrics = evaluate(model, test_loader, device, class_names)
    print("Final test of checkpoint:", args.evaluate_test)
    print_report(metrics, class_names)
    print("Test metrics as JSON:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def self_check():
    """Exercise the data and training path using synthetic images and CPU only."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(2, previous_threads))
    try:
        with tempfile.TemporaryDirectory(prefix="xray-example-check-") as directory:
            root = Path(directory)
            # 25 source images -> 20 train + 5 validation, so validation has a short batch.
            for split, counts in (("train", (10, 10)), ("val", (2, 3)), ("test", (1, 1))):
                for class_id, class_name in enumerate(("NORMAL", "PNEUMONIA")):
                    folder = root / split / class_name
                    folder.mkdir(parents=True)
                    for index in range(counts[class_id]):
                        color = (35 + 130 * class_id, 25 + index * 12, 80)
                        Image.new("RGB", (36 + index, 28), color).save(folder / f"sample-{index:02}.png")

            originals, samples, targets, train_idx, val_idx, mapping = inspect_and_split(root, 42)
            assert len(samples) == 25 and len(train_idx) == 20 and len(val_idx) == 5
            assert Counter(targets) == Counter({0: 12, 1: 13})
            assert train_idx == inspect_and_split(root, 42)[3]
            train_set, val_set = make_views(root, originals, train_idx, val_idx, "custom")
            assert train_set[0][1] == samples[train_idx[0]][1]
            assert train_set.dataset.datasets[0].transform is not val_set.dataset.datasets[0].transform
            assert train_set[0][0].shape == (3, 224, 224)
            assert train_set[0][0].dtype == torch.float32
            assert 0 <= train_set[0][0].min() <= train_set[0][0].max() <= 1
            train_loader = DataLoader(Subset(train_set, [0, 1, 2]), batch_size=2)
            val_loader = DataLoader(val_set, batch_size=2)
            assert [len(labels) for _, labels in val_loader] == [2, 2, 1]
            device = torch.device("cpu")
            model = build_model("custom", 42, pretrained=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            check_optimizer(model, optimizer)
            before = model.fc.weight.detach().clone()
            loss = train_one_epoch(model, train_loader, optimizer, device, "custom", "fixed")
            assert np.isfinite(loss) and not torch.equal(before, model.fc.weight)
            images, labels = next(iter(val_loader))
            assert model(images).shape == (2, 2) and labels.dtype == torch.long
            metrics = evaluate(model, val_loader, device, class_names_in_order(mapping))
            with torch.no_grad():
                all_images, all_labels = next(iter(DataLoader(val_set, batch_size=len(val_set))))
                direct_loss = nn.CrossEntropyLoss()(model(all_images), all_labels).item()
            assert abs(metrics["loss"] - direct_loss) < 1e-5
            assert np.asarray(metrics["confusion_matrix"]).sum() == len(val_set)

            # Random ResNet weights avoid downloading pretrained data during the check.
            for kind in ("frozen", "partial", "full"):
                resnet = build_model(kind, 42, pretrained=False)
                for name, parameter in resnet.named_parameters():
                    expected = kind == "full" or name.startswith("fc.") or (
                        kind == "partial" and name.startswith("layer4.")
                    )
                    assert parameter.requires_grad == expected
                optimizer = torch.optim.Adam([p for p in resnet.parameters() if p.requires_grad], lr=1e-3)
                check_optimizer(resnet, optimizer)
                for policy in ("fixed", "adapt"):
                    set_training_mode(resnet, kind, policy)
                    early_bn = resnet.bn1
                    late_bn = resnet.layer4[0].bn1
                    early_before, late_before = early_bn.running_mean.clone(), late_bn.running_mean.clone()
                    head_before = resnet.fc.weight.detach().clone()
                    optimizer.zero_grad(set_to_none=True)
                    scores = resnet(torch.randn(2, 3, 64, 64))
                    nn.CrossEntropyLoss()(scores, torch.tensor([0, 1])).backward()
                    optimizer.step()
                    assert not torch.equal(head_before, resnet.fc.weight)
                    early_adapts = kind == "full" and policy == "adapt"
                    late_adapts = kind in ("partial", "full") and policy == "adapt"
                    assert early_bn.training == early_adapts
                    assert late_bn.training == late_adapts
                    assert torch.equal(early_before, early_bn.running_mean) != early_adapts
                    assert torch.equal(late_before, late_bn.running_mean) != late_adapts
            print("SELF-CHECK PASSED: split, pairing, transforms, batches, logits, gradients,")
            print("parameter updates, optimizer ownership, sample-weighted loss, metrics, and BN policy.")
            print("Synthetic checks only; real X-ray training or clinical performance was not tested.")
            print("Versions:", torch.__version__, torchvision.__version__, sklearn.__version__)
    finally:
        torch.set_num_threads(previous_threads)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, help="Directory containing train, val, and test.")
    parser.add_argument("--model", choices=("custom", "frozen", "partial", "full"), default="custom")
    parser.add_argument("--epochs", type=int, default=1, help="Number of complete training epochs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, help="Override the documented starting learning rate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bn-policy", choices=("fixed", "adapt"), default="fixed",
                        help="Fix BatchNorm statistics or adapt only the trainable region. Frozen always fixes them.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes; zero is a portable baseline.")
    parser.add_argument("--output", type=Path, help="New checkpoint path; default runs/MODEL-best.pt.")
    parser.add_argument("--evaluate-test", type=Path, metavar="CHECKPOINT",
                        help="Evaluate the selected checkpoint on test once, without training.")
    parser.add_argument("--self-check", action="store_true", help="Run offline synthetic checks on CPU.")
    args = parser.parse_args()
    if not args.self_check and args.data_dir is None:
        parser.error("--data-dir is required unless --self-check is used.")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs and batch-size must be positive; num-workers must be non-negative.")
    if args.lr is not None and (not np.isfinite(args.lr) or args.lr <= 0):
        parser.error("--lr must be a positive finite number.")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an integer from 0 through 2**32 - 1.")
    return args


def main():
    args = parse_args()
    if args.self_check:
        self_check()
    elif args.evaluate_test is not None:
        evaluate_saved_test(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
