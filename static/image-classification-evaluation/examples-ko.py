#!/usr/bin/env python3
"""이미지 분류: 평가 결과를 원본 샘플까지 연결하는 실행 예제.

필요 패키지: torch, torchvision, scikit-learn, numpy, Pillow, matplotlib.
운영체제에 맞는 호환 가능한 torch / torchvision 조합을 사용합니다.

실행 예시(macOS / Linux):
  python examples-ko.py
  python examples-ko.py --demo --device cpu --output-dir runs/demo-01
  python examples-ko.py --data-root ./chest_xray/train --epochs 3 --device auto

데이터 경로를 지정하지 않으면 작은 합성 RGB 이미지를 임시 폴더에 만듭니다.
이것은 프로그램의 연결 과정 점검이며 실제 X-ray 성능 검증이 아닙니다.
--output-dir을 지정하면 합성 이미지와 보고서를 해당 새 폴더에 보존합니다.
출력 폴더는 아직 존재하지 않아야 합니다. 기존 원본 데이터는 읽기만 합니다.

실제 데이터의 --data-root는 클래스별 폴더의 부모 경로여야 합니다.
이 예제는 이미지 단위로 학습·검증을 나눕니다. 환자 번호를 추론하거나
환자별 독립성을 보장하지 않습니다. 별도로 남긴 테스트 데이터 평가,
최적 모델 저장, 데이터 또는 사전학습 가중치 다운로드는 수행하지 않습니다.

아래 222, 48, 12, 765는 글에 제공된 별도의 지표 계산 예시입니다.
이 예제 모델을 학습해서 얻은 결과가 아닙니다.
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
    """실행 연결을 확인할 작은 모델: 입력 (B, 3, 64, 64), 출력 로짓 (B, K)."""

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
        # CrossEntropyLoss에는 로짓을 전달하므로 여기서 Softmax를 적용하지 않는다.
        return self.classifier(flattened)


def choose_device(requested):
    """auto에서는 CUDA, Apple MPS, CPU 순서로 사용 가능 여부를 확인한다."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("현재 Python 환경에서 CUDA를 사용할 수 없습니다.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("현재 Python 환경에서 MPS를 사용할 수 없습니다.")
    return torch.device(requested)


def seed_everything(seed):
    """난수 초기값을 정한다. 장치가 달라도 결과가 완전히 같음을 보장하지는 않는다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_demo_images(root, seed):
    """픽셀에 작은 무작위 차이를 준 파란 이미지 24장, 빨간 이미지 24장을 만든다."""
    rng = np.random.default_rng(seed)
    for name, rgb in [("BLUE", [35, 45, 210]), ("RED", [210, 45, 35])]:
        class_dir = root / name
        class_dir.mkdir(parents=True, exist_ok=False)
        for sample_index in range(24):
            noise = rng.integers(-20, 21, size=(48, 60, 3))
            pixels = np.clip(np.array(rgb) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(class_dir / f"sample-{sample_index:02d}.png")


def make_transform():
    """이번 기준 예제는 학습·검증 모두 무작위 변환이 없는 RGB 전처리를 사용한다."""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # 8비트 RGB -> 실수 텐서 (3, 64, 64), 값 범위 [0, 1]
    ])


def prepare_data(root, batch_size, seed):
    """경로·라벨을 조사하고 원본 인덱스를 분할한 뒤 각 용도의 Dataset을 만든다."""
    metadata = ImageFolder(root=root)
    if len(metadata.classes) < 2:
        raise ValueError("클래스 폴더가 두 개 이상 필요합니다.")

    # 클래스별 개수는 저장된 라벨만으로 셀 수 있으므로 픽셀을 읽지 않는다.
    class_counts = {name: 0 for name in metadata.classes}
    for label in metadata.targets:
        class_counts[metadata.classes[label]] += 1
    assert sum(class_counts.values()) == len(metadata)
    print("class_to_idx:", metadata.class_to_idx)
    print("class_counts:", class_counts)
    print("first sample (path, label):", metadata.samples[0])

    # 이 인덱스들은 현재 파일·라벨 목록과 그 순서를 기준으로 의미를 가진다.
    original_indices = list(range(len(metadata)))
    train_idx, val_idx = train_test_split(
        original_indices,
        test_size=0.25,
        stratify=metadata.targets,
        random_state=seed,
    )
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx) | set(val_idx) == set(original_indices)

    # 별개의 ImageFolder를 만들어 학습·검증 전처리를 독립적으로 설정한다.
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
    # link_results가 결과와 파일을 연결할 수 있도록 Subset 순서를 유지한다.
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
    """클래스 가중치를 주지 않은 기본 CE 손실을 샘플 수에 비례하여 평균한다."""
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
        # 마지막 작은 배치도 실제 샘플 수만큼만 전체 평균에 반영한다.
        loss_sum += loss.item() * len(labels)
        total += len(labels)
    return loss_sum / total


def evaluate(model, loader, device):
    """검증 Subset 순서대로 정답·예측을 같은 샘플끼리 짝지어 수집한다."""
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("파일 연결에는 순차 샘플링과 drop_last=False가 필요합니다.")
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
            # 배치 목록 자체를 넣지 않고 샘플별 번호들을 하나의 목록으로 이어 붙인다.
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    assert total > 0
    assert len(y_true) == len(y_pred) == total == len(loader.dataset)
    stats = {"loss": loss_sum / total, "accuracy": correct / total, "samples": total}
    return stats, y_true, y_pred


def link_results(val_dataset, y_true, y_pred):
    """결과 인덱스 -> Subset의 원본 인덱스 -> 파일 경로와 클래스 이름으로 연결한다."""
    val_base = val_dataset.dataset
    assert len(y_true) == len(y_pred) == len(val_dataset)
    rows = []
    for result_index, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        origin_idx = val_dataset.indices[result_index]
        img_path, stored_label = val_base.samples[origin_idx]
        # 라벨의 일치를 확인한다. 같은 클래스 안의 다른 파일 구분까지 증명하지는 않는다.
        # 파일 대응은 위 DataLoader 설정과 정답·예측을 함께 모으는 구조로 유지한다.
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
    """예측에 등장하지 않은 클래스도 포함하여 전체 클래스 순서를 고정한다."""
    class_ids = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    assert cm.sum() == len(y_true) == len(y_pred)
    report = classification_report(
        y_true, y_pred, labels=class_ids, target_names=class_names,
        digits=4, zero_division=0,
    )
    # zero_division은 표시 규칙이다. 자료가 없는 경우와 실제 오답은 구분해야 한다.
    print("confusion matrix (rows=true, columns=predicted):\n", cm)
    print(report)
    return cm, report


def arithmetic_example():
    """실제 추론 결과와 별도로 글의 계산 예시 혼동 행렬을 재구성한다."""
    demo_true = [0] * (222 + 48) + [1] * (12 + 765)
    demo_pred = [0] * 222 + [1] * 48 + [0] * 12 + [1] * 765
    print("\n지표 계산 예시: 이 모델의 실제 측정 성능이 아닙니다")
    cm, _ = make_report(demo_true, demo_pred, ["NORMAL", "PNEUMONIA"])
    assert cm.tolist() == [[222, 48], [12, 765]]
    assert np.isclose(accuracy_score(demo_true, demo_pred), 987 / 1047)


def save_outputs(output_dir, metadata, history, stats, rows, cm, report):
    """파일까지 추적할 수 있는 보고서를 저장한다. 실제 원본 이미지는 수정하지 않는다."""
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
        # 실제·예측 이름을 제목 하나에 함께 쓴다. 여기서 표시하는 것은 원본 파일이며,
        # 모델에 전달했던 전처리 텐서와 정확히 같은 이미지라고 단정하지 않는다.
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
    print("mode:", "합성 이미지 점검(실제 X-ray 성능 아님)" if demo_mode else "지정한 실제 데이터")
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

    # epochs=0은 의도적으로 무작위 초기 가중치를 평가하는 점검 방식이다.
    if args.epochs == 0:
        print("학습을 생략하고 무작위 초기 가중치를 평가합니다.")
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
    source.add_argument("--demo", action="store_true", help="합성 이미지로 실행(기본값)")
    source.add_argument("--data-root", type=Path, help="클래스별 폴더의 부모 경로")
    parser.add_argument("--epochs", type=int, default=1, help="학습 횟수; 0이면 미학습 가중치 평가")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, help="보고서와 합성 이미지 등을 저장할 새 폴더")
    args = parser.parse_args()
    if args.epochs < 0 or args.batch_size <= 0:
        parser.error("epochs는 0 이상, batch-size는 1 이상이어야 합니다")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
    root = args.data_root.expanduser().resolve() if args.data_root else None
    if root is not None and not root.is_dir():
        parser.error("data-root는 실제 존재하는 폴더여야 합니다")
    if output_dir is not None:
        # 클래스 부모 아래에 출력 폴더를 만들면 새로운 클래스 폴더로 오인할 수 있다.
        if root is not None and output_dir.is_relative_to(root):
            parser.error("output-dir는 data-root 바깥에 지정해야 합니다")
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
            print("실행을 마치면 임시 합성 이미지 파일을 삭제합니다.")


if __name__ == "__main__":
    main()
