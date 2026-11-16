#!/usr/bin/env python3
"""흉부 X-ray 분류: 데이터 조사 → 분할 → 학습·검증 → 최종 테스트.

필요 패키지: torch, torchvision, scikit-learn, numpy, Pillow.
데이터 폴더에는 train/NORMAL, train/PNEUMONIA, val/NORMAL,
val/PNEUMONIA와 같은 클래스 폴더를 가진 test가 있어야 합니다.

실행 예시(macOS / Linux):
  python examples-ko.py --self-check
  python examples-ko.py --data-dir ./chest_xray --model custom --epochs 1
  python examples-ko.py --data-dir ./chest_xray --model partial --epochs 3
  python examples-ko.py --data-dir ./chest_xray --evaluate-test runs/partial-best.pt

학습 실행은 test 폴더를 읽지 않습니다. 마지막 명령은 이미 선택해 저장한
모델을 불러와 test를 한 번 평가하며, 재학습하거나 다른 모델을 선택하지 않습니다.
기존 체크포인트를 덮어쓰지 않으므로 새 실험은 다른 --output 경로를 사용합니다.

이미지 단위 분할만으로 환자 단위 독립성을 보장할 수 없습니다.
self-check는 임시 RGB 이미지와 무작위 가중치로 프로그램의 동작을 검사합니다.
실제 X-ray 이미지 학습이나 의료 환경의 성능을 검증하는 작업은 아닙니다.
ResNet 학습은 캐시에 없으면 공식 사전 학습 가중치를 다운로드합니다.
체크포인트에 실험 조건을 기록하지만 중단 시점의 정확한 학습 재개는 구현하지
않았습니다. 난수 설정이 같아도 장치·라이브러리 버전에 따라 결과가 달라질 수 있습니다.
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
    """입력은 (B, 3, 224, 224), 출력은 이미지마다 클래스 원시 점수 두 개다."""

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
        return self.fc(features)  # CrossEntropyLoss에는 Softmax가 아닌 로짓을 전달한다.


def seed_everything(seed):
    """모델 초기화의 난수를 설정한다. 분할과 DataLoader에도 별도 설정을 전달한다."""
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
        raise ValueError("현재 Python 환경에서 CUDA를 사용할 수 없습니다.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("현재 Python 환경에서 MPS를 사용할 수 없습니다.")
    return torch.device(requested)


def inspect_and_split(data_dir, seed):
    """경로·라벨 정보만 조사한다. 픽셀을 읽거나 test를 조사하지 않는다."""
    originals = [ImageFolder(data_dir / name) for name in ("train", "val")]
    class_to_idx = originals[0].class_to_idx
    if class_to_idx != originals[1].class_to_idx:
        raise ValueError("기존 train과 val의 클래스 번호 대응이 다릅니다.")
    if set(class_to_idx) != {"NORMAL", "PNEUMONIA"}:
        raise ValueError("NORMAL과 PNEUMONIA 두 클래스 폴더가 필요합니다.")

    # ConcatDataset을 연결할 때와 동일한 순서로 경로와 라벨을 이어 붙인다.
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
    # 무작위 변환이 없는 공식 평가용 전처리다. 이 기준 실험에서는 학습에도 사용한다.
    weights = ResNet18_Weights[weights_name] if weights_name is not None else ResNet18_Weights.DEFAULT
    return weights.transforms()


def make_views(data_dir, originals, train_idx, val_idx, model_kind):
    """새 학습·검증 세트가 서로 다른 Dataset과 전처리 객체를 사용하게 한다."""
    def make_view(indices):
        transform = make_transform(model_kind)
        folders = [
            ImageFolder(data_dir / name, transform=transform)
            for name in ("train", "val")
        ]
        for new, original in zip(folders, originals):
            # 같은 인덱스가 같은 샘플을 가리키려면 파일 순서까지 같아야 한다.
            assert new.samples == original.samples
            assert new.class_to_idx == original.class_to_idx
        return Subset(ConcatDataset(folders), indices)

    train_set = make_view(train_idx)
    val_set = make_view(val_idx)
    assert train_set.dataset is not val_set.dataset
    return train_set, val_set


def build_model(model_kind, seed, pretrained=True):
    """호출할 때마다 새 분류층을 포함한 독립 실험용 모델을 만든다."""
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
    """층의 동작 모드와 가중치 학습 여부를 별도로 설정한다."""
    if model_kind == "custom":
        model.train()
    elif model_kind == "full" and bn_policy == "adapt":
        model.train()
    else:
        # eval()은 저장 통계값을 고정하지만 기울기 기록을 끄지는 않는다.
        model.eval()
        model.fc.train()
        if model_kind == "partial" and bn_policy == "adapt":
            model.layer4.train()
    # Frozen은 입력 옵션과 무관하게 특징 추출층의 통계값을 고정한다.


def check_optimizer(model, optimizer):
    """다른 모델 객체의 파라미터를 가리키는 옵티마이저를 검사한다."""
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual_list = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert expected and expected == set(actual_list)
    assert len(actual_list) == len(set(actual_list))


def train_one_epoch(model, loader, optimizer, device, model_kind, bn_policy):
    set_training_mode(model, model_kind, bn_policy)
    check_optimizer(model, optimizer)
    loss_fn = nn.CrossEntropyLoss()  # 이 예제에서는 클래스 가중치나 제외할 정답을 사용하지 않는다.
    loss_sum, sample_count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("학습 손실에 NaN 또는 무한대가 발생했습니다.")
        loss.backward()
        optimizer.step()
        # 배치 평균을 합계로 되돌려 마지막 작은 배치가 과도하게 반영되지 않게 한다.
        loss_sum += loss.item() * labels.numel()
        sample_count += labels.numel()
    if sample_count == 0:
        raise ValueError("학습 DataLoader가 비어 있습니다.")
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
            raise ValueError("평가 손실에 NaN 또는 무한대가 발생했습니다.")
        loss_sum += loss.item() * labels.numel()
        sample_count += labels.numel()
        # 두 목록에 같은 배치 순서로 추가하여 이미지와 정답의 대응을 유지한다.
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
    if sample_count == 0:
        raise ValueError("평가 DataLoader가 비어 있습니다.")
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
    print("혼동 행렬: 행은 실제 정답, 열은 모델 예측")
    print("클래스 순서:", class_names)
    print(np.asarray(metrics["confusion_matrix"]))


def run_training(args):
    data_dir = args.data_dir.expanduser().resolve()
    output = (args.output or Path("runs") / f"{args.model}-best.pt").expanduser()
    if output.exists():
        raise FileExistsError(f"체크포인트가 이미 있습니다: {output}. 다른 --output 경로를 지정하세요.")
    device = choose_device(args.device)
    originals, samples, targets, train_idx, val_idx, mapping = inspect_and_split(
        data_dir, args.seed
    )
    class_names = class_names_in_order(mapping)
    train_set, val_set = make_views(data_dir, originals, train_idx, val_idx, args.model)
    print("클래스 번호:", mapping)
    for name, indices in (("train", train_idx), ("validation", val_idx)):
        print(name, len(indices), dict(Counter(targets[i] for i in indices)))
    print("연산 장치:", device)
    print("이미지 단위 분할이며 환자 중복 여부는 확인하지 못했습니다.")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    # 셔플용 난수 상태를 소비하지 않도록 Dataset에서 샘플 한 개를 확인한다.
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
    print("학습률:", learning_rate, "BatchNorm 정책:", effective_bn_policy)
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
            # 독립적인 CPU 복사본을 만든다. state_dict()만 보관하면 이후 변경이 함께 반영될 수 있다.
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
    print("선택한 에포크:", best_epoch)
    print_report(best_metrics, class_names)
    print("저장한 체크포인트:", output.resolve())
    print("test는 평가하지 않았습니다. 모델 선택 후 --evaluate-test 체크포인트 경로를 사용하세요.")


def evaluate_saved_test(args):
    """선택한 체크포인트를 불러와 재학습 없이 test를 한 번 평가한다."""
    device = choose_device(args.device)
    checkpoint = torch.load(args.evaluate_test.expanduser(), map_location="cpu", weights_only=True)
    model_kind = checkpoint["model_kind"]
    mapping = checkpoint["class_to_idx"]
    # 체크포인트에 가중치가 있으므로 사전 학습 파일을 다시 다운로드하지 않는다.
    model = build_model(model_kind, checkpoint["seed"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    test_set = ImageFolder(
        args.data_dir.expanduser().resolve() / "test",
        transform=make_transform(model_kind, checkpoint["pretrained_weights"]),
    )
    if test_set.class_to_idx != mapping:
        raise ValueError("test의 클래스 번호가 저장된 학습 클래스 번호와 다릅니다.")
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    class_names = class_names_in_order(mapping)
    metrics = evaluate(model, test_loader, device, class_names)
    print("최종 테스트에 사용한 체크포인트:", args.evaluate_test)
    print_report(metrics, class_names)
    print("테스트 지표 JSON:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def self_check():
    """합성 이미지와 CPU로 데이터 처리·학습 경로의 동작을 확인한다."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(2, previous_threads))
    try:
        with tempfile.TemporaryDirectory(prefix="xray-example-check-") as directory:
            root = Path(directory)
            # 원본 25장을 학습 20장·검증 5장으로 나눠 마지막 검증 배치가 작도록 한다.
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

            # 검사할 때는 무작위 ResNet 가중치를 사용하여 다운로드 없이 실행한다.
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
            print("SELF-CHECK PASSED: 분할, 라벨 대응, 전처리, 배치, 로짓, 기울기,")
            print("가중치 갱신, 옵티마이저 대상, 표본 수를 반영한 평균 손실, 지표, BN 정책을 확인했습니다.")
            print("합성 데이터 동작 검사입니다. 실제 X-ray 학습이나 의료 환경의 성능을 검증한 것은 아닙니다.")
            print("실행 버전:", torch.__version__, torchvision.__version__, sklearn.__version__)
    finally:
        torch.set_num_threads(previous_threads)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, help="train, val, test 폴더가 들어 있는 데이터 경로.")
    parser.add_argument("--model", choices=("custom", "frozen", "partial", "full"), default="custom")
    parser.add_argument("--epochs", type=int, default=1, help="학습 데이터를 전체 순회할 횟수.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, help="기본 시작 학습률 대신 사용할 값.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bn-policy", choices=("fixed", "adapt"), default="fixed",
                        help="BatchNorm 통계값을 고정하거나 학습 구간에서 갱신한다. Frozen은 항상 고정한다.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 작업 프로세스 수. 기본값 0으로 실행 환경 차이를 줄인다.")
    parser.add_argument("--output", type=Path, help="새 체크포인트 경로. 기본값은 runs/MODEL-best.pt.")
    parser.add_argument("--evaluate-test", type=Path, metavar="CHECKPOINT",
                        help="선택한 체크포인트를 재학습 없이 test에서 한 번 평가한다.")
    parser.add_argument("--self-check", action="store_true", help="CPU에서 다운로드 없이 합성 데이터 동작 검사를 실행한다.")
    args = parser.parse_args()
    if not args.self_check and args.data_dir is None:
        parser.error("--self-check 실행이 아니라면 --data-dir이 필요합니다.")
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs와 batch-size는 1 이상, num-workers는 0 이상이어야 합니다.")
    if args.lr is not None and (not np.isfinite(args.lr) or args.lr <= 0):
        parser.error("--lr은 0보다 큰 유한한 숫자여야 합니다.")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed는 0부터 2**32 - 1 사이의 정수여야 합니다.")
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
