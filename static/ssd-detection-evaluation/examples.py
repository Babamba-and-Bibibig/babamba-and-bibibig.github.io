"""SSD evaluation and plotting helpers for the accompanying learning article.

Python 3.10+; --demo uses only the Python standard library.
The other functions require torch, torchvision, torchmetrics, pycocotools,
Pillow and matplotlib. Supply your own trained detector and validation loader;
this file does not invent a PetDataset or download/train a model.

macOS zsh / Linux bash or zsh:
    python examples.py --demo
    python examples.py --verify-metric

Notebook usage (after creating model, device, val_loader and class_map):
    from examples import evaluate_detector, print_evaluation
    results, map_by_iou = evaluate_detector(
        model, device, val_loader, class_map,
        iou_thresholds=[0.50, 0.75, 0.80, 0.85, 0.90, 0.95],
    )
    print_evaluation(results, map_by_iou, class_map)

The evaluator leaves the model in eval mode. Call model.train() before the
next training phase. Display thresholds affect drawings only.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


DEFAULT_IOU_THRESHOLDS = [
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95,
]


def coco101_ap_from_flags(is_true_positive, num_ground_truth):
    """Calculate a teaching example after matching and score sorting.

    is_true_positive: 신뢰도 내림차순으로 정렬한 예측의 TP 여부 목록.
    num_ground_truth: 같은 클래스에 속하는 전체 정답 객체 수.
    실제 박스 대응, crowd/ignore 처리, 동점 정렬은 이 함수의 범위 밖이다.
    실제 객체 탐지 평가는 아래 evaluate_detector를 사용한다.
    """
    if not isinstance(num_ground_truth, int) or num_ground_truth <= 0:
        raise ValueError("num_ground_truth must be a positive integer")
    flags = list(is_true_positive)
    if any(type(flag) is not bool for flag in flags):
        raise TypeError("Each prediction flag must be True or False")
    if sum(flags) > num_ground_truth:
        raise ValueError("One ground-truth object cannot contribute multiple TPs")

    points = []
    true_positives = 0
    for count, is_tp in enumerate(flags, start=1):
        true_positives += int(is_tp)
        precision = true_positives / count
        recall = true_positives / num_ground_truth
        points.append((precision, recall))

    # 각 재현율 기준 이상을 달성한 결과 중 가장 높은 정밀도를 선택한다.
    sampled_precisions = []
    for recall_index in range(101):
        recall_threshold = recall_index / 100
        candidates = [p for p, r in points if r >= recall_threshold]
        sampled_precisions.append(max(candidates, default=0.0))

    return sum(sampled_precisions) / 101


def _class_names(class_map):
    """Invert an SSD foreground name-to-number mapping without guessing."""
    if not class_map:
        raise ValueError("class_map cannot be empty")
    if any(not isinstance(name, str) for name in class_map):
        raise TypeError("class_map keys must be class names")
    ids = list(class_map.values())
    if any(type(class_id) is not int or class_id <= 0 for class_id in ids):
        raise ValueError("SSD foreground class IDs must be positive integers")
    if len(ids) != len(set(ids)):
        raise ValueError("Every class must have a distinct class ID")
    return {class_id: name for name, class_id in class_map.items()}


def _check_image(image):
    import torch

    if not isinstance(image, torch.Tensor):
        raise TypeError("The detector expects tensors, not PIL images or paths")
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) <= 0:
        raise ValueError("Each SSD input image must have shape (3, H, W)")
    if not image.is_floating_point():
        raise TypeError("Use a floating-point image tensor in the 0-1 range")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("Image values must be finite")
    if float(image.min()) < 0 or float(image.max()) > 1:
        raise ValueError("These helpers expect 0-1 images before SSD normalization")


def _cpu_detection(data, class_names, *, prediction):
    """Validate aligned tensors; preserve optional target area and iscrowd."""
    import torch

    required = ("boxes", "labels", "scores") if prediction else ("boxes", "labels")
    for key in required:
        if key not in data or not isinstance(data[key], torch.Tensor):
            raise TypeError(f"{key} must be a tensor")
    keys = list(required)
    if not prediction:
        keys.extend(key for key in ("area", "iscrowd") if key in data)
    result = {}
    for key in keys:
        if not isinstance(data[key], torch.Tensor):
            raise TypeError(f"{key} must be a tensor")
        result[key] = data[key].detach().cpu()

    boxes, labels = result["boxes"], result["labels"]
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not boxes.is_floating_point():
        raise ValueError("boxes must be a floating-point tensor of shape (N, 4)")
    count = len(boxes)
    if labels.ndim != 1 or len(labels) != count or labels.dtype != torch.int64:
        raise ValueError("labels must have shape (N,) and dtype torch.int64")
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("Box coordinates must be finite")
    if count and not bool((boxes[:, 2:] > boxes[:, :2]).all()):
        raise ValueError("Every box must satisfy x2 > x1 and y2 > y1")
    unknown = set(labels.tolist()) - set(class_names)
    if unknown:
        raise ValueError(f"Class IDs missing from class_map: {sorted(unknown)}")

    for key in keys:
        if key in ("boxes", "labels"):
            continue
        value = result[key]
        if value.ndim != 1 or len(value) != count:
            raise ValueError(f"{key} must have shape (N,)")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} values must be finite")
    if prediction:
        scores = result["scores"]
        if not scores.is_floating_point():
            raise TypeError("scores must be floating-point values")
        if count and (float(scores.min()) < 0 or float(scores.max()) > 1):
            raise ValueError("SSD scores must be in the 0-1 range")
    if "area" in result and bool((result["area"] < 0).any()):
        raise ValueError("Target area cannot be negative")
    if "iscrowd" in result and not bool(
        ((result["iscrowd"] == 0) | (result["iscrowd"] == 1)).all()
    ):
        raise ValueError("iscrowd values must be 0 or 1")
    return result


def _map_by_iou(results, iou_thresholds):
    precision = results["precision"]
    if precision.ndim != 5 or precision.shape[0] != len(iou_thresholds):
        raise ValueError("Unexpected precision axes; expected (T, R, K, A, M)")
    scores = {}
    for index, threshold in enumerate(iou_thresholds):
        # IoU 하나, 재현율 전체, 클래스 전체, 크기 all, 탐지 상한 100.
        values = precision[index, :, :, 0, -1]
        valid = values[values >= 0]  # -1만 제외하고 유효한 0점은 유지한다.
        scores[threshold] = valid.mean().item() if valid.numel() else -1.0
    return scores


def evaluate_detector(model, device, val_loader, class_map, iou_thresholds=None):
    """Return (TorchMetrics results, dict[IoU, mAP]) for the whole loader.

    model은 이미 device에 올라간 학습 모델이어야 한다.
    val_loader는 (이미지 텐서 목록, 같은 순서의 정답 딕셔너리 목록)을 낸다.
    이미지 개수와 박스 개수는 다르며, 배치마다 mAP를 따로 평균하지 않는다.
    모델 자체의 후처리 뒤 반환된 예측을 추가 신뢰도 필터 없이 평가한다.
    """
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    names = _class_names(class_map)
    thresholds = list(DEFAULT_IOU_THRESHOLDS if iou_thresholds is None else iou_thresholds)
    if not thresholds or any(not math.isfinite(t) or not 0 <= t <= 1 for t in thresholds):
        raise ValueError("IoU thresholds must be finite values between 0 and 1")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Repeated IoU thresholds would make the mean ambiguous")

    # 매 호출마다 새 평가 객체를 만든다. 이전 검증 결과가 섞이지 않는다.
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=thresholds,
        max_detection_thresholds=[1, 10, 100],
        class_metrics=True,
        extended_summary=True,
        backend="pycocotools",
    )
    model.eval()
    image_count = 0
    with torch.no_grad():
        for images, targets in val_loader:
            if not images or len(images) != len(targets):
                raise ValueError("Each nonempty image batch needs one target per image")
            for image in images:
                _check_image(image)
            predictions = model([image.to(device) for image in images])
            if len(predictions) != len(images):
                raise ValueError("The detector must return one prediction dict per image")

            predictions_cpu = [
                _cpu_detection(pred, names, prediction=True) for pred in predictions
            ]
            targets_cpu = [
                _cpu_detection(target, names, prediction=False) for target in targets
            ]
            metric.update(predictions_cpu, targets_cpu)
            image_count += len(images)

    if image_count == 0:
        raise ValueError("The validation loader did not yield any images")
    results = metric.compute()
    return results, _map_by_iou(results, thresholds)


def print_evaluation(results, map_by_iou, class_map):
    """Print scalar and per-class results without treating -1 as a valid score."""
    names = _class_names(class_map)
    print("IoU thresholds:", list(map_by_iou))
    print(f"mAP (mean over these IoUs): {results['map'].item():.4f}")
    for threshold, score in map_by_iou.items():
        text = f"{score:.4f}" if score >= 0 else "undefined"
        print(f"mAP@{threshold:.2f}: {text}")
    ids = results["classes"].reshape(-1).tolist()
    aps = results["map_per_class"].reshape(-1).tolist()
    ars = results["mar_100_per_class"].reshape(-1).tolist()
    if not len(ids) == len(aps) == len(ars):
        raise ValueError("Class IDs and per-class metrics must have equal lengths")
    for class_id, ap, ar in zip(ids, aps, ars):
        ap_text = f"{ap:.4f}" if ap >= 0 else "undefined"
        ar_text = f"{ar:.4f}" if ar >= 0 else "undefined"
        print(f"{names[class_id]}: AP={ap_text}, AR@100={ar_text}")


def draw_predictions(ax, pred, class_names, display_threshold=0.4):
    """Draw retained predictions; return their count without modifying pred."""
    from matplotlib.patches import Rectangle

    if not math.isfinite(display_threshold) or not 0 <= display_threshold <= 1:
        raise ValueError("display_threshold must be between 0 and 1")
    checked = _cpu_detection(pred, class_names, prediction=True)
    count = 0
    for box, label, score in zip(checked["boxes"], checked["labels"], checked["scores"]):
        score_value = score.item()
        # 기준 미만만 건너뛴다. 평가 입력을 변경하는 필터가 아니다.
        if score_value < display_threshold:
            continue
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="red", linewidth=2,
        ))
        ax.text(x1, y1, f"{class_names[label.item()]} {score_value:.2f}", color="red")
        count += 1
    return count


def show_validation_image(model, device, val_dataset, class_map, index=0, display_threshold=0.4):
    """Return (figure, prediction, shown_count) for one matched image/target."""
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    names = _class_names(class_map)
    image, target = val_dataset[index]
    _check_image(image)
    checked_target = _cpu_detection(target, names, prediction=False)
    model.eval()
    with torch.no_grad():
        pred = model([image.to(device)])[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image.detach().cpu().permute(1, 2, 0))
    for box, label in zip(checked_target["boxes"], checked_target["labels"]):
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="lime", linewidth=2,
        ))
        ax.text(x1, y1, f"GT: {names[label.item()]}", color="lime")
    shown = draw_predictions(ax, pred, names, display_threshold)
    ax.set_title(f"Returned: {len(pred['boxes'])}; shown: {shown}")
    ax.axis("off")
    fig.tight_layout()
    return fig, pred, shown


def show_test_images(model, device, image_dir, test_names, class_map, display_threshold=0.4, seed=42):
    """Return (figure, sampled_names) for at most six unlabeled test JPGs.

    test_names는 확장자를 제외한 파일 이름 목록이다.
    호출 전에 실제 JPG 존재 여부와 train/validation 목록과의 분리를 확인한다.
    정답 박스가 없으므로 이 함수는 mAP를 계산하지 않는다.
    """
    import torch
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    names = _class_names(class_map)
    available_names = sorted(set(test_names))
    if not available_names:
        raise ValueError("test_names cannot be empty")
    sampled_names = random.Random(seed).sample(available_names, min(6, len(available_names)))
    image_dir = Path(image_dir)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax in axes.flat:
        ax.axis("off")

    model.eval()
    with torch.no_grad():
        for ax, name in zip(axes.flat, sampled_names):
            # 이미지 읽기부터 해당 이미지의 예측 표시까지 같은 반복에서 처리한다.
            with Image.open(image_dir / f"{name}.jpg") as opened:
                pil_image = opened.convert("RGB")
            image = to_tensor(pil_image)
            pred = model([image.to(device)])[0]
            ax.imshow(pil_image)
            shown = draw_predictions(ax, pred, names, display_threshold)
            ax.set_title(f"{name}\nshown: {shown}")
    fig.tight_layout()
    return fig, sampled_names


def run_demo():
    """Verify the article's AP arithmetic without third-party packages."""
    sequence_a = coco101_ap_from_flags([True, False, True], 2)
    sequence_b = coco101_ap_from_flags([False, True, True], 2)
    batch_mean = (
        coco101_ap_from_flags([True], 1)
        + coco101_ap_from_flags([False, True], 1)
    ) / 2
    trailing_fp = coco101_ap_from_flags([True, False], 1)
    assert math.isclose(sequence_a, (51 + 50 * 2 / 3) / 101, abs_tol=1e-12)
    assert math.isclose(sequence_b, 2 / 3, abs_tol=1e-12)
    assert math.isclose(batch_mean, 0.75, abs_tol=1e-12)
    assert math.isclose(trailing_fp, 1.0, abs_tol=1e-12)
    print(f"TP, FP, TP:       AP = {sequence_a:.7f}")
    print(f"FP, TP, TP:       AP = {sequence_b:.7f}")
    print(f"Mean batch AP:         {batch_mean:.7f}")
    print(f"Global AP:             {sequence_a:.7f}")
    print(f"TP, FP (one GT):  AP = {trailing_fp:.7f}")
    print("Arithmetic checks passed. No model was trained or evaluated.")


def verify_metric():
    """Cross-check the arithmetic and accumulation with synthetic boxes."""
    import importlib.metadata
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    class_map = {"dog": 1}
    targets = [
        {"boxes": torch.tensor([[0., 0., 10., 10.]]),
         "labels": torch.tensor([1], dtype=torch.int64),
         "area": torch.tensor([100.]), "iscrowd": torch.tensor([0])}
        for _ in range(2)
    ]
    predictions = [
        {"boxes": torch.tensor([[0., 0., 10., 10.]]),
         "labels": torch.tensor([1], dtype=torch.int64), "scores": torch.tensor([.9])},
        {"boxes": torch.tensor([[20., 20., 30., 30.], [0., 0., 10., 10.]]),
         "labels": torch.tensor([1, 1], dtype=torch.int64), "scores": torch.tensor([.8, .7])},
    ]

    class FixedPredictions(torch.nn.Module):
        """A deterministic fixture, not a trained SSD model."""
        def forward(self, images):
            return [
                {key: value.to(image.device) for key, value in predictions[int(image[0, 0, 0].item())].items()}
                for image in images
            ]

    images = [torch.full((3, 40, 40), float(index)) for index in range(2)]
    loader = [([image], [target]) for image, target in zip(images, targets)]
    results, by_iou = evaluate_detector(FixedPredictions(), "cpu", loader, class_map)
    expected = coco101_ap_from_flags([True, False, True], 2)
    assert math.isclose(results["map"].item(), expected, abs_tol=1e-6)
    assert tuple(results["precision"].shape) == (10, 101, 1, 4, 3)
    assert all(math.isclose(value, expected, abs_tol=1e-6) for value in by_iou.values())
    assert math.isclose(by_iou[.5], results["map_50"].item(), abs_tol=1e-6)
    assert math.isclose(by_iou[.75], results["map_75"].item(), abs_tol=1e-6)

    batch_scores = []
    for pred, target in zip(predictions, targets):
        metric = MeanAveragePrecision(iou_thresholds=[.5], backend="pycocotools")
        metric.update([pred], [target])
        batch_scores.append(metric.compute()["map"].item())
    assert math.isclose(sum(batch_scores) / 2, .75, abs_tol=1e-6)

    # FP를 맨 앞 순위로 바꾸면 마지막 TP/FP 개수는 같아도 AP가 달라진다.
    reordered = [{key: value.clone() for key, value in pred.items()} for pred in predictions]
    reordered[0]["scores"] = torch.tensor([.8])
    reordered[1]["scores"] = torch.tensor([.9, .7])
    metric = MeanAveragePrecision(iou_thresholds=[.5], backend="pycocotools")
    metric.update(reordered, targets)
    assert math.isclose(metric.compute()["map"].item(), 2 / 3, abs_tol=1e-6)

    # TP 뒤에 FP가 있어도 보간 AP가 1이 되는 예제를 실제 구현과 대조한다.
    trailing = {key: value.clone() for key, value in predictions[1].items()}
    trailing["scores"] = torch.tensor([.8, .9])
    metric = MeanAveragePrecision(iou_thresholds=[.5], backend="pycocotools")
    metric.update([trailing], [targets[1]])
    assert math.isclose(metric.compute()["map"].item(), 1.0, abs_tol=1e-6)

    print_evaluation(results, by_iou, class_map)
    print("Synthetic boxes: global AP, batch AP, ordering, AP=1 and tensor axes passed.")
    for package in ("torch", "torchvision", "torchmetrics", "pycocotools"):
        print(f"{package}=={importlib.metadata.version(package)}")
    print("This verifies the evaluation API, not a trained pet detector or GPU execution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run standard-library AP arithmetic")
    parser.add_argument("--verify-metric", action="store_true", help="Compare synthetic boxes with TorchMetrics")
    options = parser.parse_args()
    if not options.demo and not options.verify_metric:
        parser.print_help()
    if options.demo:
        run_demo()
    if options.verify_metric:
        verify_metric()
