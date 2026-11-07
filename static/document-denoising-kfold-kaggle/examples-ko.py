"""문서 복원 오토인코더: 교차 검증과 최종 학습을 분리한 실행 예제.

필요 패키지: torch, numpy, Pillow, scikit-learn
예: python examples-ko.py --mode check
예: python examples-ko.py --mode cv --data-dir ./data --output-dir ./runs
예: python examples-ko.py --mode final --data-dir ./data --output-dir ./runs

data 폴더에는 train/, train_cleaned/, test/를 직접 준비한다.
학습 손실은 패딩을 포함한 MSE, 검증 지표는 원본 영역만 사용한다.
일반 KFold 예제이므로 같은 원본 문서의 파생 이미지가 있는지 별도로 확인한다.
check는 합성 데이터 검증이며, 실제 대회 점수나 복원 성능의 재현이 아니다.
final은 제출 파일까지만 만들고 Kaggle 서버에는 전송하지 않는다.
"""

import argparse
import csv
import io
import math
import random
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def choose_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("이 환경에서는 CUDA를 사용할 수 없습니다.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("이 환경에서는 MPS를 사용할 수 없습니다.")
    return torch.device(requested)


def seed_all(seed):
    # 같은 환경에서 비교하기 위한 설정이다. 장치 간 완전한 재현을 보장하지 않는다.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_document(path):
    with Image.open(path) as opened:
        image = opened.convert("L")
        if image.size not in ((540, 420), (540, 258)):
            raise ValueError(f"예상하지 못한 크기: {path.name}, {image.size}")
        height = image.height
        array = np.array(image, dtype=np.float32)
    x = torch.from_numpy(array).unsqueeze(0) / 255.0  # (1,H,W), 0~1.
    top = (420 - height) // 2
    return F.pad(x, (0, 0, top, top), value=1.0), height


def load_pairs(data_dir):
    data_dir = Path(data_dir)
    paths = sorted((data_dir / "train").glob("*.png"))
    names = [p.name for p in paths]
    clean_names = {p.name for p in (data_dir / "train_cleaned").glob("*.png")}
    if not paths or set(names) != clean_names:
        raise ValueError("train과 train_cleaned의 PNG 파일명 집합을 확인하세요.")
    inputs, targets, heights = [], [], []
    for path in paths:
        x, height = read_document(path)
        y, clean_height = read_document(data_dir / "train_cleaned" / path.name)
        if height != clean_height:
            raise ValueError(f"입력과 정답의 원본 크기가 다릅니다: {path.name}")
        inputs.append(x)
        targets.append(y)
        heights.append(height)
    # 이미지, 파일명, 원본 높이는 모두 같은 순서다. 크기별로 다시 정렬하지 않는다.
    return torch.stack(inputs), torch.stack(targets), heights, names


def load_test(data_dir):
    paths = sorted((Path(data_dir) / "test").glob("*.png"))
    if not paths:
        raise ValueError("test 폴더에서 PNG 파일을 찾지 못했습니다.")
    images, heights = zip(*(read_document(p) for p in paths))
    return torch.stack(images), list(heights), [p.name for p in paths]


class CustomModel(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, 1, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def image_error_sums(pred, target, original_heights):
    """이미지별 원본 영역의 제곱오차 합과 비교한 원소 수를 반환한다."""
    if pred.ndim != 4 or pred.shape != target.shape or len(pred) != len(original_heights):
        raise ValueError("예측, 정답, 원본 높이의 배치 구성을 확인하세요.")
    sums, counts = [], []
    for i, height in enumerate(original_heights):
        height = int(height)
        gap = pred.shape[-2] - height
        if height <= 0 or gap < 0 or gap % 2:
            raise ValueError("원본 높이와 위아래 대칭 패딩이 일치하지 않습니다.")
        top = gap // 2
        diff = pred[i, :, top:top + height, :] - target[i, :, top:top + height, :]
        sums.append(diff.square().sum())
        counts.append(diff.numel())
    return torch.stack(sums), torch.tensor(counts, device=pred.device)


def per_image_mse(pred, target, original_heights):
    sums, counts = image_error_sums(pred, target, original_heights)
    return sums / counts


def summarize(sse, counts):
    # 집계는 CPU float64로 수행한다. PSNR의 +inf를 임의의 상한으로 바꾸지 않는다.
    sse, counts = sse.detach().cpu().double(), counts.detach().cpu().double()
    if sse.numel() == 0 or sse.shape != counts.shape or not bool((counts > 0).all()):
        raise ValueError("제곱오차 합과 픽셀 수를 확인하세요.")
    if not bool(torch.isfinite(sse).all()) or bool((sse < 0).any()):
        raise ValueError("제곱오차 합에 유효하지 않은 값이 있습니다.")
    mse = sse / counts
    return {
        "mean_image_rmse": mse.sqrt().mean().item(),
        "pooled_pixel_rmse": (sse.sum() / counts.sum()).sqrt().item(),
        "mean_image_psnr_db": (-10 * torch.log10(mse)).mean().item(),
    }


def train_model(x, y, device, *, channels=16, epochs=30, batch_size=24, lr=0.001, seed=42):
    seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                        generator=generator, num_workers=0)
    # 이 함수를 다시 호출하면 모델과 옵티마이저를 모두 새로 만든다.
    model = CustomModel(channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # 학습은 패딩을 포함한다. 검증과 평가 영역이 다르다.
    for _ in range(epochs):
        model.train()
        total = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            if pred.shape != y_batch.shape:
                raise ValueError("모델 출력과 정답 shape가 다릅니다.")
            loss = loss_fn(pred, y_batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("학습 손실이 유한하지 않습니다.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x_batch)
        last_mse = total / len(x)
    return model, last_mse


def save_checkpoint(path, model, config, train_names, val_names):
    # 추론 재사용용이다. 학습 재개에 필요한 optimizer 상태는 포함하지 않는다.
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": config,
        "preprocessing": {"mode": "L", "scale": 255.0, "padded_hw": [420, 540],
                          "allowed_heights": [258, 420], "padding": "symmetric_white"},
        "training_loss": "MSE including padding",
        "evaluation_region": "original pixels only",
        "train_names": train_names, "val_names": val_names,
        "torch_version": str(torch.__version__),
    }, path)


def run_cv(x, y, heights, names, args, device):
    n = len(x)
    model_sse = torch.empty(n, dtype=torch.float64)
    base_sse = torch.empty(n, dtype=torch.float64)
    counts = torch.empty(n, dtype=torch.int64)
    seen, fold_ids = torch.zeros(n, dtype=torch.int64), torch.zeros(n, dtype=torch.int64)
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(range(n)), start=1):
        config = dict(channels=args.channels, epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, seed=args.seed + fold)
        model, train_mse = train_model(x[train_idx], y[train_idx], device, **config)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch_size):
                idx = val_idx[start:start + args.batch_size]
                xb, yb = x[idx].to(device), y[idx].to(device)
                hs = [heights[int(i)] for i in idx]
                ms, pixels = image_error_sums(model(xb), yb, hs)
                bs, _ = image_error_sums(xb, yb, hs)
                model_sse[idx] = ms.cpu().double()
                base_sse[idx] = bs.cpu().double()
                counts[idx] = pixels.cpu()
                seen[idx] += 1
                fold_ids[idx] = fold
        score = summarize(model_sse[val_idx], counts[val_idx])
        print(f"Fold {fold}: train MSE={train_mse:.6f}, original-area metrics={score}")
        save_checkpoint(args.output_dir / f"fold-{fold}.pt", model, config,
                        [names[int(i)] for i in train_idx], [names[int(i)] for i in val_idx])
        del model
    if not bool((seen == 1).all()):
        raise RuntimeError("모든 이미지에 검증 예측이 정확히 하나씩 있어야 합니다.")
    # Fold별 평균을 다시 평균하지 않고, 원래 이미지 순서의 점수를 모아 집계한다.
    report = {"model": summarize(model_sse, counts), "no_op": summarize(base_sse, counts)}
    with (args.output_dir / "cv-summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "model", "no_op"])
        for metric in report["model"]:
            writer.writerow([metric, report["model"][metric], report["no_op"][metric]])
    with (args.output_dir / "oof.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "fold", "original_height", "pixels", "model_sse", "no_op_sse"])
        for i, name in enumerate(names):
            writer.writerow([name, fold_ids[i].item(), heights[i], counts[i].item(),
                             model_sse[i].item(), base_sse[i].item()])
    print("OOF:", report)
    return report


def prediction_batches(model, x, heights, names, batch_size, device):
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = start + batch_size
            batch = x[start:stop].to(device)
            pred = model(batch)
            if pred.shape != batch.shape:
                raise ValueError("추론 출력과 입력의 shape가 다릅니다.")
            # 모든 예측을 GPU에 쌓지 않고 CPU로 옮긴 배치부터 기록한다.
            yield names[start:stop], pred.cpu(), heights[start:stop]


def write_submission(path, batches):
    """(파일명, 예측 배치, 원본 높이)를 받아 CSV를 ZIP에 직접 기록한다."""
    row_count, image_ids = 0, set()
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        with archive.open("submission.csv", "w") as raw:
            with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["id", "value"])
                for names, pred, heights in batches:
                    if pred.ndim != 4 or pred.shape[1] != 1 or len(names) != len(pred) or len(heights) != len(pred):
                        raise ValueError("제출용 배치의 구성을 확인하세요.")
                    for i, name in enumerate(names):
                        stem, height = Path(name).stem, int(heights[i])
                        if not stem.isdigit() or stem in image_ids:
                            raise ValueError(f"이미지 번호가 숫자가 아니거나 중복입니다: {name}")
                        image_ids.add(stem)
                        gap = pred.shape[-2] - height
                        if height <= 0 or gap < 0 or gap % 2:
                            raise ValueError("원본 높이와 패딩이 일치하지 않습니다.")
                        image = pred[i, 0, gap // 2:gap // 2 + height].detach().cpu().numpy()
                        if not np.isfinite(image).all() or not ((0 <= image) & (image <= 1)).all():
                            raise ValueError("예측값은 유한한 0~1 값이어야 합니다.")
                        for row, pixels in enumerate(image, start=1):
                            for col, value in enumerate(pixels, start=1):
                                writer.writerow([f"{stem}_{row}_{col}", float(value)])
                                row_count += 1
    return row_count


def run_final(x, y, names, args, device):
    test_x, test_heights, test_names = load_test(args.data_dir)
    config = dict(channels=args.channels, epochs=args.epochs, batch_size=args.batch_size,
                  lr=args.lr, seed=args.seed)
    # 마지막 Fold 모델을 재사용하지 않고 전체 데이터로 새 모델을 학습한다.
    model, train_mse = train_model(x, y, device, **config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output_dir / "final.pt", model, config, names, [])
    batches = prediction_batches(model, test_x, test_heights, test_names, args.batch_size, device)
    path = args.output_dir / "submission.csv.zip"
    rows = write_submission(path, batches)
    expected = sum(test_heights) * test_x.shape[-1]
    if rows != expected:
        raise RuntimeError("제출 행 수가 원본 이미지의 픽셀 수와 다릅니다.")
    print(f"Final train MSE={train_mse:.6f}; {path}: {rows} pixels; not submitted.")


def self_check():
    torch.set_num_threads(2)
    seed_all(42)
    model = CustomModel()
    with torch.no_grad():
        output = model(torch.zeros(2, 1, 420, 540))
    assert output.shape == (2, 1, 420, 540)
    assert sum(p.numel() for p in model.parameters()) == 417
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.encoder[0].weight.detach().clone()
    optimizer.zero_grad()
    nn.MSELoss()(model(torch.rand(3, 1, 12, 16)), torch.ones(3, 1, 12, 16)).backward()
    optimizer.step()
    assert not torch.equal(before, model.encoder[0].weight)
    pred, target = torch.ones(2, 1, 420, 540), torch.ones(2, 1, 420, 540)
    pred[1, :, :81] = pred[1, :, 339:] = 0
    assert torch.equal(per_image_mse(pred, target, [420, 258]), torch.zeros(2))
    pred[0, 0, 0, 0] = pred[1, 0, 81, 0] = 0
    sums, counts = image_error_sums(pred, target, [420, 258])
    assert sums.tolist() == [1.0, 1.0] and counts.tolist() == [226800, 139320]
    unequal = summarize(torch.tensor([4.0, 1.0]), torch.tensor([4, 16]))
    assert unequal["mean_image_rmse"] == 0.625 and unequal["pooled_pixel_rmse"] == 0.5
    assert math.isinf(summarize(torch.zeros(2), torch.tensor([3, 7]))["mean_image_psnr_db"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "submission.csv.zip"
        small = torch.arange(24, dtype=torch.float32).reshape(2, 1, 4, 3) / 24
        assert write_submission(path, [(["12.png", "3.png"], small, [4, 2])]) == 18
        with ZipFile(path) as archive:
            records = list(csv.DictReader(io.StringIO(archive.read("submission.csv").decode())))
        assert records[0]["id"] == "12_1_1" and records[-1]["id"] == "3_2_3"
        assert float(records[12]["value"]) == float(small[1, 0, 1, 0])
    print(f"PASS: torch={torch.__version__}; shape, update, crop, metrics, CSV (synthetic only).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "cv", "final"), default="check")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.channels) < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, channels, lr는 양수여야 합니다.")
    if args.mode == "check":
        self_check()
        return
    device = choose_device(args.device)
    x, y, heights, names = load_pairs(args.data_dir)
    print(f"device={device}; images={len(x)}; shape={tuple(x.shape)}")
    if args.mode == "cv":
        if not 2 <= args.folds <= len(x):
            parser.error("folds는 2 이상이고 이미지 수 이하여야 합니다.")
        run_cv(x, y, heights, names, args, device)
    else:
        run_final(x, y, names, args, device)


if __name__ == "__main__":
    main()
