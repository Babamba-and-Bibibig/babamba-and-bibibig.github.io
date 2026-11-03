"""4채널 잡음 제거 오토인코더: 구현, 한 배치 학습, 두 종류의 비교 실험.

이 파일은 글에서 설명하는 흐름을 실행하도록 새로 작성한 독립 구현이다.
원자료에서 본문이 공개되지 않은 fit/predict_fixed/make_val_noisy의 복제본이 아니다.
필요한 패키지는 torch이며, 그림 저장 기능만 matplotlib을 추가로 사용한다.
CPU의 float32 텐서, (N, 1, 28, 28), 픽셀 범위 0~1을 사용한다.

데이터 없는 구조 검산:
    python examples-ko.py --self-check

실제 데이터를 준비한 뒤 실행:
    torch.save({"clean_train": clean_train, "clean": clean_val}, "data.pt")
    python examples-ko.py --data data.pt --protocol both

clean_train과 clean은 겹치지 않는 훈련/검증 이미지로 준비해야 한다.
키 이름만으로 두 집합의 분리가 증명되지는 않는다.
3 epoch, batch_size=128, lr=0.01은 이 실행 예제의 설정이며 최적값 보장이 아니다.
실행 전에 성능 수치를 미리 넣지 않는다. 원자료에 보고된 결과를 재현했다고 주장하지 않는다.
"""

import argparse
import math
from pathlib import Path

import torch
from torch import nn


def check_images(images, name="images"):
    """이 파일의 CPU 실습에 필요한 이미지 구조와 값 범위를 확인한다."""
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{name}: torch.Tensor required")
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError(f"{name}: expected (N, 1, 28, 28), got {tuple(images.shape)}")
    if images.shape[0] == 0:
        raise ValueError(f"{name}: empty batch")
    if images.device.type != "cpu" or images.dtype != torch.float32:
        raise ValueError(f"{name}: CPU float32 required; use images.detach().cpu().float()")
    if not torch.isfinite(images).all().item():
        raise ValueError(f"{name}: NaN or infinity")
    if images.min().item() < 0 or images.max().item() > 1:
        raise ValueError(f"{name}: pixel values must lie in [0, 1]")


def check_pair(reconstruction, clean):
    """이미지별 오차 계산에서 잘못된 모양과 빈 배치를 거부한다."""
    if reconstruction.shape != clean.shape:
        raise ValueError("reconstruction and clean must have identical shapes")
    if reconstruction.ndim != 4 or any(size == 0 for size in reconstruction.shape):
        raise ValueError("nonempty (N, C, H, W) tensors required")


def check_sigma(sigma):
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")


class StudentDAE(nn.Module):
    """(N,1,28,28) → (N,4,14,14) → (N,1,28,28)."""

    def __init__(self):
        super().__init__()
        # 생성자에는 층의 설정을 전달한다. 이미지를 전달하는 단계가 아니다.
        self.encoder = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1, bias=True)
        self.decoder = nn.ConvTranspose2d(
            4, 1, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.relu = nn.ReLU()  # 음수는 0으로 바꾸며 shape는 유지한다.
        self.sigmoid = nn.Sigmoid()  # 0~1 픽셀 타깃과 비교할 출력을 계산한다.

    def forward(self, noisy):
        # noisy는 실제 입력 데이터이며, 모델을 호출할 때 전달한다.
        z = self.relu(self.encoder(noisy))
        return self.sigmoid(self.decoder(z))


def require_optimizer_matches(model, optimizer):
    """다른 모델에 연결된 옵티마이저를 실수로 재사용하지 않도록 확인한다."""
    model_ids = {id(p) for p in model.parameters() if p.requires_grad}
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if model_ids != optimizer_ids:
        raise ValueError("optimizer must belong to this model's trainable parameters")


def train_one_batch(model, optimizer, noisy, clean):
    """파라미터를 한 번 갱신하고, 갱신 전에 측정한 손실을 반환한다."""
    model.train()  # 학습 모드를 설정한다. 이 호출만으로 학습하지는 않는다.
    optimizer.zero_grad(set_to_none=True)  # 가중치가 아니라 이전 기울기를 지운다.
    pred = model(noisy)
    check_pair(pred, clean)  # 의도하지 않은 브로드캐스팅을 막는다.
    loss = nn.MSELoss()(pred, clean)  # 전체 이미지·채널·픽셀의 제곱오차 평균.
    loss.backward()  # 각 학습 파라미터의 .grad를 계산한다.
    optimizer.step()  # 계산한 기울기로 파라미터를 갱신한다.
    return loss.item()  # 기존 loss는 step 이후 자동으로 다시 계산되지 않는다.


def per_image_mse(reconstruction, clean):
    """반복문으로 이미지마다 오차 하나를 계산한다. 반환 shape는 (N,)이다."""
    check_pair(reconstruction, clean)
    losses = []  # Python 리스트에는 tensor.shape 속성이 없다.
    for recon_img, clean_img in zip(reconstruction, clean):
        loss = (recon_img.flatten() - clean_img.flatten()).square().mean()
        losses.append(loss)  # 각 loss는 축이 없는 스칼라 텐서: shape ().
    return torch.stack(losses)  # 스칼라 N개에 새 축을 붙인다: (N,).


def per_image_mse_fast(reconstruction, clean):
    """같은 계산을 배치 전체에 적용하되 0번 이미지 축은 남긴다."""
    check_pair(reconstruction, clean)
    squared_error = (reconstruction - clean).square()  # (N,C,H,W).
    pixels_per_image = squared_error.flatten(start_dim=1)  # (N,C*H*W).
    return pixels_per_image.mean(dim=1)  # 이미지마다 평균 하나: (N,).


def add_noise(clean, sigma, generator):
    """표준정규 잡음을 배율 sigma로 조절한 뒤 픽셀 범위를 제한한다."""
    check_sigma(sigma)
    epsilon = torch.randn(clean.shape, dtype=clean.dtype, generator=generator)
    # clamp 이후 실제 (noisy-clean)의 분포는 원래 가우시안 분포와 달라진다.
    return (clean + sigma * epsilon).clamp(0.0, 1.0)


def make_val_noisy(clean, sigma, seed=9026):
    """같은 이미지·시드에는 같은 epsilon을 사용해 검증 입력을 다시 만든다."""
    check_images(clean, "clean")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return add_noise(clean, sigma, generator)


def fit(model, optimizer, clean_train, *, sigma, epochs=3, batch_size=128,
        seed=2026, verbose=True):
    """전달받은 모델을 직접 학습한다. history는 배치 크기를 반영한 epoch 평균이다."""
    check_images(clean_train, "clean_train")
    check_sigma(sigma)
    require_optimizer_matches(model, optimizer)
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    # 모델 초기화 난수와 데이터 순서·잡음 난수를 분리해서 관리한다.
    order_rng = torch.Generator(device="cpu").manual_seed(seed)
    noise_rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    history = []
    n_images = len(clean_train)

    for epoch in range(1, epochs + 1):
        order = torch.randperm(n_images, generator=order_rng)
        weighted_loss = 0.0
        for start in range(0, n_images, batch_size):
            clean = clean_train[order[start:start + batch_size]]
            noisy = add_noise(clean, sigma, noise_rng)
            loss_before_update = train_one_batch(model, optimizer, noisy, clean)
            weighted_loss += loss_before_update * len(clean)
        # 마지막 배치가 작더라도 이미지 한 장당 같은 비중으로 집계한다.
        epoch_loss = weighted_loss / n_images
        history.append(epoch_loss)
        if verbose:  # False라도 위의 학습 코드는 모두 실행된다.
            print(f"epoch={epoch}, train_mse={epoch_loss:.6f}")
    # 서로 다른 갱신 시점의 배치 손실 평균이며, 최종 모델의 검증 손실은 아니다.
    return history


def predict_fixed(model, noisy, batch_size=256):
    """전달된 입력을 바꾸지 않고 평가한다. 잡음을 새로 만들거나 학습하지 않는다."""
    check_images(noisy, "noisy")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    was_training = model.training
    model.eval()  # 일부 층의 평가 동작을 설정한다.
    try:
        with torch.no_grad():  # 역전파용 계산 기록을 만들지 않는다.
            parts = [model(batch) for batch in noisy.split(batch_size, dim=0)]
            return torch.cat(parts, dim=0)  # 원래 이미지 순서를 유지한다.
    finally:
        model.train(was_training)  # 호출 전 모드를 복구한다.


def evaluate(model, noisy, clean):
    """같은 clean을 기준으로 입력 오차와 복원 오차를 비교한다."""
    check_images(clean, "clean")
    check_pair(noisy, clean)
    recon = predict_fixed(model, noisy)
    base_each = per_image_mse_fast(noisy, clean)
    rest_each = per_image_mse_fast(recon, clean)
    delta = base_each - rest_each
    base = base_each.mean().item()
    rest = rest_each.mean().item()
    ratio = rest / base if base > 0 else float("nan")
    return {
        "base_mse": base, "rest_mse": rest, "remaining_ratio": ratio,
        "reduction_ratio": 1 - ratio,
        "improved": int((delta > 0).sum().item()),
        "worsened": int((delta < 0).sum().item()),
        "tied": int((delta == 0).sum().item()),
        "count": len(clean), "base_each": base_each, "rest_each": rest_each,
        "recon": recon,
    }


def evaluate_sigmas(model, clean_val, sigmas=(0.25, 0.50), *, val_seed=9026):
    """실험 A: 같은 학습된 모델을 유지하고 평가 입력의 잡음 세기만 바꾼다."""
    rows = []
    for sigma in sigmas:
        noisy = make_val_noisy(clean_val, sigma, seed=val_seed)
        rows.append({"sigma": sigma, **evaluate(model, noisy, clean_val)})
    return rows


def retrain_sigmas(clean_train, clean_val, sigmas=(0.25, 0.50), *, seed=2026,
                   val_seed=9026, epochs=3, batch_size=128, verbose=False):
    """실험 B: 잡음 세기마다 새 모델과 새 옵티마이저로 처음부터 학습한다."""
    rows = []
    for sigma in sigmas:
        torch.manual_seed(seed)  # 같은 환경에서 같은 초기 가중치로 시작한다.
        trial = StudentDAE().cpu()  # 기존 모델이나 학습 결과를 불러오지 않는다.
        trial_optimizer = torch.optim.Adam(trial.parameters(), lr=0.01)
        history = fit(trial, trial_optimizer, clean_train, sigma=sigma,
                      epochs=epochs, batch_size=batch_size, seed=seed, verbose=verbose)
        noisy = make_val_noisy(clean_val, sigma, seed=val_seed)
        rows.append({"sigma": sigma, "model": trial, "history": history,
                     **evaluate(trial, noisy, clean_val)})
    return rows


def save_triplet(clean, noisy, recon, index, out_dir):
    """같은 위치의 원본·잡음·복원을 저장하고 실제 저장 경로 하나를 반환한다."""
    import matplotlib.pyplot as plt  # 그림을 저장할 때만 필요한 패키지.

    check_pair(noisy, clean)
    check_pair(recon, clean)
    if not 0 <= index < len(clean):
        raise IndexError("image index outside the batch")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for ax, images, title in zip(axes, (clean, noisy, recon), ("Clean", "Noisy", "Recon")):
        ax.imshow(images[index, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)  # 세 이미지에 같은 밝기 범위를 적용한다.
        ax.axis("off")
    fig.suptitle(f"Image index {index}")
    fig.tight_layout()
    path = out_dir / f"image_{index}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def print_rows(label, rows):
    print(label)
    for row in rows:
        print(
            f"sigma={row['sigma']:.2f}, input_mse={row['base_mse']:.6f}, "
            f"recon_mse={row['rest_mse']:.6f}, remaining={row['remaining_ratio']:.1%}, "
            f"reduction={row['reduction_ratio']:.1%}, "
            f"improved/worsened/tied={row['improved']}/{row['worsened']}/{row['tied']}"
        )


def self_check():
    """인공 텐서로 구조와 계산만 검산한다. 손글씨 복원 성능 실험이 아니다."""
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = StudentDAE().cpu()
    clean = torch.rand(3, 1, 28, 28)
    noisy = make_val_noisy(clean, 0.25)
    assert tuple(model.encoder.weight.shape) == (4, 1, 3, 3)
    assert tuple(model.decoder.weight.shape) == (4, 1, 4, 4)
    assert sum(p.numel() for p in model.parameters()) == 105
    with torch.no_grad():
        assert tuple(model.encoder(clean).shape) == (3, 4, 14, 14)
        assert tuple(model(clean).shape) == (3, 1, 28, 28)
        assert tuple(model(torch.zeros(2, 1, 27, 27)).shape) == (2, 1, 28, 28)

    each = per_image_mse(noisy, clean)
    assert tuple(each.shape) == (3,)
    assert torch.allclose(each, per_image_mse_fast(noisy, clean))
    assert torch.allclose(each.mean(), nn.MSELoss()(noisy, clean))
    scalars = [torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3)]
    assert scalars[0].shape == torch.Size([])
    assert tuple(torch.stack(scalars).shape) == (3,)
    assert tuple(torch.stack(scalars).unsqueeze(1).shape) == (3, 1)
    a, b = torch.tensor([10, 20, 30]), torch.tensor([40, 50, 60])
    assert torch.stack([a, b], dim=0).tolist() == [[10, 20, 30], [40, 50, 60]]
    assert torch.stack([a, b], dim=1).tolist() == [[10, 40], [20, 50], [30, 60]]
    assert tuple(a.unsqueeze(0).shape) == (1, 3) and tuple(a.unsqueeze(1).shape) == (3, 1)
    assert tuple(a.shape) == (3,)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    require_optimizer_matches(model, optimizer)
    before_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        loss_before = nn.MSELoss()(model(noisy), clean).item()
    returned_loss = train_one_batch(model, optimizer, noisy, clean)
    with torch.no_grad():
        loss_after = nn.MSELoss()(model(noisy), clean).item()
    assert math.isclose(returned_loss, loss_before, rel_tol=1e-7)
    assert not math.isclose(returned_loss, loss_after, rel_tol=1e-7)
    assert any(not torch.equal(before_parameters[name], p) for name, p in model.named_parameters())

    saved_state = {name: p.detach().clone() for name, p in model.state_dict().items()}
    model.train()
    prediction = predict_fixed(model, noisy, batch_size=2)
    assert model.training and not prediction.requires_grad
    assert all(torch.equal(saved_state[name], p) for name, p in model.state_dict().items())
    rows_a = evaluate_sigmas(model, clean)
    assert len(rows_a) == 2
    assert all(torch.equal(saved_state[name], p) for name, p in model.state_dict().items())
    assert torch.equal(noisy, make_val_noisy(clean, 0.25))
    middle = torch.full((2, 1, 28, 28), 0.5)
    eps = torch.randn(middle.shape, generator=torch.Generator().manual_seed(77))
    for sigma in (0.25, 0.50):
        assert torch.equal(make_val_noisy(middle, sigma, 77), (middle + sigma * eps).clamp(0, 1))

    other_model = StudentDAE().cpu()
    try:
        fit(other_model, optimizer, clean, sigma=0.25, epochs=1, verbose=False)
    except ValueError:
        pass
    else:
        raise AssertionError("optimizer for a different model was accepted")
    rows_b = retrain_sigmas(clean, middle, sigmas=(0.25, 0.25), epochs=1, batch_size=2)
    assert rows_b[0]["model"] is not rows_b[1]["model"]
    assert rows_b[0]["model"] is not model
    assert all(torch.equal(a, b) for a, b in zip(rows_b[0]["model"].parameters(), rows_b[1]["model"].parameters()))
    assert len(rows_b[0]["history"]) == 1
    print(f"SELF-CHECK PASS (synthetic tensors only; torch {torch.__version__})")
    print(f"parameters=105, scalar=(), per_image=(3,), unsqueeze=(3,1)")
    print(f"loss_before={loss_before:.8f}, returned={returned_loss:.8f}, loss_after={loss_after:.8f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--protocol", choices=("evaluate", "retrain", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.data is None:
        parser.error("use --self-check, or provide --data data.pt")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    # 텐서 딕셔너리를 읽는다. 원자료의 숨겨진 함수를 불러오는 코드가 아니다.
    data = torch.load(args.data, map_location="cpu", weights_only=True)
    clean_train = data["clean_train"].detach().cpu().float()
    clean_val = data["clean"].detach().cpu().float()
    check_images(clean_train, "clean_train")
    check_images(clean_val, "clean_val")
    seed = 2026

    if args.protocol in ("evaluate", "both"):
        # 독립 실행을 위해 먼저 모델 하나를 sigma=0.25 조건에서 학습한다.
        print("TRAIN ONE MODEL for experiment A (sigma=0.25)")
        torch.manual_seed(seed)
        model = StudentDAE().cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        fit(model, optimizer, clean_train, sigma=0.25, epochs=args.epochs,
            batch_size=args.batch_size, seed=seed)
        print_rows("A: SAME MODEL, CHANGE EVALUATION NOISE ONLY", evaluate_sigmas(model, clean_val))

    if args.protocol in ("retrain", "both"):
        # 이 분기는 두 조건 각각에 대해 학습을 다시 시작한다.
        rows = retrain_sigmas(clean_train, clean_val, seed=seed, epochs=args.epochs,
                             batch_size=args.batch_size, verbose=True)
        print_rows("B: NEW MODEL AND TRAINING FOR EACH NOISE LEVEL", rows)


if __name__ == "__main__":
    main()
