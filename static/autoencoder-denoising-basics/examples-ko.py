"""오토인코더: 순전파, 한 번의 갱신, AE/DAE 학습.

PyTorch, NumPy, scikit-learn, Matplotlib이 설치된 Python 환경에서 실행한다.
기본 모드는 작은 수치 예제만 실행한다. digits 모드에는 데이터 다운로드나
별도 MNIST 파일이 필요하지 않다.

실행 예:
    python examples-ko.py
    python examples-ko.py --mode ae --epochs 20
    python examples-ko.py --mode dae --epochs 20 --output-dir figures
    python examples-ko.py --pair assets/mnist_day1_pair.npz --output-dir figures

선택적인 pair 기능은 입력·타깃 확인용이며 모델의 예측 결과가 아니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


def tiny_ae(x: torch.Tensor, e: torch.Tensor, d: torch.Tensor):
    """잠재 스칼라, 복원 벡터, MSE 스칼라를 순서대로 반환한다."""
    # 각 인자는 샘플 하나에 관한 값 두 개를 담는다. 배치 입력용 함수가 아니다.
    if x.shape != (2,) or e.shape != (2,) or d.shape != (2,):
        raise ValueError("x, e, d의 shape는 각각 (2,)여야 합니다.")
    # 원소별로 곱한 뒤 합산한다. shape (2,)에서 스칼라 shape ()가 된다.
    z = (e * x).sum()
    # 같은 스칼라 z에 두 디코더 가중치를 각각 곱한다. 결과 shape는 (2,)다.
    reconstruction = d * z
    # 같은 위치의 성분끼리 빼고 제곱한 뒤, 원소 두 개의 평균을 구한다.
    loss = (reconstruction - x).square().mean()
    return z, reconstruction, loss


def run_tiny():
    """본문의 순전파·기울기·한 번의 갱신·Q1/Q2 결과를 확인한다."""
    # float64는 원소를 64비트 실수로 저장한다. 작은 계산을 자세히 비교하는 데 쓴다.
    x_a = torch.tensor([1.0, 2.0], dtype=torch.float64)
    x_b = torch.tensor([2.0, 1.0], dtype=torch.float64)
    # requires_grad=True는 이 가중치에서 이어지는 연산을 자동 미분용으로 추적한다.
    e = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
    d = torch.tensor([0.8, 1.2], dtype=torch.float64, requires_grad=True)
    # detach는 미분 연결을 끊고, clone은 저장 공간을 독립적으로 복사한다.
    e_initial = e.detach().clone()
    d_initial = d.detach().clone()

    z_a0, r_a0, loss_a0 = tiny_ae(x_a, e, d)
    with torch.no_grad():
        # B는 결과 비교용이다. 이번 갱신의 손실에는 B의 손실을 포함하지 않는다.
        z_b0, r_b0, loss_b0 = tiny_ae(x_b, e, d)

    print("갱신 전")
    for name, z, r, loss in [
        ("A", z_a0, r_a0, loss_a0),
        ("B", z_b0, r_b0, loss_b0),
    ]:
        # item()은 원소 하나를 Python 숫자로, tolist()는 값을 Python 리스트로 꺼낸다.
        print(f"{name}: z={z.item():.6f}, reconstruction={r.tolist()}, MSE={loss.item():.10f}")

    errors = (r_a0 - x_a).square()
    # 스칼라는 이미 값 하나다. mean()이 합산 전 원소 개수를 알아내지는 않는다.
    wrong_mean = errors.sum().mean()
    correct_mean = errors.mean()
    print("mean(sum(errors)):", wrong_mean.item())  # 0.08이다. 원하는 MSE가 아니다.
    print("mean(errors):", correct_mean.item())     # 0.04.
    # stack은 새 축으로 텐서들을 묶으며 기존 자동 미분 연결을 유지한다.
    loss_by_stack = torch.stack([errors[0], errors[1]]).mean()
    assert loss_by_stack.requires_grad and loss_by_stack.grad_fn is not None
    torch.testing.assert_close(loss_by_stack, loss_a0, rtol=0, atol=1e-12)
    # stack 대신 torch.tensor([errors[0], errors[1]])를 사용하지 않는다.
    # 해당 생성자는 값을 복사한 새 텐서를 만들며 원래 자동 미분 연결을 끊는다.

    # backward()는 기울기를 계산한다. 이 호출만으로 e와 d가 바뀌지는 않는다.
    loss_a0.backward()
    assert e.grad is not None and d.grad is not None
    expected_e_grad = torch.tensor([-0.08, -0.16], dtype=torch.float64)
    expected_d_grad = torch.tensor([0.3, -0.3], dtype=torch.float64)
    torch.testing.assert_close(e.grad, expected_e_grad, rtol=0, atol=1e-12)
    torch.testing.assert_close(d.grad, expected_d_grad, rtol=0, atol=1e-12)
    torch.testing.assert_close(e.detach(), e_initial, rtol=0, atol=0)
    torch.testing.assert_close(d.detach(), d_initial, rtol=0, atol=0)
    print("e.grad:", e.grad.tolist())
    print("d.grad:", d.grad.tolist())

    # 두 갱신은 모두 동일한 갱신 전 가중치에서 계산한 기울기를 사용한다.
    grad_e = e.grad.clone()
    grad_d = d.grad.clone()
    learning_rate = 0.1
    with torch.no_grad():
        # no_grad는 갱신 연산의 자동 미분 기록을 생략한다. 값 수정을 금지하지 않는다.
        e -= learning_rate * grad_e
        d -= learning_rate * grad_d
        # 갱신된 가중치의 출력은 새 순전파로 다시 계산해야 한다.
        after_a = tiny_ae(x_a, e, d)
        after_b = tiny_ae(x_b, e, d)
    print("A의 손실로 한 번 갱신한 뒤")
    print("e:", e.tolist(), "d:", d.tolist())
    for name, result, expected_loss in [
        ("A", after_a, 0.02285764),
        ("B", after_b, 0.7275415696),
    ]:
        z, r, loss = result
        torch.testing.assert_close(
            loss, torch.tensor(expected_loss, dtype=torch.float64),
            rtol=0, atol=1e-12,
        )
        print(f"{name}: z={z.item():.6f}, reconstruction={r.tolist()}, MSE={loss.item():.10f}")

    # 축 0은 샘플 축(크기 2), 축 1은 특징 축(크기 2)이다.
    x_batch = torch.stack([x_a, x_b])  # (2, 2).
    # dim=1은 각 샘플 안의 특징을 합산한다. keepdim=True는 그 축을 크기 1로 남긴다.
    z_batch = (x_batch * e_initial).sum(dim=1, keepdim=True)
    r_batch = z_batch * d_initial     # (2,1)과 (2,)를 곱하면 각 샘플별로 두 출력이 생겨 (2,2)가 된다.
    assert z_batch.shape == (2, 1) and r_batch.shape == (2, 2)
    print("배치 Z:", z_batch.tolist(), "shape:", tuple(z_batch.shape))
    print("배치 복원:", r_batch.tolist())
    print("배치 MSE:", (r_batch - x_batch).square().mean().item())  # 0.34.

    # Q1/Q2는 별도 가중치 조건이다. 앞의 A 갱신 실험과 섞지 않는다.
    conditions = [
        ("Q1", [0.2, 0.4], [1.0, 1.5], 0.8, [0.8, 1.2], 0.74),
        ("Q2", [0.38, 0.49], [1.096, 1.484], 1.25, [1.37, 1.855], 0.5639625),
    ]
    print("Q1 / Q2: 입력 B는 같고 인코더·디코더 가중치가 모두 다름")
    for name, e_values, d_values, expected_z, expected_r, expected_loss in conditions:
        qe = torch.tensor(e_values, dtype=torch.float64)
        qd = torch.tensor(d_values, dtype=torch.float64)
        z, r, loss = tiny_ae(x_b, qe, qd)
        for actual, expected in [(z, expected_z), (r, expected_r), (loss, expected_loss)]:
            torch.testing.assert_close(
                actual, torch.tensor(expected, dtype=torch.float64),
                rtol=0, atol=1e-12,
            )
        print(f"{name}: z={z.item():.6f}, reconstruction={r.tolist()}, MSE={loss.item():.7f}")
    print("작은 수치 예제의 모든 검산을 통과했습니다.")


class DenseAutoencoder(nn.Module):
    """입력 shape (N,F)에 적용하는 완전연결 인코더·디코더 모델."""

    def __init__(self, in_features: int, hidden_features: int, latent_features: int):
        # Module은 하위 계층과 그 계층의 학습 매개변수를 관리한다.
        super().__init__()
        # 생성자 인자는 특징 개수다. 배치 크기는 이 인자에 포함하지 않는다.
        # Sequential은 층을 순서대로 실행한다. Linear는 기본으로 편향을 포함한다.
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),  # 음수는 0으로 바꾼다. shape는 유지된다.
            nn.Linear(hidden_features, latent_features),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features),
            nn.Sigmoid(),  # 이번 타깃의 범위가 0~1이므로 사용한다.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # model(x)는 실제 배치를 받아 이 메서드를 실행한다. 새 층을 만들지 않는다.
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction


def run_digits(denoise: bool = False, epochs: int = 20, output_dir: Path | None = None):
    """digits 1,437장으로 학습하고 분리된 원본 360장으로 평가한다."""
    from sklearn.datasets import load_digits
    from torch.utils.data import DataLoader, TensorDataset

    if epochs < 1:
        raise ValueError("epochs는 1 이상이어야 합니다.")
    # 실행마다 난수 시드와 모델을 새로 준비한다. 라이브러리 버전에 따라 수치는 달라질 수 있다.
    torch.manual_seed(42)
    # digits는 MNIST와 다른 8×8 데이터다. 원본 픽셀 범위는 0~16이다.
    # flatten(start_dim=1)은 샘플 축 0을 남기고 이미지별 픽셀 축들을 합친다.
    x = torch.tensor(load_digits().images, dtype=torch.float32)
    x = x.flatten(start_dim=1) / 16.0  # (1797,8,8) -> (1797,64), 픽셀 범위 0~1.

    # 잡음 버전을 만들기 전에 원본부터 분리하여 학습·평가 원본의 중복을 막는다.
    indices = torch.randperm(len(x))  # 인덱스 0~1796을 무작위 순서로 나열한다.
    split = int(len(x) * 0.8)         # 1437.
    train_x = x[indices[:split]]      # split 앞까지 선택한다. 끝 인덱스는 포함하지 않는다.
    test_x = x[indices[split:]]       # 나머지 360장이다. 가중치 갱신에 사용하지 않는다.
    assert train_x.shape == (1437, 64) and test_x.shape == (360, 64)
    # TensorDataset은 여기서 원소 하나인 튜플을 반환한다. batch_size는 샘플 수다.
    loader = DataLoader(TensorDataset(train_x), batch_size=64, shuffle=True)

    model = DenseAutoencoder(64, 32, 16)  # (N,64)->(N,32)->(N,16)->(N,32)->(N,64).
    loss_fn = nn.MSELoss()                # 기본값은 출력의 모든 원소에 대한 평균이다.
    # parameters()에는 인코더·디코더의 가중치와 편향이 모두 포함된다.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("실행 모드:", "DAE" if denoise else "AE")
    print("학습 입력 shape:", tuple(train_x.shape), "분리한 평가 입력 shape:", tuple(test_x.shape))

    for epoch in range(epochs):
        model.train()  # Dropout·BatchNorm 등 해당 계층을 학습 동작으로 설정한다.
        total_loss = 0.0
        # 쉼표를 포함한 (clean_batch,)는 원소 하나인 묶음에서 텐서를 꺼낸다.
        for (clean_batch,) in loader:
            if denoise:
                # randn_like는 같은 shape의 평균 0, 표준편차 1인 정규 잡음을 만든다.
                # 0.25는 clamp 적용 전 잡음의 표준편차다. clamp는 값을 0~1로 제한한다.
                input_batch = (
                    clean_batch + 0.25 * torch.randn_like(clean_batch)
                ).clamp(0, 1)
            else:
                input_batch = clean_batch

            # 이전 기울기를 비운다. 새 순전파는 새 계산 그래프를 만든다.
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(input_batch)
            assert reconstruction.shape == clean_batch.shape
            # AE와 DAE 모두 타깃은 clean으로 유지한다.
            loss = loss_fn(reconstruction, clean_batch)
            loss.backward()   # 기울기를 계산하여 .grad에 누적한다.
            optimizer.step()  # 기울기를 이용해 등록된 가중치를 갱신한다.

            # 마지막 배치가 더 작으므로 배치 평균에 해당 배치의 샘플 수를 곱해 더한다.
            # 모든 샘플의 특징 개수가 같은 64개이므로 가능한 집계 방식이다.
            total_loss += loss.item() * len(clean_batch)
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"epoch={epoch + 1}, train_MSE={total_loss / len(train_x):.6f}")

    # 별도 난수 생성기를 사용하여 평가 잡음을 학습 중 난수 소비와 분리한다.
    generator = torch.Generator().manual_seed(123)
    if denoise:
        test_input = (
            test_x + 0.25 * torch.randn(test_x.shape, generator=generator)
        ).clamp(0, 1)
    else:
        test_input = test_x

    # eval()은 해당 계층의 동작을 바꾼다. no_grad는 별도로 자동 미분 기록을 끈다.
    model.eval()
    with torch.no_grad():
        z = model.encoder(test_input)
        reconstruction = model.decoder(z)
        model_mse = loss_fn(reconstruction, test_x).item()
        baseline_mse = loss_fn(test_input, test_x).item()
    assert z.shape == (360, 16) and reconstruction.shape == (360, 64)
    assert torch.isfinite(reconstruction).all()
    print("평가 입력 shape:", tuple(test_input.shape))
    print("평가 잠재 표현 shape:", tuple(z.shape))
    print("평가 복원 shape:", tuple(reconstruction.shape))
    print(f"분리한 평가 데이터의 모델 MSE: {model_mse:.8f}")
    if denoise:
        print(f"무처리 noisy 기준선 MSE: {baseline_mse:.8f}")
        print(f"MSE 개선량: {baseline_mse - model_mse:.8f}")
        print("모델 MSE가 noisy 기준선보다 작음:", model_mse < baseline_mse)
    else:
        # 깨끗한 입력을 그대로 복사하면 MSE는 0이다. 이 복사 동작이 학습된 AE는 아니다.
        print("AE 입력은 clean입니다. 재구성 MSE는 병목을 통과한 모델의 복원을 평가합니다.")

    if output_dir is not None:
        _plot_digits(test_input, reconstruction, test_x, output_dir, denoise)
    # 이 분리 데이터를 보며 모델 설정을 고르면 사실상 검증 데이터로 쓰는 것이다.
    # 최종 성능을 보고할 때에는 별도의 미사용 테스트 데이터를 둔다.
    return model, test_input, reconstruction, test_x


def _plot_digits(inputs, reconstruction, targets, output_dir: Path, denoise: bool):
    """분리한 평가 이미지 6개의 입력·실제 모델 출력·타깃을 저장한다."""
    import matplotlib
    matplotlib.use("Agg")  # 데스크톱 화면 없이 파일로 저장한다.
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    # 그림의 각 행은 같은 원본 샘플이다. 열마다 입력·출력·타깃의 역할이 다르다.
    fig, axes = plt.subplots(6, 3, figsize=(7, 11), squeeze=False)
    columns = [
        ("Noisy input" if denoise else "Clean input", inputs),
        ("Model reconstruction", reconstruction),
        ("Clean target", targets),
    ]
    for col, (title, values) in enumerate(columns):
        for row in range(6):
            ax = axes[row, col]
            # detach/cpu/numpy는 그림에 쓸 데이터를 준비한다. 이 과정에서 학습하지 않는다.
            image = values[row].detach().cpu().reshape(8, 8).numpy()
            ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"held-out {row}")
    fig.suptitle("digits 8x8 — DAE" if denoise else "digits 8x8 — AE")
    fig.tight_layout()
    path = output_dir / ("digits-dae.png" if denoise else "digits-ae.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("저장한 그림:", path)


def plot_pair(pair_path: Path, output_dir: Path):
    """모델 실행 없이 실제 NPZ 파일의 clean/noisy 쌍을 확인한다."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not pair_path.is_file():
        raise FileNotFoundError(f"NPZ 쌍 파일을 찾을 수 없습니다: {pair_path}")
    # np.load는 이름으로 배열을 선택하는 객체를 반환한다. with 종료 시 파일을 닫는다.
    # allow_pickle=False는 pickle 객체 복원을 허용하지 않고 숫자 배열을 읽는다.
    with np.load(pair_path, allow_pickle=False) as pair:
        clean = pair["clean"].copy()
        noisy = pair["noisy"].copy()
    if clean.shape != (1, 28, 28) or noisy.shape != clean.shape:
        raise ValueError("clean과 noisy의 shape는 각각 (1,28,28)이어야 합니다.")
    if not np.isfinite(clean).all() or not np.isfinite(noisy).all():
        raise ValueError("배열에 NaN이나 무한대 값이 있습니다.")
    # 첫 축의 의미는 자료를 만든 규약으로 확인한다. shape만으로 확정하지 않는다.
    print("입력·타깃 쌍의 확인입니다. 모델을 실행한 결과가 아닙니다.")
    for name, values in [("clean", clean), ("noisy", noisy)]:
        print(name, values.shape, values.dtype, values.min(), values.max())
        print(f"{name}[0, 14, 14] = {values[0, 14, 14]}")
    print("출력 숫자는 현재 지정한 파일에서 읽은 실제 값입니다. 다른 자료의 참고 숫자를 가정하지 않습니다.")

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for col, (name, values) in enumerate([("Noisy input", noisy), ("Clean target", clean)]):
        full = axes[0, col]  # 화면의 표시 영역 인덱스다. 이미지 픽셀 인덱스가 아니다.
        full.imshow(values[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest", origin="upper")
        full.set_title(f"{name}: full 28x28")
        # 7×7 영역의 픽셀 중심은 원래 행·열 인덱스 11~17에 있다.
        full.add_patch(Rectangle((10.5, 10.5), 7, 7, fill=False, edgecolor="#ff7043", linewidth=2))
        zoom = axes[1, col]
        patch = values[0, 11:18, 11:18]  # 끝 인덱스 18은 제외한다. 결과 shape는 (7,7)이다.
        zoom.imshow(
            patch, cmap="gray", vmin=0, vmax=1, interpolation="nearest",
            origin="upper", extent=(10.5, 17.5, 17.5, 10.5),
        )
        # extent는 (왼쪽,오른쪽,아래,위) 경계이며 원래 픽셀 중심 좌표를 유지한다.
        zoom.set_xticks(range(11, 18))
        zoom.set_yticks(range(11, 18))
        zoom.set_title(f"{name}: 7x7 crop")
        # (열 14,행 14) 중심 픽셀의 양쪽 경계는 각 방향에서 13.5와 14.5다.
        zoom.add_patch(Rectangle((13.5, 13.5), 1, 1, fill=False, edgecolor="#ff7043", linewidth=2))
        for ax in (full, zoom):
            ax.set_xlabel("column index")
            ax.set_ylabel("row index")
    fig.suptitle("Input/target inspection — not a model prediction")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pair-inspection.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("저장한 그림:", path)


def main():
    parser = argparse.ArgumentParser(description="AE/DAE 순전파·학습 예제.")
    # choices는 모드 이름을 제한한다. 기본 모드는 digits 반복 학습을 하지 않는다.
    parser.add_argument("--mode", choices=["tiny", "ae", "dae"], default="tiny", help="실행할 예제.")
    parser.add_argument("--epochs", type=int, default=20, help="ae/dae의 전체 학습 반복 횟수.")
    parser.add_argument("--pair", type=Path, help="선택적으로 확인할 실제 clean/noisy MNIST NPZ 파일.")
    parser.add_argument("--output-dir", type=Path, help="선택적인 그림 파일을 저장할 폴더.")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs는 1 이상이어야 합니다.")
    if args.pair is not None and not args.pair.is_file():
        parser.error(f"NPZ 쌍 파일을 찾을 수 없습니다: {args.pair}")

    print("PyTorch:", torch.__version__)
    if args.mode == "tiny":
        run_tiny()
    else:
        run_digits(denoise=args.mode == "dae", epochs=args.epochs, output_dir=args.output_dir)
    if args.pair is not None:
        plot_pair(args.pair, args.output_dir or Path("ae-figures"))


# 이 파일을 import하면 함수를 정의한다. 파일을 직접 실행하면 main()을 호출한다.
if __name__ == "__main__":
    main()
