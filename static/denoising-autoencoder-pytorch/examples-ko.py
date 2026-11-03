"""잡음 제거 오토인코더: 이미 준비한 실습 데이터를 사용하는 통합 코드.

필요한 외부 패키지: torch, matplotlib.
이 파일에는 원본 이미지 데이터나 수업의 d2_tools.py가 들어 있지 않다.
숫자로 된 복원 성능을 미리 넣지 않고, 전달받은 실제 데이터로 계산한다.

노트북에서 이 파일의 함수들을 불러온 뒤 다음과 같이 실행한다.
    result = run_experiment(data, seed=SEED)
    plot_history(result["history"])
    plot_triplets(result["clean_val"], result["noisy_val"], result["recon"])

run_experiment를 호출할 때마다 모델과 옵티마이저를 새로 만든다.
이 파일을 읽거나 함수만 정의하는 단계에서는 학습하지 않는다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


def check_images(images, name):
    """CPU 실습에 사용할 (N, 1, 28, 28) 이미지의 구조와 값 범위를 검사한다."""
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{name}: torch.Tensor가 필요합니다.")
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError(
            f"{name}: 기대 shape는 (N, 1, 28, 28), "
            f"실제 shape는 {tuple(images.shape)}"
        )
    if images.shape[0] == 0:
        raise ValueError(f"{name}: 이미지가 없습니다.")
    if not torch.isfinite(images).all().item():
        raise ValueError(f"{name}: NaN 또는 무한대 값이 있습니다.")
    if images.min().item() < 0 or images.max().item() > 1:
        raise ValueError(f"{name}: 픽셀값을 0~1 범위로 준비해야 합니다.")


def require_same_shape(prediction, target):
    """MSE 계산에서 의도하지 않은 브로드캐스팅을 막는다."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"shape 불일치: 예측 {tuple(prediction.shape)}, "
            f"정답 {tuple(target.shape)}"
        )


class LabDenoiser(nn.Module):
    """흑백 이미지 한 채널을 중간 특징 여러 채널로 바꾸고 다시 복원한다."""

    def __init__(self, feature_channels=8):
        super().__init__()
        if not isinstance(feature_channels, int) or feature_channels <= 0:
            raise ValueError("feature_channels는 양의 정수여야 합니다.")

        # 생성자에는 이미지가 아니라 층의 설정값을 전달한다.
        self.encoder = nn.Sequential(
            # 입력 채널 1, 출력 채널 feature_channels, 커널 3×3.
            # stride=2: 커널 이동 간격 2. padding=1: 입력 가장자리 1칸.
            nn.Conv2d(1, feature_channels, 3, stride=2, padding=1),
            nn.ReLU(),  # 음수를 0으로 바꾼다. shape와 파라미터 수는 늘지 않는다.
        )
        self.decoder = nn.Sequential(
            # 전치 합성곱: 14×14 특징을 28×28 출력으로 계산한다.
            nn.ConvTranspose2d(feature_channels, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),  # 0~1 범위의 정답 픽셀과 비교할 출력값을 만든다.
        )

    def forward(self, x):
        # x: (N, 1, 28, 28), features: (N, feature_channels, 14, 14).
        features = self.encoder(x)
        return self.decoder(features)  # 반환 shape: (N, 1, 28, 28).


def parameter_counts(model):
    """모듈별 파라미터 원소 수를 센다. 파라미터 텐서의 개수가 아니다."""
    counts = {}
    for name, part in [("encoder", model.encoder), ("decoder", model.decoder)]:
        counts[name] = sum(parameter.numel() for parameter in part.parameters())
    counts["total"] = sum(counts.values())
    return counts


def image_mse(prediction, target):
    """각 이미지에 대한 평균 제곱 오차 하나씩을 (N,) 텐서로 반환한다."""
    require_same_shape(prediction, target)
    # 뺄셈과 제곱은 같은 위치끼리 이루어지며 원래 shape가 유지된다.
    pixel_errors = (prediction - target).square()
    # 0번 이미지 축은 남기고 1번 축부터 합친 뒤, 각 이미지 안에서 평균낸다.
    return pixel_errors.flatten(start_dim=1).mean(dim=1)


def train_step(net, optimizer, noisy_images, clean_images):
    """한 배치의 손실·기울기를 계산하고 파라미터를 한 번 갱신한다."""
    net.train()  # 학습 모드 설정이며, 이 줄 자체는 가중치를 바꾸지 않는다.
    optimizer.zero_grad(set_to_none=True)  # 이전 기울기만 초기화한다.
    reconstructed = net(noisy_images)
    require_same_shape(reconstructed, clean_images)
    error = nn.MSELoss()(reconstructed, clean_images)
    if not torch.isfinite(error).item():
        raise ValueError("학습 손실에 NaN 또는 무한대가 발생했습니다.")
    error.backward()  # 파라미터별 기울기를 계산한다.
    optimizer.step()  # 기울기와 옵티마이저 상태로 파라미터를 갱신한다.
    # 이번 갱신 전에 계산한 손실이다. item()은 Python 숫자를 반환한다.
    return error.detach().item()


def predict_fixed(model, noisy, batch_size=256):
    """입력을 순서대로 나누어 복원하고, 같은 순서의 전체 텐서를 반환한다."""
    if noisy.shape[0] == 0:
        raise ValueError("복원할 이미지가 없습니다.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size는 양의 정수여야 합니다.")
    model.eval()  # 평가 동작 모드와 자동 미분 기록은 별개의 설정이다.
    with torch.no_grad():
        results = []
        for part in noisy.split(batch_size, dim=0):
            results.append(model(part))
        # 기존 이미지 번호 축으로 연결한다. 픽셀값을 합하거나 평균내지 않는다.
        return torch.cat(results, dim=0)


def run_experiment(data, seed, feature_channels=8):
    """기존 실습 data로 새 모델을 3에포크 학습하고 평가 결과를 반환한다.

    data 계약:
      clean_train: 0~1 범위의 깨끗한 훈련 이미지. 앞의 5,000장을 사용한다.
      clean:       0~1 범위의 깨끗한 검증 이미지.
      noisy:       clean과 같은 순서·같은 원본에 대응하는 고정 잡음 이미지.
      모든 이미지 텐서는 (N, 1, 28, 28)이어야 한다.

    shape 검사만으로 원본 ID 대응이나 훈련/검증 분리를 확인할 수는 없다.
    데이터 준비 단계에서 이 조건을 별도로 확인해야 한다.
    """
    for key in ("clean_train", "clean", "noisy"):
        if key not in data:
            raise KeyError(f"data에 필요한 키가 없습니다: {key!r}")
        if not isinstance(data[key], torch.Tensor):
            raise TypeError(f"data[{key!r}]는 torch.Tensor여야 합니다.")

    # float32 변환은 0~255 값을 0~1로 나누는 정규화가 아니다.
    # detach는 실습 입력이 다른 계산 그래프와 연결되지 않게 한다.
    clean_train = data["clean_train"][:5000].detach().to("cpu", torch.float32)
    clean_val = data["clean"].detach().to("cpu", torch.float32)
    noisy_val = data["noisy"].detach().to("cpu", torch.float32)
    for images, name in [
        (clean_train, "훈련 정답"),
        (clean_val, "검증 정답"),
        (noisy_val, "검증 입력"),
    ]:
        check_images(images, name)
    if len(clean_train) != 5000:
        raise ValueError("이 실습에는 훈련 이미지 5,000장 이상이 필요합니다.")
    require_same_shape(noisy_val, clean_val)

    # CPU 스레드 수이며 이미지 수나 채널 수를 설정하는 값이 아니다.
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    model = LabDenoiser(feature_channels=feature_channels).cpu()
    # 새 모델의 파라미터로 새 옵티마이저를 만든다. Adam 상태도 새로 시작한다.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    shuffle = torch.Generator().manual_seed(seed + 4000)
    noise_rng = torch.Generator().manual_seed(seed + 5000)
    counts = parameter_counts(model)

    initial_recon = predict_fixed(model, noisy_val)
    require_same_shape(initial_recon, clean_val)
    initial_mse = nn.MSELoss()(initial_recon, clean_val).item()
    # None은 아직 훈련 MSE가 없다는 뜻이며, 오차 0이라는 뜻이 아니다.
    history = [[0, None, initial_mse]]
    print("파라미터 원소 수:", counts)

    for epoch in range(1, 4):  # 실제 값은 1, 2, 3. 끝값 4는 포함하지 않는다.
        sse, count = 0.0, 0
        shuffled_ids = torch.randperm(len(clean_train), generator=shuffle)
        for ids in shuffled_ids.split(64):
            clean = clean_train[ids]
            noise = 0.25 * torch.randn(
                clean.shape,
                generator=noise_rng,
                dtype=clean.dtype,
                device=clean.device,
            )
            # clamp 뒤의 noisy-clean은 원래의 정규분포와 달라질 수 있다.
            noisy = (clean + noise).clamp(0, 1)
            loss = train_step(model, optimizer, noisy, clean)
            # 작은 마지막 배치까지 원소 수에 맞는 비중으로 기록한다.
            sse += loss * clean.numel()
            count += clean.numel()

        train_mse = sse / count
        # 학습 도중의 train_mse와 달리, 에포크 끝 모델을 고정 입력으로 평가한다.
        recon = predict_fixed(model, noisy_val)
        require_same_shape(recon, clean_val)
        validation_mse = nn.MSELoss()(recon, clean_val).item()
        history.append([epoch, train_mse, validation_mse])
        print(f"epoch {epoch}: train={train_mse:.4f}, validation={validation_mse:.4f}")

    baseline = nn.MSELoss()(noisy_val, clean_val).item()
    restored = nn.MSELoss()(recon, clean_val).item()
    delta = image_mse(noisy_val, clean_val) - image_mse(recon, clean_val)
    if not torch.isfinite(delta).all().item():
        raise ValueError("이미지별 오차에 NaN 또는 무한대가 있습니다.")
    print(f"입력 MSE={baseline:.4f}, 복원 MSE={restored:.4f}")
    print(
        "개선 / 악화 / 정확히 동률:",
        (delta > 0).sum().item(),
        (delta < 0).sum().item(),
        (delta == 0).sum().item(),
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "history": history,
        "recon": recon,
        "initial_recon": initial_recon,
        "clean_val": clean_val,
        "noisy_val": noisy_val,
        "baseline": baseline,
        "restored": restored,
        "delta": delta,
        "counts": counts,
    }


def plot_history(history):
    """훈련 중 평균과 에포크 종료 후 검증 MSE를 구분해서 표시한다."""
    fig, ax = plt.subplots(figsize=(7, 4))
    train_records = [record for record in history if record[1] is not None]
    ax.plot(
        [record[0] for record in train_records],
        [record[1] for record in train_records],
        marker="o",
        label="train MSE (during updates)",
    )
    ax.plot(
        [record[0] for record in history],
        [record[2] for record in history],
        marker="o",
        label="validation MSE (after epoch)",
    )
    ax.set(xlabel="epoch", ylabel="MSE")
    ax.set_xticks([record[0] for record in history])
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_triplets(clean, noisy, recon, indices=None):
    """각 열에 같은 인덱스의 원본·잡음·복원 이미지를 0~1 고정 명암으로 놓는다."""
    for images, name in [(clean, "원본"), (noisy, "입력"), (recon, "복원")]:
        check_images(images, name)
    require_same_shape(noisy, clean)
    require_same_shape(recon, clean)
    if indices is None:
        indices = list(range(min(6, clean.shape[0])))
    else:
        indices = list(indices)
    if not indices or any(
        not isinstance(index, int) or not 0 <= index < clean.shape[0]
        for index in indices
    ):
        raise ValueError("indices에는 유효한 이미지 인덱스를 하나 이상 넣으세요.")

    fig, axes = plt.subplots(3, len(indices), squeeze=False, figsize=(2.1 * len(indices), 6))
    for row, (name, images) in enumerate([
        ("clean", clean), ("noisy", noisy), ("reconstructed", recon)
    ]):
        for column, index in enumerate(indices):
            ax = axes[row, column]
            image = images[index, 0].detach().cpu().numpy()
            # 그림마다 자동 명암을 적용하지 않고 동일한 범위를 사용한다.
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                # 데이터셋 고유 ID가 아니라, 전달한 텐서의 0부터 시작하는 인덱스다.
                ax.set_title(f"Index {index}")
            if column == 0:
                ax.set_ylabel(name)
    fig.tight_layout()
    plt.show()
    return fig, axes


def locate_assets():
    """수업의 원래 도우미 파일을 찾는다. 없으면 검사한 절대 경로를 알려준다."""
    candidates = [Path("assets"), Path("../assets")]
    found = next(
        (
            directory.resolve()
            for directory in candidates
            if (directory / "d2_tools.py").is_file()
        ),
        None,
    )
    if found is None:
        checked = "\n".join(
            str((directory / "d2_tools.py").resolve()) for directory in candidates
        )
        raise FileNotFoundError(
            f"현재 작업 디렉터리: {Path.cwd()}\n"
            f"d2_tools.py를 찾지 못했습니다.\n검사한 경로:\n{checked}"
        )
    return found


if __name__ == "__main__":
    print(
        "이 파일은 수업에서 준비한 data를 사용합니다.\n"
        "원본 데이터나 d2_tools.py는 포함되어 있지 않습니다.\n"
        "함수를 불러온 뒤 result = run_experiment(data, seed=SEED)를 실행하세요.\n"
        "파일 실행만으로 학습을 시작하지 않습니다."
    )
