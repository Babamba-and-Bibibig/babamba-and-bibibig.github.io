"""합성곱을 숫자로 검산하고, 선택하면 실제 MNIST 이미지를 시각화한다.

기본 실행은 PyTorch만 사용하며 데이터를 다운로드하지 않는다.
    python examples-ko.py

MNIST 실습에는 torchvision과 matplotlib도 필요하다.
    python examples-ko.py --mnist
    python examples-ko.py --mnist --save-figure mnist-laplacian.png

--save-figure를 지정하면 창을 띄우는 대신 그림을 저장한다.
--data-root로 MNIST를 저장할 폴더를 바꿀 수 있다.
"""

import argparse  # 실행할 때 전달한 --mnist 등의 옵션을 읽는다.
from pathlib import Path  # 그림을 저장할 경로를 다룬다.
import time  # 데이터 준비와 계산에 걸린 시간을 잰다.

import torch  # 텐서를 만들고 계산한다.
from torch import nn  # 합성곱 계층 클래스를 가져온다.
import torch.nn.functional as F  # 함수 형태의 합성곱, 패딩, ReLU를 사용한다.


# 1. 원소 20개를 이미지 1장, 채널 1개, 높이 4, 너비 5로 표현한다.
X = torch.tensor([
    [1., 2., 0., 1., 3.],
    [0., 3., 1., 2., 1.],
    [2., 1., 0., 1., 0.],
    [1., 0., 2., 3., 2.],
])
x = X.reshape(1, 1, 4, 5)
assert torch.equal(x[0, 0], X)

# 정수 인덱스 0으로 이미지와 채널을 고르고, :3으로 행·열 0, 1, 2를 읽는다.
patch = x[0, 0, :3, :3]
K = torch.tensor([
    [1., 0., -1.],
    [1., 0., -1.],
    [1., 0., -1.],
])

# 같은 위치끼리 곱한 결과 9개를 만든 뒤, 그 9개를 모두 더한다.
products = patch * K
row_sums = products.sum(dim=1)  # 각 행의 열들을 합쳐 [1, -1, 2]를 얻는다.
first_output = products.sum()  # 모든 원소를 더한 0차원 텐서, tensor(2.).
print("첫 패치:\n", patch)
print("원소별 곱:\n", products)
print("행별 합계:", row_sums.tolist())
print("출력값 하나:", first_output.item())
assert first_output.item() == 2.0

# 2. nn.Conv2d(...)는 계층 생성, conv(x)는 입력을 넣어 계산하는 호출이다.
conv = nn.Conv2d(1, 1, 3, bias=False)
with torch.no_grad():  # 값을 복사하는 초기화 연산을 자동 미분으로 기록하지 않는다.
    conv.weight.copy_(K.reshape(1, 1, 3, 3))

out = conv(x)
expected = torch.tensor([[2., 2., -3.], [0., -2., 0.]])
assert tuple(out.shape) == (1, 1, 2, 3)
assert torch.equal(out[0, 0], expected)
assert out[0, 0, 0, 0].item() == first_output.item()
assert conv.weight.requires_grad is True  # no_grad() 복사로 영구 동결되지는 않는다.
print("합성곱 출력:\n", out[0, 0].detach())

# ReLU는 합성곱 다음에 별도로 적용한다. 음수만 0으로 바꾸고 shape를 유지한다.
relu_out = F.relu(out)
assert torch.equal(relu_out[0, 0], torch.tensor([[2., 2., 0.], [0., 0., 0.]]))
assert relu_out.shape == out.shape

# 3. 같은 입력·가중치로 stride와 padding 설정만 바꾼다.
# 이 딕셔너리의 값은 계산 검사용 예상값이며 데이터셋의 정답 라벨이 아니다.
expected_maps = {
    (1, 0): [[2., 2., -3.], [0., -2., 0.]],
    (2, 0): [[2., -3.]],
    (1, 1): [
        [-5., 0., 2., -3., 3.],
        [-6., 2., 2., -3., 4.],
        [-4., 0., -2., 0., 6.],
        [-1., 1., -3., 0., 4.],
    ],
    (2, 1): [[-5., 2., 3.], [-4., -2., 6.]],
}
H, W = x.shape[-2:]  # shape의 마지막 두 크기인 높이와 너비를 꺼낸다.
for (stride, padding), expected_map in expected_maps.items():
    actual = F.conv2d(x, conv.weight, stride=stride, padding=padding)
    # 좌표는 패딩된 입력을 기준으로 하며 range의 종료값은 포함하지 않는다.
    starts_h = list(range(0, H + 2 * padding - 3 + 1, stride))
    starts_w = list(range(0, W + 2 * padding - 3 + 1, stride))
    assert tuple(actual.shape[-2:]) == (len(starts_h), len(starts_w))
    assert torch.equal(actual[0, 0], torch.tensor(expected_map))
    print(f"stride={stride}, padding={padding}, 시작 행={starts_h}, 시작 열={starts_w}")
    print("출력 shape:", tuple(actual.shape))
    print(actual[0, 0].detach())

# 4. 양쪽 패딩 1칸을 실제 텐서로 만들어 첫 출력값이 -5인 이유를 확인한다.
padded_x = F.pad(x, (1, 1, 1, 1))  # 왼쪽, 오른쪽, 위쪽, 아래쪽 순서.
padded_patch = padded_x[0, 0, :3, :3]
padded_products = padded_patch * K
pad_out = F.conv2d(x, conv.weight, padding=1)
assert tuple(padded_x.shape) == (1, 1, 6, 7)
assert padded_products.sum().item() == -5.0
assert pad_out[0, 0, 0, 0].item() == -5.0
assert torch.equal(padded_x[0, 0, 1:4, 1:4], patch)
assert pad_out[0, 0, 1, 1].item() == 2.0
# 이미 패딩한 입력에는 다시 패딩을 넣지 않아야 같은 계산이 된다.
assert torch.equal(F.conv2d(padded_x, conv.weight), pad_out)
print("패딩된 첫 패치:\n", padded_patch)
print("패딩된 첫 출력값:", padded_products.sum().item())

# 길이 6, 커널 길이 3, stride 2: 허용 최댓값 M은 3, 실제 시작은 [0, 2].
L, p, k, s = 4, 1, 3, 2
last_allowed = L + 2 * p - k
starts = list(range(0, last_allowed + 1, s))
output_length = last_allowed // s + 1
assert starts == [0, 2]
assert len(starts) == output_length == 2

# 5. 라플라시안은 상하좌우의 합에서 중심값의 4배를 빼는 고정 필터다.
laplace = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
laplace_x = torch.tensor([
    [1., 2., 3., 0., 1.],
    [0., 1., 2., 3., 1.],
    [3., 0., 1., 2., 0.],
    [2., 3., 0., 1., 2.],
    [1., 2., 3., 0., 1.],
]).reshape(1, 1, 5, 5)
laplace_conv = nn.Conv2d(1, 1, 3, bias=False)
with torch.no_grad():
    laplace_conv.weight.copy_(laplace.reshape(1, 1, 3, 3))
laplace_out = laplace_conv(laplace_x)
assert torch.equal(
    laplace_out[0, 0],
    torch.tensor([[0., 0., -7.], [8., 0., -3.], [-8., 8., 0.]]),
)
print("라플라시안 출력:\n", laplace_out[0, 0].detach())

# 새 계층에는 새 가중치가 생긴다. 아래 두 줄은 출력 크기만 확인한다.
conv_pad = nn.Conv2d(1, 1, 3, padding=1, bias=False)
conv_stride = nn.Conv2d(1, 1, 3, stride=2, bias=False)
assert tuple(conv_pad(laplace_x).shape) == (1, 1, 5, 5)
assert tuple(conv_stride(laplace_x).shape) == (1, 1, 2, 2)
# 같은 필터의 값을 비교할 때는 기존 가중치를 명시적으로 전달한다.
same_filter_padded = F.conv2d(laplace_x, laplace_conv.weight, padding=1)
same_filter_strided = F.conv2d(laplace_x, laplace_conv.weight, stride=2)

# 6. 별도 계층에서 기울기만 계산한다. 옵티마이저 갱신은 하지 않는다.
grad_conv = nn.Conv2d(1, 1, 3, bias=False)
with torch.no_grad():
    grad_conv.weight.copy_(conv.weight)
weight_before = grad_conv.weight.detach().clone()  # 비교를 위한 독립 복사본.
grad_out = grad_conv(x)
grad_out.retain_grad()  # 중간 출력의 기울기도 .grad에 남긴다.
grad_loss = grad_out.square().mean()  # 확인용 목표: 출력 제곱의 평균.
grad_loss.backward()  # 기울기를 구할 뿐, 가중치 숫자를 수정하지 않는다.
assert tuple(grad_out.grad.shape) == (1, 1, 2, 3)
assert tuple(grad_conv.weight.grad.shape) == (1, 1, 3, 3)
assert torch.equal(grad_conv.weight, weight_before)
# 출력 6개의 제곱 평균이므로 각 출력의 기울기는 2 * 출력 / 6이다.
torch.testing.assert_close(grad_out.grad, 2 * grad_out.detach() / grad_out.numel())
print("확인용 손실:", grad_loss.item())
print("출력 기울기:\n", grad_out.grad[0, 0])
print("가중치 기울기:\n", grad_conv.weight.grad[0, 0])

# 7. detach는 계산된 값의 자동 미분 연결만 분리한다. 저장 공간은 공유한다.
result = laplace_conv(laplace_x)
detached_result = result.detach()
assert torch.equal(result, detached_result)
assert result.requires_grad is True
assert detached_result.requires_grad is False
assert result.data_ptr() == detached_result.data_ptr()
# 독립된 저장 공간이 필요할 때는 detach().clone()을 쓴다.
copied_result = result.detach().clone()
assert copied_result.data_ptr() != result.data_ptr()
with torch.no_grad():  # 블록 안의 계산을 처음부터 자동 미분으로 기록하지 않는다.
    result_without_graph = laplace_conv(laplace_x)
assert result_without_graph.requires_grad is False
assert laplace_conv.weight.requires_grad is True
print("기본 예제의 모든 검산을 통과했습니다.")


def run_mnist(data_root, figure_path):
    """MNIST 첫 이미지에 같은 라플라시안 필터를 적용해 표시하거나 저장한다."""
    # 기본 검산에는 필요하지 않으므로 MNIST 옵션을 선택했을 때만 가져온다.
    from torchvision import datasets, transforms
    if figure_path is not None:
        import matplotlib
        matplotlib.use("Agg")  # 화면이 없는 환경에서도 파일을 저장한다.
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    mnist = datasets.MNIST(
        data_root,
        train=True,  # 학습용 데이터 묶음을 고른다. 모델 학습 명령은 아니다.
        download=True,  # 필요한 파일이 있으면 재사용하고, 없으면 내려받는다.
        transform=transforms.ToTensor(),  # 꺼낼 때 8비트 밝기를 0~1로 변환한다.
    )
    img, label = mnist[0]  # 첫 이미지와 그 정답 숫자를 각각 받는다.
    batch = img.unsqueeze(0)  # (1,28,28) 앞에 배치 축을 추가: (1,1,28,28).
    result = laplace_conv(batch)  # 기존 라플라시안 가중치로 계산한다.
    edge = result.detach()[0, 0]  # 자동 미분 연결을 분리하고 26×26 결과를 선택한다.
    elapsed = time.perf_counter() - started

    assert tuple(img.shape) == (1, 28, 28)
    assert tuple(batch.shape) == (1, 1, 28, 28)
    assert tuple(edge.shape) == (26, 26)
    # 절댓값은 표시 범위를 고를 때만 사용한다. 그림에는 부호가 있는 edge를 준다.
    limit = max(float(edge.abs().max()), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    axes[0].imshow(img[0].cpu().numpy(), cmap="gray", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[0].set_title(f"Original MNIST image (label={label})")
    im = axes[1].imshow(edge.cpu().numpy(), cmap="coolwarm", vmin=-limit,
                        vmax=limit, interpolation="nearest")
    axes[1].set_title("Fixed Laplacian response")
    fig.colorbar(im, ax=axes[1], shrink=0.9)
    for ax in axes:
        ax.axis("off")

    print("MNIST 개수:", len(mnist), "정답:", label)
    print("MNIST 입력/출력 shape:", tuple(img.shape), tuple(edge.shape))
    print("출력 범위:", float(edge.min()), float(edge.max()), "표시 범위:", (-limit, limit))
    print("데이터 준비·계산 경과 초:", round(elapsed, 3))
    if figure_path is None:
        plt.show()
    else:
        destination = Path(figure_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180, facecolor="white")
        plt.close(fig)
        print("그림 저장:", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist", action="store_true", help="MNIST 다운로드·시각화 실행")
    parser.add_argument("--data-root", default="./data", help="MNIST 저장·조회 폴더")
    parser.add_argument("--save-figure", help="화면 표시 대신 저장할 그림 파일 경로")
    args = parser.parse_args()
    if args.save_figure and not args.mnist:
        parser.error("--save-figure는 --mnist와 함께 사용합니다.")
    if args.mnist:
        run_mnist(args.data_root, args.save_figure)
