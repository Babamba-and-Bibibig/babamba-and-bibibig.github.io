# CNN 잔차 연결: CPU에서 shape, Identity, 1×1 계산을 확인한다.
# 본문: https://babamba-and-bibibig.github.io/cnn-residual-projection/

import torch
from torch import nn
import torch.nn.functional as F

# 같은 실행 순서에서 입력과 초기 가중치를 재현하기 쉽게 한다.
# 이번 예제의 모델과 텐서는 모두 기본 장치인 CPU에 둔다.
torch.manual_seed(16)


class MyProjectionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        # nn.Module의 계층·파라미터 관리 기능을 먼저 초기화한다.
        super().__init__()

        # 주 경로 1: 채널을 바꾸고, 전달받은 stride로 공간을 조절한다.
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,  # 출력 채널마다 편향 하나를 사용한다.
        )

        # 주 경로 2: 첫 합성곱의 채널·공간 크기를 유지한다.
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # 이 주 경로의 K·P 설정에서 채널과 공간을 유지하는 조건이다.
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            # 같은 원래 입력을 주 경로의 최종 shape에 맞게 계산한다.
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=True,
            )

    def forward(self, x):
        # F.relu는 음수를 0으로 바꾸며 shape는 유지한다.
        main = F.relu(self.conv1(x))
        main = self.conv2(main)

        # conv1의 출력이 아니라 원래 입력 x를 전달한다.
        skip = self.shortcut(x)

        # 먼저 같은 위치의 값끼리 더하고, 그 결과에 ReLU를 적용한다.
        return F.relu(main + skip)


# 각 항목은 입력채널, 출력채널, stride, 입력shape, 예상출력shape다.
cases = [
    (8, 16, 2, (2, 8, 16, 16), (2, 16, 8, 8)),
    (8, 8, 1, (1, 8, 15, 13), (1, 8, 15, 13)),
    (8, 16, 1, (1, 8, 15, 13), (1, 16, 15, 13)),
    (8, 8, 2, (1, 8, 15, 13), (1, 8, 8, 7)),
    (16, 8, 1, (1, 16, 15, 13), (1, 8, 15, 13)),
]

for in_c, out_c, stride, input_shape, expected_shape in cases:
    block = MyProjectionResidualBlock(in_c, out_c, stride)
    x = torch.randn(input_shape)

    with torch.no_grad():
        # 두 경로를 따로 확인한다. 아직 둘을 더한 결과가 아니다.
        main = block.conv2(F.relu(block.conv1(x)))
        skip = block.shortcut(x)
        assert main.shape == skip.shape, "두 경로의 shape가 다릅니다."

        # block(x)는 PyTorch의 호출 절차를 통해 forward를 실행한다.
        y = block(x)
        assert tuple(y.shape) == expected_shape

        # allclose는 실수의 작은 계산 오차를 허용해 값들을 비교한다.
        # atol은 절대 오차, rtol은 비교값 크기에 비례하는 오차 기준이다.
        assert torch.allclose(y, F.relu(main + skip), atol=1e-6, rtol=1e-5)

    expected_identity = in_c == out_c and stride == 1
    assert isinstance(block.shortcut, nn.Identity) == expected_identity

    if expected_identity:
        # Identity는 복사본이 아니라 같은 입력 객체를 반환한다.
        assert skip is x
        shortcut_name = "Identity"
    else:
        # 가중치의 축은 출력필터, 입력채널, 커널높이, 커널너비다.
        assert tuple(block.shortcut.weight.shape) == (out_c, in_c, 1, 1)
        shortcut_name = "1x1 Conv"

    print(f"{input_shape} -> {tuple(y.shape)} | {shortcut_name}")


# 1×1 Conv의 한 출력 위치를 표의 숫자로 검산한다.
# zeros는 지정한 shape를 0으로 채운 텐서를 만든다.
point_x = torch.zeros(1, 8, 16, 16)
projection = nn.Conv2d(8, 16, kernel_size=1, stride=2, padding=0)

with torch.no_grad():
    # :는 입력 채널 8개 전체를 선택한다.
    # torch.tensor는 주어진 숫자들을 텐서로 만든다.
    point_x[0, :, 4, 6] = torch.tensor([1., 2., 0., -1., 3., 1., 2., -2.])
    projection.weight[0, :, 0, 0] = torch.tensor(
        [0.5, -1., 2., 1., 0., 0.25, -0.5, 1.]
    )
    projection.bias[0] = 0.5

    point_y = projection(point_x)
    # stride=2이므로 출력 위치 (2,3)은 입력 위치 (4,6)을 읽는다.
    actual = point_y[0, 0, 2, 3]
    expected = torch.tensor(-4.75)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

# item은 원소 하나짜리 텐서에서 일반 Python 숫자를 꺼낸다.
print("1x1 Conv의 한 출력값:", actual.item())


# 같은 stride라도 K·P가 다르면 출력 공간이 다를 수 있다.
size_x = torch.zeros(1, 8, 16, 16)
settings = [
    (3, 1, 2, (8, 8)),
    (1, 0, 2, (8, 8)),
    (1, 1, 2, (9, 9)),
    (3, 0, 2, (7, 7)),
]

with torch.no_grad():
    for kernel, padding, stride, expected_hw in settings:
        conv = nn.Conv2d(8, 16, kernel, stride=stride, padding=padding)
        # shape[2:]는 축 2·3, 즉 출력 높이·너비를 선택한다.
        actual_hw = tuple(conv(size_x).shape[2:])
        assert actual_hw == expected_hw
        print(f"K={kernel}, P={padding}, S={stride} -> {actual_hw}")


# None과 False는 이 예제에서 정한 준비 상태의 초기값이다.
projection_block = None
projection_ready = False

projection_block = MyProjectionResidualBlock(8, 16, stride=2)
with torch.no_grad():
    check_y = projection_block(torch.randn(2, 8, 16, 16))

assert tuple(check_y.shape) == (2, 16, 8, 8)
projection_ready = True  # 검사를 통과한 뒤 상태 표시를 바꾼다.
print("블록 구조 확인 완료:", projection_ready)
