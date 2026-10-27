# CNN의 계산 흐름: 본문의 Python 예제를 위에서 아래로 실행한다.
# PyTorch가 설치된 Python 환경에서 실행한다.
# 수치 예제와 구조 검사를 수행하며, 실제 데이터 학습은 호출하지 않는다.
# https://babamba-and-bibibig.github.io/cnn-convolution-basics/

import torch  # 텐서 생성과 계산 기능을 가져온다.
from torch import nn  # torch.nn의 층 클래스들을 nn이라는 이름으로 사용한다.
import torch.nn.functional as F  # 신경망 계산 함수들을 F라는 이름으로 사용한다.

conv_example = nn.Conv2d(
    in_channels=3,       # 실제 입력이 RGB이므로 입력 채널을 3으로 맞춘다.
    out_channels=32,     # 필터 32개를 만든다. 출력 채널 수도 32가 된다.
    kernel_size=3,       # 채널별 커널의 세로·가로를 3×3으로 정한다.
    stride=1,           # 계산 영역의 시작 위치를 한 칸씩 옮긴다.
    padding=1,          # 입력의 위·아래·왼쪽·오른쪽에 각각 한 칸을 추가해 계산한다.
    dilation=1,          # 커널 가중치를 이웃한 입력 칸에 적용한다.
    groups=1,           # 각 필터가 모든 입력 채널을 사용한다.
    bias=True,          # 출력 채널마다 학습 가능한 편향 하나를 만든다.
    padding_mode="zeros",  # 추가한 가장자리 칸은 0으로 취급한다.
)


x7 = torch.arange(1, 50, dtype=torch.float32).reshape(1, 1, 7, 7)
# 이미지 1장, 채널 1개, 세로·가로 7칸이다.

kernel3 = torch.tensor(
    [[1, 2, -1], [-2, 1, 3], [2, -1, 1]],
    dtype=torch.float32,  # 합성곱 가중치와 같은 실수 자료형을 사용한다.
)

patch = x7[0, 0, 0:3, 0:3]  # 첫 이미지·첫 채널에서 행 0~2, 열 0~2를 선택한다.
products = patch * kernel3  # 같은 위치끼리 곱한다. 결과 shape는 (3, 3)이다.


print(products.tolist())  # 텐서를 Python 리스트로 바꾸어 값들을 확인한다.
# [[1.0, 4.0, -3.0], [-16.0, 9.0, 30.0], [30.0, -16.0, 17.0]]

print(products.sum(dim=1).tolist())  # 각 행 안의 세 값을 더한다: [2.0, 23.0, 31.0].
print(products.sum().item())  # 아홉 값을 모두 더한다: 56.0.
print((products.sum() + 2).item())  # 편향 2를 더한 최종 출력 한 칸: 58.0.


conv3 = nn.Conv2d(
    in_channels=1, out_channels=32, kernel_size=3,
    stride=1, padding=0, bias=True,
)  # 입력 채널 1개, 필터 32개인 층을 만든다.

with torch.no_grad():  # 이 구간의 역전파 계산 기록은 만들지 않는다.
    conv3.weight[0, 0].copy_(kernel3)  # 첫 필터의 첫 채널용 가중치를 지정한다.
    conv3.bias[0].fill_(2.0)  # 첫 출력 채널의 편향을 2로 지정한다.
    y3 = conv3(x7)  # 실제 합성곱: 전체 출력 shape는 (1, 32, 5, 5)다.

print(y3[0, 0])  # 첫 이미지의 첫 출력 채널만 확인한다.


manual3 = torch.empty_like(y3[0, 0])  # 첫 출력 채널과 같은 (5, 5) 공간을 만든다.
with torch.no_grad():
    for row in range(y3.shape[2]):  # 출력의 세로 인덱스 0~4를 선택한다.
        for col in range(y3.shape[3]):  # 출력의 가로 인덱스 0~4를 선택한다.
            patch = x7[0, 0, row:row + 3, col:col + 3]  # 현재 위치의 3×3 영역.
            manual3[row, col] = (patch * kernel3).sum() + conv3.bias[0]

assert torch.allclose(manual3, y3[0, 0])  # 작은 실수 오차를 허용해 두 결과를 비교한다.


kernel5 = torch.tensor(
    [[1, -1, 2, 1, -2],
     [2, 1, -1, 2, 1],
     [-1, 2, 1, -2, 2],
     [1, -2, 2, 1, -1],
     [2, 1, -1, 2, 1]],
    dtype=torch.float32,  # 5×5 가중치도 실수로 만든다.
)
conv5 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
with torch.no_grad():
    conv5.weight[0, 0].copy_(kernel5)  # 첫 필터의 커널을 직접 지정한다.
    conv5.bias[0].fill_(2.0)  # 비교를 위해 편향은 같은 2를 사용한다.
    y5 = conv5(x7)  # 같은 원본 x7을 입력한다: (1, 32, 3, 3).
    first5 = (x7[0, 0, :5, :5] * kernel5).sum() + conv5.bias[0]

print(first5.item())  # 263.0: 첫 위치에서 25개 곱을 합친 261에 편향 2를 더한다.
print(y5[0, 0].tolist())
# [[263.0, 277.0, 291.0], [361.0, 375.0, 389.0], [459.0, 473.0, 487.0]]


import torch  # 텐서와 장치 선택 기능을 사용한다.
from torch import nn  # 신경망 층과 모델 기본 클래스를 사용한다.
import torch.nn.functional as F  # ReLU 계산 함수를 사용한다.


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()  # 하위 층과 파라미터를 관리할 기본 기능을 초기화한다.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 입력 1채널 → 출력 32채널. 28×28 공간 크기는 유지한다.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 입력 32채널 → 출력 64채널. 들어오는 공간 크기를 유지한다.
        self.pool = nn.MaxPool2d(kernel_size=2)  # 기본 stride도 2다.
        self.fc = nn.Linear(64 * 7 * 7, 10)  # 이미지당 값 3136개 → 점수 10개.

    def features(self, x, trace=False):
        if trace:
            print("입력:", tuple(x.shape))  # tuple은 shape를 일반 튜플로 표시한다.
        x = self.conv1(x)
        if trace:
            print("conv1:", tuple(x.shape))
        x = F.relu(x)  # 음수만 0으로 바꾼다. shape는 유지한다.
        if trace:
            print("ReLU 1:", tuple(x.shape))
        x = self.pool(x)  # 28×28 → 14×14.
        if trace:
            print("Pool 1:", tuple(x.shape))
        x = self.conv2(x)
        if trace:
            print("conv2:", tuple(x.shape))
        x = F.relu(x)
        if trace:
            print("ReLU 2:", tuple(x.shape))
        x = self.pool(x)  # 14×14 → 7×7. 같은 풀링 객체를 다시 호출한다.
        if trace:
            print("Pool 2:", tuple(x.shape))
        return x  # 마지막 특징 맵을 반환한다.

    def forward(self, x, trace=False):
        x = self.features(x, trace=trace)  # 합성곱·활성화·풀링을 순서대로 실행한다.
        x = x.flatten(start_dim=1)  # 배치 차원을 남기고 이미지당 값을 펼친다.
        if trace:
            print("Flatten:", tuple(x.shape))
        if x.shape[1] != self.fc.in_features:  # 실제 값 수와 Linear의 요구값을 비교한다.
            raise ValueError(
                f"Linear에 필요한 값은 이미지당 {self.fc.in_features}개, "
                f"현재 전달된 값은 이미지당 {x.shape[1]}개입니다."
            )
        x = self.fc(x)  # 클래스별 로짓 10개를 만든다.
        if trace:
            print("Linear:", tuple(x.shape))
        return x


# CUDA는 NVIDIA GPU, MPS는 Apple GPU를 사용할 때의 장치 이름이다.
if torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU를 사용할 수 있으면 선택한다.
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple GPU를 사용할 수 있으면 선택한다.
else:
    device = torch.device("cpu")  # GPU를 사용할 수 없으면 CPU에서 실행한다.

my_model = MyCNN().to(device)  # 모델을 만들고 파라미터를 선택한 장치로 옮긴다.
my_model.eval()  # 평가 모드로 전환한다. 이 호출이 자동 미분을 끄지는 않는다.
probe = torch.zeros(2, 1, 28, 28, device=device)  # 연결 확인용 0 입력 두 장.

with torch.no_grad():  # 역전파 기록 없이 순전파만 확인한다.
    output = my_model(probe, trace=True)

assert output.shape == (2, 10)  # 이미지 두 장에 점수 10개씩 나왔는지 확인한다.


for name, parameter in my_model.named_parameters():
    print(name, tuple(parameter.shape), parameter.numel())  # 이름·shape·원소 수.

# 파라미터마다 원소 수를 구하고, Python의 sum으로 모두 더한다.
parameter_count = sum(p.numel() for p in my_model.parameters())
assert parameter_count == 50186  # 손으로 센 전체 파라미터 수와 비교한다.

with torch.no_grad():
    feature_maps = my_model.features(probe)  # (2, 64, 7, 7).
    flattened = feature_maps.flatten(start_dim=1)  # (2, 3136).
    print("Linear 입력 수:", flattened.shape[1])  # 3136: 배치 크기를 제외한 값 수.
    scores_direct = flattened @ my_model.fc.weight.T + my_model.fc.bias
    scores_layer = my_model.fc(flattened)  # 같은 계산을 Linear에 맡긴다.
    assert torch.allclose(scores_direct, scores_layer)  # 두 방식의 점수를 비교한다.
    predicted_classes = scores_layer.argmax(dim=1)  # 이미지마다 최대 점수의 위치.
    print("예측 번호의 shape:", tuple(predicted_classes.shape))  # (2,).


def train_one_batch(model, images, labels, optimizer):
    model_device = next(model.parameters()).device  # 첫 파라미터가 있는 장치를 읽는다.
    images = images.to(device=model_device, dtype=torch.float32)  # 실수 이미지 입력.
    labels = labels.to(device=model_device, dtype=torch.long)  # 정답 번호는 정수형.
    model.train()  # 학습 모드로 바꾼다.
    optimizer.zero_grad(set_to_none=True)  # 이전 .grad를 None으로 비운다.
    logits = model(images)  # (N, 10): 확률로 바꾸지 않은 점수.
    loss = F.cross_entropy(logits, labels)  # 이미지들의 손실을 기본 설정으로 평균 낸다.
    loss.backward()  # 각 학습 파라미터의 기울기를 계산한다.
    optimizer.step()  # 옵티마이저가 실제 파라미터를 갱신한다.
    return loss.item()  # 숫자 하나인 손실 텐서에서 Python 숫자를 꺼낸다.


optimizer = torch.optim.Adam(
    my_model.parameters(),  # 갱신할 파라미터들을 전달한다.
    lr=0.001,  # 학습률: 파라미터 갱신 크기에 사용하는 설정값.
)

# 실제 이미지와 정답을 준비한 뒤 다음과 같이 호출한다.
# loss_value = train_one_batch(my_model, images, labels, optimizer)


def conv_output_length(input_length, kernel, padding=0, stride=1, dilation=1):
    # 모든 인자는 정수 크기다. 입력·커널·이동 간격·적용 간격은 양수여야 한다.
    if input_length <= 0 or kernel <= 0 or stride <= 0 or dilation <= 0:
        raise ValueError("입력·커널·stride·dilation은 양수여야 합니다.")
    if padding < 0:
        raise ValueError("패딩은 0 이상이어야 합니다.")
    effective_kernel = dilation * (kernel - 1) + 1  # 간격을 포함한 커널 적용 범위.
    padded_length = input_length + 2 * padding  # 양쪽 패딩을 더한 입력 길이.
    if padded_length < effective_kernel:
        raise ValueError("커널이 패딩을 포함한 입력보다 큽니다.")
    return (padded_length - effective_kernel) // stride + 1  # 시작 위치의 개수.


assert conv_output_length(7, 3) == 5  # 패딩이 없으면 출력 길이 5.
assert conv_output_length(28, 3, padding=1) == 28  # 패딩 1이면 길이 유지.
