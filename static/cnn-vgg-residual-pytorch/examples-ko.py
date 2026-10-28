"""CNN 연결 구조를 검산하는 독립 실행 예제.

실행: python examples-ko.py
필요한 패키지: torch

실제 사진 대신 CPU의 작은 무작위 텐서를 사용한다.
입력 데이터, 모델 파라미터, 출력 데이터를 구분하며 계산을 확인한다.
train_one_batch는 학습 순서 설명용으로 정의만 한다.
직접 실행해도 이 함수를 호출하거나 가중치를 갱신하지 않는다.
"""

import torch
from torch import nn
import torch.nn.functional as F


class MyVGGBlock(nn.Module):
    """공간 크기를 유지하는 Conv 두 층 뒤에 MaxPool을 적용한다."""

    def __init__(self, in_channels, out_channels):
        # 부모 클래스가 하위 모듈과 파라미터를 관리할 준비를 한다.
        super().__init__()

        # 첫 층은 원래 입력 채널을 블록의 출력 채널 수로 바꾼다.
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
        )

        # 둘째 층은 첫 층의 출력 데이터를 받는다. 가중치는 별개다.
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
        )

        # Pool에는 학습할 가중치가 없다. 채널별 공간 크기만 줄인다.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        result = F.relu(self.conv1(x))
        result = F.relu(self.conv2(result))
        return self.pool(result)


class MySmallVGG(nn.Module):
    """32×32 RGB 입력에서 이미지마다 클래스 점수 다섯 개를 만든다."""

    def __init__(self):
        super().__init__()
        self.block1 = MyVGGBlock(3, 8)
        self.block2 = MyVGGBlock(8, 16)

        # Pool 두 번 뒤의 특징값 수: 16채널 × 높이 8 × 너비 8.
        # 배치 크기는 이미지 한 장의 특징값 수에 곱하지 않는다.
        self.fc = nn.Linear(16 * 8 * 8, 5)

    def forward(self, x):
        result = self.block1(x)              # [N, 8, 16, 16]
        result = self.block2(result)         # [N, 16, 8, 8]
        result = result.flatten(start_dim=1) # [N, 1024]: 배치 축 유지.
        return self.fc(result)              # [N, 5]: 확률 전의 점수.


class ResidualBlock(nn.Module):
    """같은 shape의 주 경로 결과와 원래 입력을 더하는 교육용 블록."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)

        # 같은 배치·채널·행·열 위치의 값을 더한다.
        # concat이나 행렬곱이 아니다. x는 원래 입력으로 남아 있다.
        return F.relu(out + x)


class ProjectionResidualBlock(nn.Module):
    """두 경로를 [N, out_channels, ceil(H/2), ceil(W/2)]로 맞춘다."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1
        )

        # 1×1은 공간 커널 크기다. 모든 입력 채널의 값을 조합한다.
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, 1, stride=2
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        shortcut_result = self.shortcut(x)
        return F.relu(out + shortcut_result)


def train_one_batch(model, images, labels, optimizer, criterion):
    """학습 순서 설명용 함수. run_checks에서는 호출하지 않는다.

    images: 모델과 같은 장치의 [N, 3, 32, 32] 입력.
    labels: 같은 장치의 [N] 정답. 정수 클래스 방식이면 torch.long, 0~4.
    criterion: 여기서는 nn.CrossEntropyLoss()를 사용하는 경우를 설명한다.
    optimizer: model.parameters()를 전달해 만든 파라미터 갱신 객체.
    """
    model.train()  # Dropout·BatchNorm 등의 동작을 학습 방식으로 설정한다.

    # 이전 배치에서 누적된 기울기를 제거한다.
    # None으로 설정하는 것과 0 텐서를 저장하는 것은 세부 동작이 다르다.
    optimizer.zero_grad(set_to_none=True)

    # 학습의 순전파는 no_grad로 감싸지 않는다.
    # CrossEntropyLoss에는 Softmax를 적용하기 전 로짓을 전달한다.
    logits = model(images)
    loss = criterion(logits, labels)

    loss.backward()  # 파라미터의 기울기를 계산한다.
    optimizer.step() # 계산한 기울기로 파라미터를 갱신한다.
    return loss.item()


def run_checks():
    """출력 크기와 계산값만 검사한다. 손실 계산·역전파·갱신은 하지 않는다."""
    device = torch.device("cpu")
    torch.manual_seed(16)

    # 시드는 같은 실행 순서의 무작위 값을 재현하기 쉽게 만든다.
    # 모든 장치·모든 버전에서 동일한 부동소수점 결과를 보장하지 않는다.
    # 아래 일반 합성곱은 groups=1, dilation=1을 사용한다.
    print("PyTorch:", torch.__version__)

    # ------------------------------------------------------------------
    # 1. 입력 네 채널과 필터 여섯 개: 출력 숫자 하나를 직접 계산한다.
    # ------------------------------------------------------------------
    conv = nn.Conv2d(
        in_channels=4,
        out_channels=6,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,  # 곱셈과 합산만 검산하려고 이번 편향은 제외한다.
    ).to(device)
    x = torch.randn(1, 4, 7, 7, device=device)

    assert tuple(conv.weight.shape) == (6, 4, 3, 3)
    assert conv.weight.numel() == 216
    print("전체 가중치:", tuple(conv.weight.shape), conv.weight.numel())
    print("필터 하나:", tuple(conv.weight[0].shape))
    print("채널별 커널 하나:", tuple(conv.weight[0, 0].shape))

    with torch.no_grad():
        y = conv(x)
        actual_value = y[0, 0, 3, 3]

        # 출력 행 3, 열 3에 필요한 원래 입력의 행·열 2, 3, 4.
        # ':'는 입력 네 채널 모두를 고른다. 끝 인덱스 5는 제외한다.
        patch = x[0, :, 2:5, 2:5]  # [4, 3, 3]: 입력 데이터.
        kernels = conv.weight[0]   # [4, 3, 3]: 필터 하나의 가중치.
        products = patch * kernels # [4, 3, 3]: 같은 위치끼리 곱한 36개 값.

        # 지금 텐서의 1·2번 축은 선택한 영역의 높이와 너비다.
        channel_results = products.sum(dim=(1, 2)) # [4]: 채널별 합계.
        manual_value = channel_results.sum()      # []: 채널까지 합한 값.
        assert torch.allclose(
            manual_value, actual_value, atol=1e-6, rtol=1e-4
        )

    assert tuple(y.shape) == (1, 6, 7, 7)
    print("출력:", tuple(y.shape))
    print("채널별 계산 결과:", channel_results)
    print("직접 계산 / Conv 출력:", manual_value.item(), actual_value.item())

    # 필터 하나의 가중치는 36개지만, 출력 채널 하나에는 49개 값이 있다.
    # 같은 가중치를 여러 공간 위치에서 반복 사용하기 때문이다.
    assert kernels.numel() == 36
    assert y[0, 0].numel() == 49

    # ------------------------------------------------------------------
    # 2. ReLU 함수 자체, 계층 객체, 실제 계산 결과를 구분한다.
    # ------------------------------------------------------------------
    relu_input = torch.tensor([-2.0, 3.0, 0.0, -4.0], device=device)
    relu_function = F.relu # 괄호가 없으므로 함수 자체를 저장한다.
    relu_layer = nn.ReLU() # ReLU 계층 객체를 만든다. 아직 입력 계산은 없다.
    relu_result1 = relu_function(relu_input)
    relu_result2 = relu_layer(relu_input)

    assert torch.equal(relu_result1, relu_result2)
    assert tuple(relu_result1.shape) == tuple(relu_input.shape)
    print("ReLU 결과:", relu_result1)

    # ------------------------------------------------------------------
    # 3. MaxPool은 각 채널 안의 2×2 영역에서 최댓값을 고른다.
    # ------------------------------------------------------------------
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    pool_input = torch.tensor(
        [[[
            [3.0, 7.0, 1.0, 4.0],
            [2.0, 5.0, 8.0, 0.0],
            [6.0, 1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0, 9.0],
        ]]],
        device=device,
    )
    pool_output = pool(pool_input)
    expected_pool = torch.tensor([[[[7.0, 8.0], [6.0, 9.0]]]], device=device)
    assert torch.equal(pool_output, expected_pool)
    assert tuple(pool_output.shape) == (1, 1, 2, 2)
    print("Pool 결과:", pool_output)

    # ------------------------------------------------------------------
    # 4. 입력 없이 생성한 파라미터 수를 비교한다.
    # ------------------------------------------------------------------
    channels = 16
    two_small = nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
    ).to(device)
    one_large = nn.Conv2d(
        channels, channels, 5, padding=2, bias=False
    ).to(device)

    # 각 Conv는 별도 객체이므로 가중치도 별개다. ReLU에는 파라미터가 없다.
    small_count = sum(p.numel() for p in two_small.parameters())
    large_count = sum(p.numel() for p in one_large.parameters())
    assert small_count == 2 * 16 * 16 * 3 * 3 == 4608
    assert large_count == 16 * 16 * 5 * 5 == 6400
    print("3×3 두 층 / 5×5 한 층 가중치:", small_count, large_count)

    # parameters()는 동결된 파라미터도 제공한다.
    # 현재 학습 가능한 것만 세려면 requires_grad 조건을 붙인다.
    trainable_count = sum(
        p.numel() for p in two_small.parameters() if p.requires_grad
    )
    assert trainable_count == small_count

    # ------------------------------------------------------------------
    # 5. 실제 입력을 전달하고 중간 3×3 영역과 원래 입력 5×5를 확인한다.
    # ------------------------------------------------------------------
    rf_x = torch.randn(1, 16, 7, 7, device=device)
    with torch.no_grad():
        rf_m = two_small[1](two_small[0](rf_x))
        rf_y = two_small[2](rf_m)
        rf_large = one_large(rf_x)

    assert rf_x.shape == rf_m.shape == rf_y.shape == rf_large.shape
    middle_patch = rf_m[0, :, 2:5, 2:5] # 출력 [0,0,3,3]이 읽는 중간값.
    input_region = rf_x[0, :, 1:6, 1:6] # 그 중간값들과 연결된 원래 입력 범위.
    assert tuple(middle_patch.shape) == (16, 3, 3)
    assert tuple(input_region.shape) == (16, 5, 5)
    print("중간 패치 / 원래 입력 범위:", middle_patch.shape, input_region.shape)

    # 두 구조는 이론적 수용 영역이 같다. 출력 숫자가 같다는 뜻은 아니다.
    # 이 조건은 stride=1, dilation=1, 중간 Pool 없음인 내부 위치 기준이다.
    # ReLU는 한 위치의 값만 바꾸므로 참조 범위를 넓히지 않는다.

    # ------------------------------------------------------------------
    # 6. VGG 블록은 Conv 사이에서 공간 크기를 유지하고 Pool에서 줄인다.
    # ------------------------------------------------------------------
    vgg_block = MyVGGBlock(3, 8).to(device)
    with torch.no_grad():
        for input_shape, expected_shape in [
            ((2, 3, 32, 32), (2, 8, 16, 16)),
            ((1, 3, 30, 26), (1, 8, 15, 13)),
        ]:
            # *는 튜플 네 값을 각각의 위치 인자로 풀어 전달한다.
            images = torch.randn(*input_shape, device=device)
            after_conv1 = F.relu(vgg_block.conv1(images))
            after_conv2 = F.relu(vgg_block.conv2(after_conv1))
            after_pool = vgg_block.pool(after_conv2)
            block_forward = vgg_block(images)

            assert tuple(block_forward.shape) == expected_shape
            assert torch.allclose(
                block_forward, after_pool, atol=1e-6, rtol=1e-4
            )
            print(
                "VGG 블록:", input_shape,
                "→", tuple(after_conv1.shape),
                "→", tuple(after_conv2.shape),
                "→", tuple(after_pool.shape),
            )

    # 계층별 실행과 전체 실행을 비교하려고 같은 입력을 두 번 계산했다.
    # 일반 사용에서는 vgg_block(images) 한 번이면 된다.

    # ------------------------------------------------------------------
    # 7. 두 블록, Flatten, Linear를 연결해 배치 2와 1을 검사한다.
    # ------------------------------------------------------------------
    torch.manual_seed(16)
    small_model = MySmallVGG().to(device)
    with torch.no_grad():
        for batch_size in (2, 1):
            images = torch.randn(batch_size, 3, 32, 32, device=device)
            feature1 = small_model.block1(images)
            feature2 = small_model.block2(feature1)
            flat_features = feature2.flatten(start_dim=1)
            manual_logits = small_model.fc(flat_features)
            logits = small_model(images)

            assert tuple(feature1.shape) == (batch_size, 8, 16, 16)
            assert tuple(feature2.shape) == (batch_size, 16, 8, 8)
            assert tuple(flat_features.shape) == (batch_size, 1024)
            assert tuple(logits.shape) == (batch_size, 5)
            assert torch.allclose(
                logits, manual_logits, atol=1e-6, rtol=1e-4
            )

            # Linear의 weight 저장 순서는 [출력 수, 입력 수]다.
            # [N,1024] @ [1024,5] + [5] = [N,5].
            # 편향 다섯 개는 각 이미지의 점수에 똑같이 적용된다.
            matrix_logits = flat_features @ small_model.fc.weight.T
            matrix_logits = matrix_logits + small_model.fc.bias
            assert torch.allclose(
                manual_logits, matrix_logits, atol=1e-6, rtol=1e-4
            )
            print(
                "분류 모델:", tuple(feature1.shape),
                "→", tuple(feature2.shape),
                "→", tuple(flat_features.shape),
                "→", tuple(logits.shape),
            )

    # 이미지 한 장의 특징값 1024개와, 전체 모델의 파라미터 수는 다르다.
    assert tuple(small_model.fc.weight.shape) == (5, 1024)
    assert small_model.fc.weight.numel() == 5120
    assert small_model.fc.bias.numel() == 5
    parameter_count = sum(p.numel() for p in small_model.parameters())
    assert parameter_count == 224 + 584 + 1168 + 2320 + 5125 == 9421
    print("전체 모델 파라미터:", parameter_count)

    # ------------------------------------------------------------------
    # 8. 로짓을 확률 형태로 바꾸되, 학습된 모델의 정확도로 해석하지 않는다.
    # ------------------------------------------------------------------
    small_model.eval()
    with torch.no_grad():
        prediction_images = torch.randn(2, 3, 32, 32, device=device)
        logits = small_model(prediction_images)
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = logits.argmax(dim=1)
        probability_sums = probabilities.sum(dim=1)

        assert torch.allclose(probability_sums, torch.ones(2, device=device))
        assert torch.equal(predicted_classes, probabilities.argmax(dim=1))

    print("확률 합계:", probability_sums)
    print("학습 전 점수의 최대 클래스 번호:", predicted_classes)

    # ------------------------------------------------------------------
    # 9. 사람이 정한 작은 배열에서 잔차 덧셈과 ReLU를 확인한다.
    # ------------------------------------------------------------------
    residual_x = torch.tensor([[2.0, -1.0], [0.0, 3.0]], device=device)
    residual_f = torch.tensor([[-1.0, 2.0], [1.0, -4.0]], device=device)
    residual_sum = residual_f + residual_x
    residual_y = F.relu(residual_sum)
    expected_sum = torch.tensor([[1.0, 1.0], [1.0, -1.0]], device=device)
    expected_y = torch.tensor([[1.0, 1.0], [1.0, 0.0]], device=device)
    assert torch.equal(residual_sum, expected_sum)
    assert torch.equal(residual_y, expected_y)
    print("잔차 합:", residual_sum)
    print("잔차 합 뒤 ReLU:", residual_y)
    print("shortcut 없이 ReLU:", F.relu(residual_f))

    # 위 residual_f는 사람이 정한 값이다. 아래에서 실제 Conv를 사용한다.
    block = ResidualBlock(16).to(device)
    sample = torch.randn(1, 16, 14, 14, device=device)
    with torch.no_grad():
        transformed = block.conv2(F.relu(block.conv1(sample)))
        block_output = block(sample)
        assert transformed.shape == sample.shape
        assert torch.allclose(
            block_output, F.relu(transformed + sample), atol=1e-6, rtol=1e-4
        )

    index = (0, 0, 0, 0) # 첫 데이터, 첫 채널, 첫 행, 첫 열.
    x_value = sample[index].item()
    f_value = transformed[index].item()
    print("잔차 한 위치의 입력 / 주 경로:", x_value, f_value)
    print("두 값의 합 / 실제 ReLU 출력:", x_value + f_value, block_output[index].item())

    # shape만 맞는다고 shortcut을 사용했다는 뜻은 아니다.
    # 실제 forward에서 out + x를 계산하는지도 확인해야 한다.

    # ------------------------------------------------------------------
    # 10. Projection은 덧셈 전에 두 경로의 채널과 공간 크기를 맞춘다.
    # ------------------------------------------------------------------
    projection_block = ProjectionResidualBlock(16, 32).to(device)
    projection_input = torch.randn(1, 16, 14, 14, device=device)
    with torch.no_grad():
        main_path = projection_block.conv2(
            F.relu(projection_block.conv1(projection_input))
        )
        shortcut_path = projection_block.shortcut(projection_input)
        projection_output = projection_block(projection_input)

        # 브로드캐스팅에 의해 우연히 덧셈 가능한지만 확인하지 않는다.
        # 여기서는 두 경로의 모든 축 크기가 정확히 같아야 한다.
        assert tuple(main_path.shape) == (1, 32, 7, 7)
        assert main_path.shape == shortcut_path.shape
        assert torch.allclose(
            projection_output, F.relu(main_path + shortcut_path),
            atol=1e-6, rtol=1e-4,
        )

    assert tuple(projection_block.shortcut.weight.shape) == (32, 16, 1, 1)
    assert projection_block.shortcut.weight.numel() == 512
    assert projection_block.shortcut.bias.numel() == 32
    print("Projection 주 경로 / shortcut / 출력:",
          main_path.shape, shortcut_path.shape, projection_output.shape)
    print("1×1 shortcut 가중치 / 편향:",
          projection_block.shortcut.weight.numel(),
          projection_block.shortcut.bias.numel())

    # 1×1 커널 하나는 가중치 한 개지만, 필터 하나는 16개를 가진다.
    # 필터가 32개이므로 전체 가중치는 32×16×1×1 = 512개다.
    # 두 잔차 블록은 BatchNorm 등을 생략한 교육용 구성이다.

    # ------------------------------------------------------------------
    # 11. eval은 계층의 동작 모드, no_grad는 역전파 기록을 제어한다.
    # ------------------------------------------------------------------
    probe_input = torch.randn(1, 3, 32, 32, device=device)
    small_model.eval()
    tracked_output = small_model(probe_input)
    with torch.no_grad():
        untracked_output = small_model(probe_input)

    # 입력이 기울기를 요구하지 않아도 가중치 때문에 출력은 추적될 수 있다.
    assert probe_input.requires_grad is False
    assert tracked_output.requires_grad is True
    assert untracked_output.requires_grad is False
    assert small_model.fc.weight.requires_grad is True
    assert small_model.training is False
    print("eval만 사용:", tracked_output.requires_grad)
    print("eval + no_grad:", untracked_output.requires_grad)
    print("기존 가중치의 requires_grad:", small_model.fc.weight.requires_grad)

    # backward를 호출하지 않았으므로 파라미터 기울기도 쌓이지 않았다.
    # eval 자체가 자동 미분을 끄거나, no_grad가 기존 .grad를 지우지는 않는다.
    assert all(p.grad is None for p in small_model.parameters())
    print("모든 구조·계산 검산을 통과했습니다. 학습과 가중치 갱신은 실행하지 않았습니다.")


if __name__ == "__main__":
    # import할 때는 정의만 읽고, 파일을 직접 실행할 때만 검사한다.
    run_checks()
