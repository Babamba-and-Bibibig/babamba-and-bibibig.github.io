"""RNN으로 다음 날 방문자 수 예측하기: CPU 실행용 전체 예제.

PyTorch가 설치된 환경에서 python examples-ko.py로 실행한다.
가상 데이터 실험이며, 실제 서비스의 예측 성능을 입증하지 않는다.
"""

import argparse
import copy
import json
import platform
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# nn은 torch.nn의 짧은 이름이다. nn.Module은 신경망의 기본 클래스다.
# TensorDataset은 같은 위치의 X와 y를 한 샘플로 묶는다.
# DataLoader는 그 샘플들을 미니배치로 묶어 전달한다.
VALUE_SCALE = 100.0  # 방문자 수를 나누는 고정 배율. 확률 변환이 아니다.
SEQ_LEN = 14        # 한 입력창의 시점 수 T.
HIDDEN_SIZE = 4     # 개발자가 선택한 은닉값 수 H. 14에서 계산한 수가 아니다.
DATA_SEED = 1       # 가상 데이터의 난수 시작 설정.
TRAIN_SEED = 7      # 모델 초기화와 학습 샘플 순서를 재현하기 위한 설정.


def make_windows(values: torch.Tensor, seq_len: int):
    """1차원 기록을 입력 (N,T,1), 다음 날 정답 (N,1)로 만든다."""
    if values.ndim != 1 or not 1 <= seq_len < len(values):
        raise ValueError("1차원 기록과, 다음 날 정답이 남는 양의 창 길이가 필요합니다.")
    windows = []
    # range(n)의 끝값 n은 제외한다. i는 입력창의 시작 인덱스다.
    for i in range(len(values) - seq_len):
        windows.append(values[i:i + seq_len])  # 끝 인덱스는 제외한다.
    # stack은 새 샘플 축을 만든다. unsqueeze는 특징 축을 추가한다.
    X = torch.stack(windows, dim=0).unsqueeze(-1)
    y = values[seq_len:].unsqueeze(-1)  # 각 입력창 바로 다음 값.
    return X, y


def make_data():
    torch.manual_seed(DATA_SEED)
    weekly = torch.tensor(
        [62, 55, 58, 60, 75, 118, 105], dtype=torch.float32,
    )
    # 7개짜리 묶음 전체를 12번 반복한다. 84일의 가상 기록이다.
    repeated = weekly.repeat(12)
    # randn은 평균 0·표준편차 1인 분포에서 뽑는다. 3은 변동 배율이다.
    visitors = (repeated + torch.randn(len(repeated)) * 3.0) / VALUE_SCALE
    X, y = make_windows(visitors, SEQ_LEN)
    # 정답이 원본 기록에서 차지하는 인덱스: 14부터 83까지.
    target_positions = torch.arange(SEQ_LEN, len(visitors))
    return visitors, X, y, target_positions


class VisitorRNN(nn.Module):
    """매번 0 상태에서 시작하는 한 층·단방향 RNN과 예측층."""

    def __init__(self, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,           # 시점당 관측값은 방문자 수 하나다.
            hidden_size=hidden_size,
            num_layers=1,           # RNN 층 하나.
            nonlinearity="tanh",    # 가중합의 각 원소에 tanh를 적용한다.
            batch_first=True,       # 입력과 output의 축 순서는 (B,T,특징).
        )
        self.head = nn.Linear(
            in_features=hidden_size,  # 마지막 상태의 H개 값을 받는다.
            out_features=1,           # 샘플마다 다음 날 값 하나를 만든다.
        )

    def forward(self, x):
        # x: (B,T,1). h0를 생략하면 각 호출은 초기 상태 0에서 시작한다.
        output, _ = self.rnn(x)  # output: (B,T,H). _는 사용하지 않는 h_n.
        last = output[:, -1, :]  # 모든 샘플의 마지막 시점: (B,H).
        return self.head(last)   # 최종 예측: (B,1).


def verify_shapes_and_state(visitors, X, y):
    """창, 시간 반복, 개별/배치 호출이 의도한 계산인지 확인한다."""
    demo = torch.tensor([10., 20., 30., 40., 50., 60., 70.])
    demo_X, demo_y = make_windows(demo, 3)
    assert demo_X[:, :, 0].tolist() == [
        [10., 20., 30.], [20., 30., 40.],
        [30., 40., 50.], [40., 50., 60.],
    ]
    assert demo_y[:, 0].tolist() == [40., 50., 60., 70.]
    assert X.shape == (70, 14, 1) and y.shape == (70, 1)
    assert torch.equal(X[0, :, 0], visitors[0:14])
    assert torch.equal(y[0, 0], visitors[14])
    assert torch.equal(X[69, :, 0], visitors[69:83])
    assert torch.equal(y[69, 0], visitors[83])

    torch.manual_seed(TRAIN_SEED)
    rnn = nn.RNN(1, HIDDEN_SIZE, batch_first=True)
    cell = nn.RNNCell(1, HIDDEN_SIZE)
    # 별도 생성한 객체의 난수 가중치는 다르다. 비교를 위해 값을 맞춘다.
    with torch.no_grad():
        cell.weight_ih.copy_(rnn.weight_ih_l0)
        cell.weight_hh.copy_(rnn.weight_hh_l0)
        cell.bias_ih.copy_(rnn.bias_ih_l0)
        cell.bias_hh.copy_(rnn.bias_hh_l0)
    before = {name: p.detach().clone() for name, p in cell.named_parameters()}
    xb = X[:2]  # B=2, T=14, F=1. 전체 데이터 N=70과 구분한다.
    state = xb.new_zeros(2, HIDDEN_SIZE)  # 입력과 같은 dtype/device, (B,H).
    history = []
    # unbind(dim=1)은 시간축을 제거한 (B,F) 텐서를 14개 반환한다.
    for xt in xb.unbind(dim=1):
        state = cell(xt, state)  # 새 상태를 계산한다. 가중치 갱신이 아니다.
        history.append(state)   # 상태를 detach하지 않아 시간 연결을 유지한다.
    cell_output = torch.stack(history, dim=1)  # (2,14,4).
    output, h_n = rnn(xb)
    assert output.shape == (2, 14, 4) and h_n.shape == (1, 2, 4)
    # allclose는 실수 연산의 작은 차이를 허용한다.
    assert torch.allclose(cell_output, output, atol=1e-6, rtol=1e-6)
    assert torch.allclose(output[:, -1, :], h_n[0])
    assert all(torch.equal(before[n], p) for n, p in cell.named_parameters())

    model = VisitorRNN()
    model.eval()  # 평가 동작으로 전환한다. 자동미분을 끄는 기능은 아니다.
    with torch.no_grad():  # 이 블록에서 역전파용 연산 기록을 만들지 않는다.
        together = model(X)
        individual = torch.cat(
            [model(row.unsqueeze(0)) for row in X], dim=0,
        )
    assert torch.allclose(individual, together, atol=1e-6, rtol=1e-6)

    # 입력 특징이 여러 개인 일반 구조도 확인한다: B2,T5,F4,H3.
    multi = torch.randn(2, 5, 4)
    multi_rnn = nn.RNN(4, 3, batch_first=True)
    multi_output, multi_hn = multi_rnn(multi)
    assert multi_output.shape == (2, 5, 3) and multi_hn.shape == (1, 2, 3)
    return {
        "X_shape": list(X.shape), "y_shape": list(y.shape),
        "demo_windows": demo_X[:, :, 0].tolist(),
        "demo_targets": demo_y[:, 0].tolist(),
        "output_shape": list(output.shape), "h_n_shape": list(h_n.shape),
        "cell_rnn_max_difference": (cell_output - output).abs().max().item(),
        "individual_batch_max_difference": (individual - together).abs().max().item(),
        "state_loop_keeps_weights": True,
        "multifeature_output_shape": list(multi_output.shape),
    }


def verify_learning_step(X, y):
    """순전파·역전파·갱신의 역할을 별도 모델로 확인한다."""
    torch.manual_seed(TRAIN_SEED)
    model = VisitorRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)  # 가중치가 아니라 이전 .grad를 비운다.
    prediction = model(X[:2])
    loss = nn.MSELoss()(prediction, y[:2])  # 두 샘플 제곱오차의 평균.
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    loss.backward()  # 시간 연결 전체를 거쳐 가중치별 기울기를 계산한다.
    assert all(p.grad is not None and p.grad.shape == p.shape for p in model.parameters())
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    recurrent_gradient_shape = list(model.rnn.weight_hh_l0.grad.shape)
    optimizer.step()  # 준비된 기울기를 이용해 가중치를 수정한다.
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters())
    return {
        "forward_keeps_weights": True, "backward_keeps_weights": True,
        "step_changes_weights": True,
        "recurrent_gradient_shape": recurrent_gradient_shape,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }


def train_and_evaluate(visitors, X, y, target_positions):
    train_mask = target_positions < 56
    valid_mask = (target_positions >= 56) & (target_positions < 70)
    test_mask = target_positions >= 70
    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    assert len(X_train) == 42 and len(X_valid) == len(X_test) == 14

    # 완성된 창들의 순서만 섞는다. 창 내부의 날짜 순서는 유지된다.
    generator = torch.Generator().manual_seed(TRAIN_SEED)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=16,    # 한 호출에 최대 16개 창을 전달한다.
        shuffle=True,    # 서로 독립적으로 시작하는 학습창의 순서를 섞는다.
        drop_last=False, # 마지막 10개 샘플도 학습에 사용한다.
        generator=generator,  # 샘플 섞기용 난수 발생기를 별도로 둔다.
    )
    torch.manual_seed(TRAIN_SEED)
    device = torch.device("cpu")  # 이 글의 재현 결과는 CPU로 측정한다.
    model = VisitorRNN().to(device)
    loss_fn = nn.MSELoss()  # 기본 reduction='mean': 원소별 제곱오차의 평균.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_valid_loss = float("inf")
    best_state = None
    best_epoch = 0
    batch_sizes = []
    first_epoch_loss = None
    last_epoch_loss = None

    # 검증으로 저장할 시점을 선택한다. 테스트 데이터는 이 반복에 쓰지 않는다.
    for epoch in range(1, 301):  # 1부터 300까지, 301은 제외한다.
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            assert prediction.shape == yb.shape
            loss = loss_fn(prediction, yb)
            loss.backward()
            # 큰 기울기를 제한할 필요가 있다면 backward와 step 사이에 둔다.
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # 배치마다 평균이므로 샘플 수를 곱해 합계를 구한다.
            loss_sum += loss.item() * len(xb)
            sample_count += len(xb)
            if epoch == 1:
                batch_sizes.append(len(xb))
        train_loss = loss_sum / sample_count
        model.eval()
        with torch.no_grad():
            valid_loss = loss_fn(model(X_valid.to(device)), y_valid.to(device)).item()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            # state_dict를 독립 복사해야 이후 학습에 저장값이 따라 바뀌지 않는다.
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1:
            first_epoch_loss = train_loss
        last_epoch_loss = train_loss
        if epoch in (1, 50, 100, 150, 200, 250, 300):
            print(f"epoch={epoch:3d} train_mse={train_loss:.6f} valid_mse={valid_loss:.6f}")

    assert batch_sizes == [16, 16, 10]
    model.load_state_dict(best_state)  # 검증이 가장 좋았던 가중치로 되돌린다.
    model.eval()
    with torch.no_grad():
        test_prediction = model(X_test.to(device)).cpu()
        # 마지막 14일로 아직 정답이 없는 85일째를 예측한다.
        next_input = visitors[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        next_prediction = model(next_input.to(device)).item() * VALUE_SCALE
    test_positions = target_positions[test_mask]
    yesterday_prediction = visitors[test_positions - 1].unsqueeze(-1)
    weekly_prediction = visitors[test_positions - 7].unsqueeze(-1)

    def mae_in_visitors(prediction):
        # MAE: 오차의 절댓값 평균. /100을 되돌려 방문자 수 단위로 표현한다.
        return (prediction - y_test).abs().mean().item() * VALUE_SCALE

    return {
        "split_samples": {"train": 42, "validation": 14, "test": 14},
        "target_index_ranges": {"train": [14, 55], "validation": [56, 69], "test": [70, 83]},
        "batch_sizes": batch_sizes, "steps_per_epoch": len(train_loader),
        "epochs_run": 300, "optimizer_steps": 300 * len(train_loader),
        "best_epoch": best_epoch, "best_validation_mse": best_valid_loss,
        "first_epoch_training_mse": first_epoch_loss,
        "last_epoch_training_mse": last_epoch_loss,
        "test_mae_visitors": {
            "rnn": mae_in_visitors(test_prediction),
            "yesterday": mae_in_visitors(yesterday_prediction),
            "previous_week": mae_in_visitors(weekly_prediction),
        },
        "test_predictions": (test_prediction[:, 0] * VALUE_SCALE).tolist(),
        "test_targets": (y_test[:, 0] * VALUE_SCALE).tolist(),
        "day_85_prediction": next_prediction,
        "evaluation_protocol": "Rolling one-step-ahead with observed history; not a 14-day-ahead forecast.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="선택: 실행 결과 JSON 저장 경로")
    args = parser.parse_args()
    torch.set_num_threads(1)  # 작은 CPU 예제의 실행 비용과 변동을 줄인다.
    visitors, X, y, target_positions = make_data()
    results = {
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "device": "cpu", "dtype": "float32", "data_seed": DATA_SEED,
        "training_seed": TRAIN_SEED, "sequence_length": SEQ_LEN,
        "hidden_size": HIDDEN_SIZE,
        "shape_checks": verify_shapes_and_state(visitors, X, y),
        "learning_step_checks": verify_learning_step(X, y),
        "training": train_and_evaluate(visitors, X, y, target_positions),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.results is not None:
        args.results.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
