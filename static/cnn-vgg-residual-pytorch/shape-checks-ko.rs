// CNN 개수와 한 위치의 잔차 계산을 Rust 표준 기능으로 확인한다.
// 이 파일은 PyTorch 모델이나 자동 미분을 구현하지 않는다.
// 별도 크레이트 없이 Rust 컴파일러로 실행할 수 있다.
//
// 컴파일: rustc shape-checks-ko.rs -o shape-checks-ko
// 실행:   ./shape-checks-ko

fn relu(x: f32) -> f32 {
    // 이 예제의 유한한 실수에서 음수이면 0, 아니면 입력값을 반환한다.
    // f32는 32비트 부동소수점 숫자 타입이다.
    x.max(0.0)
}

fn main() {
    // 괄호가 없으므로 함수 자체를 함수 포인터에 저장한다.
    // fn(f32) -> f32는 f32 하나를 받아 f32 하나를 반환하는 함수 타입이다.
    let relu_function: fn(f32) -> f32 = relu;

    // 괄호를 붙여 호출하면 함수가 계산한 숫자가 result에 저장된다.
    let result = relu_function(-2.0);
    assert_eq!(result, 0.0);

    // usize는 크기와 인덱스 등에 사용하는 부호 없는 정수 타입이다.
    // Conv2d(4, 6, 3), groups=1의 전체 가중치 수. 편향은 제외한다.
    let input_channels: usize = 4;
    let output_filters: usize = 6;
    let weights = output_filters * input_channels * 3 * 3;
    assert_eq!(weights, 216);
    println!("합성곱 가중치: {weights}");

    // 1×1은 공간 크기이며 입력 채널과 출력 필터의 개수가 사라지지 않는다.
    // Conv2d(16, 32, 1)의 가중치는 512개, 기본 편향은 별도로 32개다.
    let projection_weights: usize = 32 * 16 * 1 * 1;
    assert_eq!(projection_weights, 512);
    println!("1×1 shortcut 가중치: {projection_weights}");

    // 두 블록의 2×2, stride=2 MaxPool이 높이와 너비를 두 번 줄인다.
    // 이 예제에서는 32가 2로 정확히 나뉘므로 32 → 16 → 8이다.
    let height: usize = 32 / 2 / 2;
    let width: usize = 32 / 2 / 2;
    let in_features: usize = 16 * height * width;
    assert_eq!(in_features, 1024);

    // 이미지 개수가 바뀌어도 이미지 한 장의 특징값은 1024개다.
    // 이 배열들은 shape의 숫자를 기록할 뿐 실제 텐서를 계산하지 않는다.
    for batch_size in [2_usize, 1_usize] {
        let flat_shape = [batch_size, in_features];
        let score_shape = [batch_size, 5];
        println!("Flatten: {flat_shape:?}, 점수: {score_shape:?}");
    }

    // 같은 위치의 원래 입력과 주 경로 값을 더한 뒤 ReLU를 적용한다.
    let original: f32 = 2.0;
    let transformed: f32 = -1.0;
    let residual_result = relu(transformed + original);
    assert_eq!(residual_result, 1.0);
    println!("잔차 연결의 한 위치: {residual_result}");
}
