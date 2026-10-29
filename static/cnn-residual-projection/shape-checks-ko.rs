// CNN 잔차 연결의 조건과 공간 크기 계산만 확인한다.
// 본문: https://babamba-and-bibibig.github.io/cnn-residual-projection/

// 이 글의 주 경로: conv1은 K=3/P=1, conv2는 K=3/P=1/S=1.
fn use_identity(in_channels: usize, out_channels: usize, stride: usize) -> bool {
    in_channels == out_channels && stride == 1
}

// dilation=1, 양쪽에 같은 정수 패딩을 사용하는 한 축의 출력 크기다.
fn output_size(input: usize, kernel: usize, padding: usize, stride: usize) -> usize {
    assert!(input > 0 && kernel > 0 && stride > 0);
    let padded = input + 2 * padding;

    // 커널이 입력에 들어가는지 검사한 뒤 부호 없는 정수를 뺀다.
    assert!(padded >= kernel);

    // 분자가 0 이상이므로 정수 나눗셈의 몫이 floor와 같다.
    (padded - kernel) / stride + 1
}

fn main() {
    assert!(use_identity(8, 8, 1));
    assert!(!use_identity(8, 16, 1));  // !는 참·거짓을 뒤집는다.
    assert!(!use_identity(8, 8, 2));

    // 두 커널·패딩 조합이 짝수와 홀수 입력 모두에서 같은 크기를 만든다.
    for input in [16_usize, 15, 13] {
        let main_size = output_size(input, 3, 1, 2);
        let skip_size = output_size(input, 1, 0, 2);
        assert_eq!(main_size, skip_size);
        println!("입력 {input}: 주 경로 {main_size}, Shortcut {skip_size}");
    }

    assert_eq!(output_size(16, 1, 1, 2), 9);  // 패딩을 1로 바꾼 경우
    assert_eq!(output_size(16, 3, 0, 2), 7);  // 3×3에서 패딩을 뺀 경우
    println!("Identity 조건과 출력 크기 확인 완료");
}
