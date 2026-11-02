// Rust 표준 라이브러리로 계산하는 오토인코더 순전파.
// 이 프로그램은 기울기 계산이나 가중치 갱신을 수행하지 않는다.
// 컴파일: rustc --edition=2021 forward-ko.rs -o ae-forward
// 실행: ./ae-forward

// [f64; 2]는 64비트 실수 두 개를 담는 고정 길이 배열이다.
// 반환 튜플에는 z, 복원 성분 두 개의 배열, MSE 스칼라가 들어 있다.
fn tiny_ae(x: [f64; 2], e: [f64; 2], d: [f64; 2]) -> (f64, [f64; 2], f64) {
    // 인덱스 0과 1은 샘플 하나 안의 첫 번째·두 번째 특징을 선택한다.
    let z = e[0] * x[0] + e[1] * x[1];

    // 디코더의 각 가중치는 같은 잠재 스칼라 z에 곱해진다.
    let reconstruction = [d[0] * z, d[1] * z];

    // powi(2)는 차이를 제곱한다. 두 제곱오차의 합을 성분 수 2로 나눈다.
    let mse = ((reconstruction[0] - x[0]).powi(2)
        + (reconstruction[1] - x[1]).powi(2))
        / 2.0;

    (z, reconstruction, mse)
}

fn assert_close(actual: f64, expected: f64) {
    // 부동소수점 계산에는 작은 반올림 차이가 있으므로 허용 오차로 비교한다.
    assert!((actual - expected).abs() < 1e-12);
}

fn main() {
    // A에는 본문의 한 번 갱신 예제에서 사용한 초기 가중치를 적용한다.
    // 이 코드에서는 순전파만 계산한다.
    let (z_a, r_a, loss_a) = tiny_ae([1.0, 2.0], [0.5, 0.5], [0.8, 1.2]);
    assert_close(z_a, 1.5);
    assert_close(r_a[0], 1.2);
    assert_close(r_a[1], 1.8);
    assert_close(loss_a, 0.04);
    println!("A: z={z_a:.3}, reconstruction={r_a:?}, MSE={loss_a:.7}");

    // Q1·Q2는 같은 샘플 B에 서로 다른 인코더·디코더 가중치를 적용한다.
    let x_b = [2.0, 1.0];
    let conditions = [
        ("Q1", [0.2, 0.4], [1.0, 1.5], 0.8, [0.8, 1.2], 0.74),
        (
            "Q2",
            [0.38, 0.49],
            [1.096, 1.484],
            1.25,
            [1.37, 1.855],
            0.5639625,
        ),
    ];

    // 튜플 풀기는 조건 하나에 묶인 각 값을 개별 변수에 담는다.
    for (name, e, d, expected_z, expected_r, expected_loss) in conditions {
        let (z, reconstruction, mse) = tiny_ae(x_b, e, d);
        assert_close(z, expected_z);
        assert_close(reconstruction[0], expected_r[0]);
        assert_close(reconstruction[1], expected_r[1]);
        assert_close(mse, expected_loss);

        // .3과 .7은 소수점 아래 표시 자릿수다. 저장된 실제 숫자는 바뀌지 않는다.
        println!("{name}: z={z:.3}, reconstruction={reconstruction:?}, MSE={mse:.7}");
    }
}
