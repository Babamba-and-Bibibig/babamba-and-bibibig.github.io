// 외부 패키지 없이 네 배송 건의 요금을 경사하강법으로 학습한다.
// 각 행은 [기본요금용 1, 추가 거리(km), 특수 포장 여부(없음 0, 있음 1)]이다.
// 배송비의 단위는 천 원이다. 파일을 main.rs로 저장한 Rust 프로젝트에서 실행한다.
fn main() {
    let xb: [[f64; 3]; 4] = [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let y: [f64; 4] = [1.0, 3.0, 4.0, 6.0];
    let mut theta = [0.0_f64; 3]; // [기본요금 b, 거리당 요금 w_u, 특수 포장 요금 w_v]
    let lr = 0.2;
    let n = y.len() as f64;

    for _ in 0..2000 {
        let mut gradient = [0.0_f64; 3];
        for i in 0..y.len() {
            // 한 배송 건의 기본요금, 거리 요금, 포장 요금을 더한다.
            let mut prediction = 0.0;
            for j in 0..theta.len() {
                prediction += xb[i][j] * theta[j];
            }
            let error = prediction - y[i];
            for j in 0..theta.len() {
                gradient[j] += xb[i][j] * error / n;
            }
        }
        // 모든 배송 건의 미분값을 기존 계수에서 계산한 뒤 요금 세 개를 함께 고친다.
        for j in 0..theta.len() {
            theta[j] -= lr * gradient[j];
        }
    }

    let expected = [1.0, 2.0, 3.0];
    for j in 0..theta.len() {
        assert!((theta[j] - expected[j]).abs() < 1e-8);
    }
    // 추가 거리 3km에 특수 포장을 쓰는 배송을 예측한다.
    let new_input = [1.0, 3.0, 1.0];
    let mut new_prediction = 0.0;
    for j in 0..theta.len() {
        new_prediction += new_input[j] * theta[j];
    }
    assert!((new_prediction - 10.0).abs() < 1e-8);
    println!("계수 [b, w_u, w_v]: {:?}", theta);
    println!("배송비 예측(천 원): {:.6}", new_prediction);
}
