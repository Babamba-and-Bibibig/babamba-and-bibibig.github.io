// 외부 패키지 없이 같은 4행 데이터의 경사하강법을 계산한다.
// 파일을 main.rs로 저장한 Rust 실행 프로젝트에서 실행할 수 있다.
fn main() {
    // 각 행은 [절편용 1, x1, x2]. f64는 64비트 실수 형식이다.
    let xb: [[f64; 3]; 4] = [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let y: [f64; 4] = [1.0, 3.0, 4.0, 6.0];
    let mut theta = [0.0_f64; 3]; // mut: 갱신 가능한 변수, [b, w1, w2]
    let lr = 0.2;
    let n = y.len() as f64;       // 배열 길이를 실수 계산용으로 변환

    for _ in 0..2000 {           // 끝의 2000은 포함하지 않아 총 2000회
        let mut gradient = [0.0_f64; 3];
        for i in 0..y.len() {
            let mut prediction = 0.0;
            for j in 0..theta.len() {
                prediction += xb[i][j] * theta[j];
            }
            let error = prediction - y[i];
            for j in 0..theta.len() {
                gradient[j] += xb[i][j] * error / n;
            }
        }
        // 모든 행의 기울기를 기존 계수에서 계산한 후 한 번에 갱신한다.
        for j in 0..theta.len() {
            theta[j] -= lr * gradient[j];
        }
    }

    let expected = [1.0, 2.0, 3.0];
    for j in 0..theta.len() {
        assert!((theta[j] - expected[j]).abs() < 1e-8);
    }
    let new_input = [1.0, 3.0, 4.0];
    let mut new_prediction = 0.0;
    for j in 0..theta.len() {
        new_prediction += new_input[j] * theta[j];
    }
    println!("계수 [b, w1, w2]: {:?}", theta);
    println!("새 입력의 예측: {:.6}", new_prediction);
}
