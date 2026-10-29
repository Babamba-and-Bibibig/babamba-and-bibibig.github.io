// 첫 번째 시퀀스의 앞 두 시점을 표준 Rust로 검산한다.
fn main() {
    // 현재 입력용 가중치: 뉴런 3개 × 특징 4개
    let w_ih: [[f64; 4]; 3] = [
        [ 0.11, -0.07,  0.04,  0.02],
        [-0.09,  0.13,  0.06, -0.03],
        [ 0.05,  0.08, -0.12,  0.15],
    ];

    // 이전 은닉 상태용 가중치: 현재 뉴런 3개 × 이전 은닉값 3개
    let w_hh: [[f64; 3]; 3] = [
        [ 0.21, -0.14,  0.07],
        [-0.16,  0.25,  0.12],
        [ 0.19, -0.06,  0.31],
    ];

    let b_ih = [0.015, -0.025, 0.035];
    let b_hh = [-0.008, 0.018, -0.028];

    // 첫 번째 시퀀스의 앞 두 시점
    let inputs: [[f64; 4]; 2] = [
        [2.0, 5.0, 7.0, 11.0],
        [3.0, 8.0, 12.0, 6.0],
    ];

    // mut: 다음 시점으로 넘어갈 때 새 상태를 대입할 수 있게 한다.
    let mut state = [0.0_f64; 3];

    // iter(): 각 시점의 배열을 참조로 차례대로 읽는다.
    // enumerate(): 0부터 시작하는 순서 번호를 함께 제공한다.
    for (t, xt) in inputs.iter().enumerate() {
        let mut next_state = [0.0_f64; 3];

        // 코드에서는 반복문으로 계산하지만,
        // 뉴런들은 서로의 현재 결과를 입력으로 사용하지 않는다.
        for neuron in 0..3 {
            let mut input_sum = 0.0;
            let mut previous_sum = 0.0;

            for feature in 0..4 {
                input_sum += xt[feature] * w_ih[neuron][feature];
            }

            for previous_neuron in 0..3 {
                previous_sum +=
                    state[previous_neuron] * w_hh[neuron][previous_neuron];
            }

            next_state[neuron] = (
                input_sum
                + previous_sum
                + b_ih[neuron]
                + b_hh[neuron]
            ).tanh();
        }

        // 모든 뉴런이 같은 이전 state를 사용한 뒤,
        // 완성된 새 상태 전체로 교체한다.
        state = next_state;

        // {:?}: 배열 등을 확인하는 디버그 출력 형식
        println!("시점 {}: {:?}", t + 1, state);
    }
}
