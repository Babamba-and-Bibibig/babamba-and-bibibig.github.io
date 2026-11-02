// Rust 표준 라이브러리로 입력창과 RNN 순전파·스칼라 역전파를 확인한다.
// 자동미분이나 학습 옵티마이저를 제공하는 구현은 아니다.

fn recurrent_step(
    input: &[f64; 4],          // 한 시점의 현재 입력 특징 F=4개.
    previous: &[f64; 3],       // 이전 시점의 은닉값 H=3개.
    weight_ih: &[[f64; 4]; 3], // 현재 뉴런별 입력 가중치: (H,F).
    weight_hh: &[[f64; 3]; 3], // 현재 뉴런별 이전 상태 가중치: (H,H).
    bias_ih: &[f64; 3],        // 입력 계산용 편향 H개.
    bias_hh: &[f64; 3],        // 이전 상태 계산용 편향 H개.
) -> [f64; 3] {
    let mut next = [0.0; 3];
    for j in 0..3 { // 현재 은닉 뉴런 인덱스 0,1,2. 끝값 3은 제외한다.
        let input_sum: f64 = (0..4).map(|k| weight_ih[j][k] * input[k]).sum();
        let state_sum: f64 = (0..3).map(|q| weight_hh[j][q] * previous[q]).sum();
        next[j] = (input_sum + state_sum + bias_ih[j] + bias_hh[j]).tanh();
    }
    // 모든 뉴런이 같은 previous를 읽는다. 계산 중 previous를 덮어쓰지 않는다.
    next
}

fn scalar_loss(recurrent_weight: f64) -> f64 {
    let mut state = 0.0_f64;
    for value in [1.0, 2.0, 3.0] {
        state = (0.5 * value + recurrent_weight * state).tanh();
    }
    0.5 * (state - 0.5).powi(2)
}

fn main() {
    let visits = [10, 20, 30, 40, 50, 60, 70];
    let seq_len = 3;
    // 끝에 다음 날 정답이 없는 창을 제외하므로 len - seq_len개다.
    for start in 0..visits.len() - seq_len {
        let input = &visits[start..start + seq_len]; // 끝 인덱스 제외.
        let target = visits[start + seq_len];
        println!("window={input:?}, target={target}");
    }

    // 별도의 다변량 구조 검산: B=2,T=5,F=4,H=3. 실제 관측값은 아니다.
    let samples = [
        [[0.1,0.2,0.3,0.4],[0.2,0.3,0.4,0.5],[0.3,0.4,0.5,0.6],
         [0.4,0.5,0.6,0.7],[0.5,0.6,0.7,0.8]],
        [[0.8,0.7,0.6,0.5],[0.7,0.6,0.5,0.4],[0.6,0.5,0.4,0.3],
         [0.5,0.4,0.3,0.2],[0.4,0.3,0.2,0.1]],
    ];
    let weight_ih = [[0.2,-0.1,0.3,0.1],[-0.2,0.4,0.1,-0.3],[0.3,0.1,-0.2,0.2]];
    let weight_hh = [[0.1,0.2,-0.1],[-0.2,0.1,0.3],[0.2,-0.1,0.1]];
    let bias_ih = [0.01,-0.02,0.03];
    let bias_hh = [0.02,0.01,-0.01];
    for (sample_index, sequence) in samples.iter().enumerate() {
        let mut state = [0.0_f64; 3]; // 각 샘플은 자신의 초기 상태로 시작한다.
        for input in sequence {
            state = recurrent_step(input, &state, &weight_ih, &weight_hh, &bias_ih, &bias_hh);
        }
        println!("sample={sample_index}, last_hidden={state:?}");
    }

    // 여기부터는 F=H=B=1인 특수 검산이다. 본문의 일반 구조와 구분한다.
    let inputs = [1.0_f64, 2.0, 3.0];
    let input_weight = 0.5_f64;
    let recurrent_weight = 0.5_f64;
    let target = 0.5_f64;
    let mut hs = [0.0_f64; 4]; // h0를 포함해 시점 3개의 결과를 저장한다.
    for (index, &value) in inputs.iter().enumerate() {
        hs[index + 1] = (input_weight * value + recurrent_weight * hs[index]).tanh();
    }
    let mut adj_h = hs[3] - target; // L=0.5*(h3-target)^2의 dL/dh3.
    let mut pieces = [0.0; 3];
    for t in (1..=3).rev() { // 1..=3은 양끝 포함, rev는 3,2,1로 뒤집는다.
        let adj_a = adj_h * (1.0 - hs[t].powi(2));
        pieces[t - 1] = adj_a * hs[t - 1]; // 해당 시점의 W_hh 사용분.
        adj_h = adj_a * recurrent_weight; // 이전 은닉값으로 전달할 기울기.
    }
    let gradient: f64 = pieces.iter().sum();
    let epsilon = 1e-6;
    let finite_difference = (scalar_loss(recurrent_weight + epsilon)
        - scalar_loss(recurrent_weight - epsilon)) / (2.0 * epsilon);
    assert!((gradient - finite_difference).abs() < 1e-8);
    println!("hidden_history={hs:?}");
    println!("time_pieces={pieces:?}");
    println!("gradient={gradient:.15}, finite_difference={finite_difference:.15}");

    // 별도 plain SGD 한 스텝: 위 검산용 가중치는 두고 새 변수만 수정한다.
    let mut updated_weight = recurrent_weight;
    let learning_rate = 0.01; // 가중치 이동량에 곱하는 학습률.
    updated_weight -= learning_rate * gradient; // 이 줄이 실제 숫자를 갱신한다.
    println!("plain_sgd old={recurrent_weight:.15}, updated={updated_weight:.15}");
}
