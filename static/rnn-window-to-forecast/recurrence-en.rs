// Use the Rust standard library to verify windows, RNN forward computation, and scalar BPTT.
// This implementation does not provide automatic differentiation or a training optimizer.

fn recurrent_step(
    input: &[f64; 4],          // F=4 current input features at one time step.
    previous: &[f64; 3],       // H=3 hidden values from the previous time step.
    weight_ih: &[[f64; 4]; 3], // Input weights for each current neuron: (H,F).
    weight_hh: &[[f64; 3]; 3], // Previous-state weights for each current neuron: (H,H).
    bias_ih: &[f64; 3],        // H input-side biases.
    bias_hh: &[f64; 3],        // H previous-state-side biases.
) -> [f64; 3] {
    let mut next = [0.0; 3];
    for j in 0..3 { // Current hidden neuron indices 0,1,2; the end value 3 is excluded.
        let input_sum: f64 = (0..4).map(|k| weight_ih[j][k] * input[k]).sum();
        let state_sum: f64 = (0..3).map(|q| weight_hh[j][q] * previous[q]).sum();
        next[j] = (input_sum + state_sum + bias_ih[j] + bias_hh[j]).tanh();
    }
    // Every neuron reads the same previous state; do not overwrite it during the calculation.
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
    // Exclude the last window without a next-day target, giving len - seq_len samples.
    for start in 0..visits.len() - seq_len {
        let input = &visits[start..start + seq_len]; // Exclude the stop index.
        let target = visits[start + seq_len];
        println!("window={input:?}, target={target}");
    }

    // Separate multifeature shape check: B=2,T=5,F=4,H=3; not real observations.
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
        let mut state = [0.0_f64; 3]; // Each sample begins with its own initial state.
        for input in sequence {
            state = recurrent_step(input, &state, &weight_ih, &weight_hh, &bias_ih, &bias_hh);
        }
        println!("sample={sample_index}, last_hidden={state:?}");
    }

    // From here, F=H=B=1 is a special scalar check, separate from the general structure.
    let inputs = [1.0_f64, 2.0, 3.0];
    let input_weight = 0.5_f64;
    let recurrent_weight = 0.5_f64;
    let target = 0.5_f64;
    let mut hs = [0.0_f64; 4]; // Store the three time-step outputs together with h0.
    for (index, &value) in inputs.iter().enumerate() {
        hs[index + 1] = (input_weight * value + recurrent_weight * hs[index]).tanh();
    }
    let mut adj_h = hs[3] - target; // dL/dh3 for L=0.5*(h3-target)^2.
    let mut pieces = [0.0; 3];
    for t in (1..=3).rev() { // 1..=3 includes both ends; rev changes the order to 3,2,1.
        let adj_a = adj_h * (1.0 - hs[t].powi(2));
        pieces[t - 1] = adj_a * hs[t - 1]; // W_hh term from this time step.
        adj_h = adj_a * recurrent_weight; // Loss gradient passed to the previous state.
    }
    let gradient: f64 = pieces.iter().sum();
    let epsilon = 1e-6;
    let finite_difference = (scalar_loss(recurrent_weight + epsilon)
        - scalar_loss(recurrent_weight - epsilon)) / (2.0 * epsilon);
    assert!((gradient - finite_difference).abs() < 1e-8);
    println!("hidden_history={hs:?}");
    println!("time_pieces={pieces:?}");
    println!("gradient={gradient:.15}, finite_difference={finite_difference:.15}");

    // Separate plain SGD step: keep the checked weight and update a new variable.
    let mut updated_weight = recurrent_weight;
    let learning_rate = 0.01; // Learning rate that scales the weight change.
    updated_weight -= learning_rate * gradient; // This line changes the actual value.
    println!("plain_sgd old={recurrent_weight:.15}, updated={updated_weight:.15}");
}
