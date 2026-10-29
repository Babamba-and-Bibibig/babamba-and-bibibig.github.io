// Reproduce the first two steps of the first sequence using standard Rust.
fn main() {
    // Input weights: 3 current neurons x 4 input features.
    let w_ih: [[f64; 4]; 3] = [
        [ 0.11, -0.07,  0.04,  0.02],
        [-0.09,  0.13,  0.06, -0.03],
        [ 0.05,  0.08, -0.12,  0.15],
    ];

    // Recurrent weights: 3 current neurons x 3 previous hidden values.
    let w_hh: [[f64; 3]; 3] = [
        [ 0.21, -0.14,  0.07],
        [-0.16,  0.25,  0.12],
        [ 0.19, -0.06,  0.31],
    ];

    let b_ih = [0.015, -0.025, 0.035];
    let b_hh = [-0.008, 0.018, -0.028];

    // The first two time steps of the first sequence.
    let inputs: [[f64; 4]; 2] = [
        [2.0, 5.0, 7.0, 11.0],
        [3.0, 8.0, 12.0, 6.0],
    ];

    // mut allows a new state to be assigned for the next time step.
    let mut state = [0.0_f64; 3];

    // iter(): read each time-step array by reference, in order.
    // enumerate(): also provide its index, starting from 0.
    for (t, xt) in inputs.iter().enumerate() {
        let mut next_state = [0.0_f64; 3];

        // This code uses a loop to compute the neurons, but each neuron
        // reads the previous state, never another neuron's current result.
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

        // After every neuron has used the same previous state,
        // replace it with the complete new state.
        state = next_state;

        // {:?}: debug formatting, which can display an array.
        println!("Time step {}: {:?}", t + 1, state);
    }
}
