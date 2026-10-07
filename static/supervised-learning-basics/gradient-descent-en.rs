// Compute gradient descent on the same four-row dataset without external packages.
// Save this file as main.rs in a Rust binary project to run it.
fn main() {
    // Each row is [1 for the intercept, x1, x2]. f64 is a 64-bit floating-point type.
    let xb: [[f64; 3]; 4] = [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let y: [f64; 4] = [1.0, 3.0, 4.0, 6.0];
    let mut theta = [0.0_f64; 3]; // mut: a variable that can be updated, [b, w1, w2]
    let lr = 0.2;
    let n = y.len() as f64;       // Convert the array length for floating-point calculations

    for _ in 0..2000 {           // The upper bound 2000 is excluded, giving 2000 iterations
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
        // Compute gradients for all rows at the existing coefficients, then update them together.
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
    println!("Coefficients [b, w1, w2]: {:?}", theta);
    println!("Prediction for the new input: {:.6}", new_prediction);
}
