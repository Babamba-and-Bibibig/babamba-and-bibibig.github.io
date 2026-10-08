// Learn fees for four deliveries with gradient descent and no external packages.
// Each row is [1 for the base fee, additional distance (km), special packaging (no=0, yes=1)].
// Fees are in thousand won. Save this as main.rs and run it in a Rust project.
fn main() {
    let xb: [[f64; 3]; 4] = [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let y: [f64; 4] = [1.0, 3.0, 4.0, 6.0];
    let mut theta = [0.0_f64; 3]; // [base fee b, fee per km w_u, special packaging fee w_v]
    let lr = 0.2;
    let n = y.len() as f64;

    for _ in 0..2000 {
        let mut gradient = [0.0_f64; 3];
        for i in 0..y.len() {
            // Add the base fee, distance fee, and packaging fee for one delivery.
            let mut prediction = 0.0;
            for j in 0..theta.len() {
                prediction += xb[i][j] * theta[j];
            }
            let error = prediction - y[i];
            for j in 0..theta.len() {
                gradient[j] += xb[i][j] * error / n;
            }
        }
        // Calculate derivatives for all deliveries at the old coefficients, then update all three fees.
        for j in 0..theta.len() {
            theta[j] -= lr * gradient[j];
        }
    }

    let expected = [1.0, 2.0, 3.0];
    for j in 0..theta.len() {
        assert!((theta[j] - expected[j]).abs() < 1e-8);
    }
    // Predict a delivery with 3 additional km and special packaging.
    let new_input = [1.0, 3.0, 1.0];
    let mut new_prediction = 0.0;
    for j in 0..theta.len() {
        new_prediction += new_input[j] * theta[j];
    }
    assert!((new_prediction - 10.0).abs() < 1e-8);
    println!("Coefficients [b, w_u, w_v]: {:?}", theta);
    println!("Predicted delivery fee (thousand won): {:.6}", new_prediction);
}
