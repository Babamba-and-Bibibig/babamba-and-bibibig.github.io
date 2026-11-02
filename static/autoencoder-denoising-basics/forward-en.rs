// Autoencoder forward pass with the Rust standard library.
// This program does NOT calculate gradients or update weights.
// Compile with: rustc --edition=2021 forward-en.rs -o ae-forward
// Run with: ./ae-forward

// [f64; 2] is a fixed-size array holding two 64-bit floating-point numbers.
// The return tuple contains z, two reconstructed components, and scalar MSE.
fn tiny_ae(x: [f64; 2], e: [f64; 2], d: [f64; 2]) -> (f64, [f64; 2], f64) {
    // Indices 0 and 1 select features of ONE sample.
    let z = e[0] * x[0] + e[1] * x[1];

    // Each decoder weight receives the same latent scalar z.
    let reconstruction = [d[0] * z, d[1] * z];

    // powi(2) squares each difference. Divide their sum by two components.
    let mse = ((reconstruction[0] - x[0]).powi(2)
        + (reconstruction[1] - x[1]).powi(2))
        / 2.0;

    (z, reconstruction, mse)
}

fn assert_close(actual: f64, expected: f64) {
    // Floating-point calculations can differ by tiny rounding amounts.
    assert!((actual - expected).abs() < 1e-12);
}

fn main() {
    // A uses the initial weights in the article's one-update example.
    // Only its FORWARD pass is computed here.
    let (z_a, r_a, loss_a) = tiny_ae([1.0, 2.0], [0.5, 0.5], [0.8, 1.2]);
    assert_close(z_a, 1.5);
    assert_close(r_a[0], 1.2);
    assert_close(r_a[1], 1.8);
    assert_close(loss_a, 0.04);
    println!("A: z={z_a:.3}, reconstruction={r_a:?}, MSE={loss_a:.7}");

    // Q1 and Q2 use the SAME sample B but separate encoder/decoder weights.
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

    // Tuple destructuring binds a name to each field in one condition.
    for (name, e, d, expected_z, expected_r, expected_loss) in conditions {
        let (z, reconstruction, mse) = tiny_ae(x_b, e, d);
        assert_close(z, expected_z);
        assert_close(reconstruction[0], expected_r[0]);
        assert_close(reconstruction[1], expected_r[1]);
        assert_close(mse, expected_loss);

        // .3/.7 control displayed decimals, not the stored numeric values.
        println!("{name}: z={z:.3}, reconstruction={reconstruction:?}, MSE={mse:.7}");
    }
}
