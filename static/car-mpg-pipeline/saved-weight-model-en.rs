// A standalone example that reads stored model state to make a prediction.
// The saved values below are arbitrary numbers, not actual training results.
// origin processing and Ridge fitting are excluded. Weight is measured in lb.
struct WeightModel {
    mean: f64,  // The mean weight stored during training.
    scale: f64, // The transformation divisor stored during training.
    w: f64,     // The coefficient multiplying transformed weight.
    b: f64,     // The intercept added at the end of the prediction equation.
}

impl WeightModel {
    // &self is a borrowed reference for reading the model. Stored values are unchanged.
    // weight is the new car's weight; the return value is predicted fuel economy.
    fn predict(&self, weight: f64) -> f64 {
        assert!(self.scale.is_finite() && self.scale > 0.0);
        let weight_scaled = (weight - self.mean) / self.scale;
        self.w * weight_scaled + self.b
    }
}

fn main() {
    // Create the object with the same arbitrary saved values as the final Python example.
    let model = WeightModel {
        mean: 3000.0,
        scale: 500.0,
        w: -4.0,
        b: 25.0,
    };
    let prediction = model.predict(2800.0);
    // Allow a small floating-point representation error when comparing with the expected 26.6.
    assert!((prediction - 26.6).abs() < 1e-12);
    println!("Prediction checked with arbitrary saved values (mpg): {prediction:.1}");
}
