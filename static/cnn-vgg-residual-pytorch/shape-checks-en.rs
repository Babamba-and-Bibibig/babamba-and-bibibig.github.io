// Check CNN counts and one residual output using Rust standard features.
// This file does not implement a PyTorch model or automatic differentiation.
// Compile it with the Rust compiler without adding any crates.
//
// Compile: rustc shape-checks-en.rs -o shape-checks-en
// Run:     ./shape-checks-en

fn relu(x: f32) -> f32 {
    // For the finite values in this example, return 0 for negatives, else x.
    // f32 is the 32-bit floating-point number type.
    x.max(0.0)
}

fn main() {
    // With no parentheses, store the function itself as a function pointer.
    // fn(f32) -> f32 describes a function taking one f32 and returning one f32.
    let relu_function: fn(f32) -> f32 = relu;

    // Adding parentheses calls the function and stores its returned number.
    let result = relu_function(-2.0);
    assert_eq!(result, 0.0);

    // usize is an unsigned integer type used for sizes and indices.
    // Count Conv2d(4, 6, 3) weights with groups=1, excluding biases.
    let input_channels: usize = 4;
    let output_filters: usize = 6;
    let weights = output_filters * input_channels * 3 * 3;
    assert_eq!(weights, 216);
    println!("Convolution weights: {weights}");

    // 1×1 is a spatial size; input channels and output filters still count.
    // Conv2d(16, 32, 1) has 512 weights plus 32 default biases.
    let projection_weights: usize = 32 * 16 * 1 * 1;
    assert_eq!(projection_weights, 512);
    println!("1×1 shortcut weights: {projection_weights}");

    // Two 2×2, stride=2 MaxPool layers reduce height and width twice.
    // In this example, 32 is exactly divisible by 2: 32 → 16 → 8.
    let height: usize = 32 / 2 / 2;
    let width: usize = 32 / 2 / 2;
    let in_features: usize = 16 * height * width;
    assert_eq!(in_features, 1024);

    // The feature count per image remains 1024 when the batch size changes.
    // These arrays record shape numbers; they do not calculate actual tensors.
    for batch_size in [2_usize, 1_usize] {
        let flat_shape = [batch_size, in_features];
        let score_shape = [batch_size, 5];
        println!("Flatten: {flat_shape:?}, scores: {score_shape:?}");
    }

    // Add the main-path value to the input at one position, then apply ReLU.
    let original: f32 = 2.0;
    let transformed: f32 = -1.0;
    let residual_result = relu(transformed + original);
    assert_eq!(residual_result, 1.0);
    println!("One residual output position: {residual_result}");
}
