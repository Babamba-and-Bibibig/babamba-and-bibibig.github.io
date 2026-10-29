// Main path in this post: conv1 uses K=3/P=1; conv2 uses K=3/P=1/S=1.
fn use_identity(in_channels: usize, out_channels: usize, stride: usize) -> bool {
    in_channels == out_channels && stride == 1
}

// Output size along one axis with dilation=1 and equal integer padding on each side.
fn output_size(input: usize, kernel: usize, padding: usize, stride: usize) -> usize {
    assert!(input > 0 && kernel > 0 && stride > 0);
    let padded = input + 2 * padding;

    // Check that the kernel fits before subtracting unsigned integers.
    assert!(padded >= kernel);

    // The numerator is nonnegative, so integer division gives floor.
    (padded - kernel) / stride + 1
}

fn main() {
    assert!(use_identity(8, 8, 1));
    assert!(!use_identity(8, 16, 1));  // ! reverses a Boolean value.
    assert!(!use_identity(8, 8, 2));

    // Both kernel/padding combinations match for even and odd input sizes.
    for input in [16_usize, 15, 13] {
        let main_size = output_size(input, 3, 1, 2);
        let skip_size = output_size(input, 1, 0, 2);
        assert_eq!(main_size, skip_size);
        println!("Input {input}: main path {main_size}, shortcut {skip_size}");
    }

    assert_eq!(output_size(16, 1, 1, 2), 9);  // Change padding to 1.
    assert_eq!(output_size(16, 3, 0, 2), 7);  // Remove padding from 3×3.
    println!("Identity conditions and output sizes verified");
}
