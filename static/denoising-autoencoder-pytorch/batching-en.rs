// Explore Python-style split/cat and weighted averaging with the Rust standard library.
// This example does not implement a neural network, multidimensional tensors, or autograd.

// Compute an overall mean from each batch's (mean MSE, number of averaged elements).
fn weighted_epoch_mse(batches: &[(f64, usize)]) -> Option<f64> {
    let mut sse = 0.0;
    let mut count: usize = 0;

    // Destructure each tuple into variables, as with a Python for-loop pattern.
    for &(mean_loss, elements) in batches {
        sse += mean_loss * elements as f64;
        count += elements;
    }

    // Return None when no elements were compared, because their mean is undefined.
    if count == 0 {
        None
    } else {
        Some(sse / count as f64)
    }
}

fn main() {
    // 600 is an illustrative ID count for checking chunking, not the actual validation size.
    // The endpoint of 0..600 is excluded, so the IDs run from 0 through 599.
    let image_ids: Vec<usize> = (0..600).collect();
    let mut combined: Vec<usize> = Vec::new();

    // chunks(256) yields borrowed slices containing at most 256 elements each.
    for part in image_ids.chunks(256) {
        println!("Images in chunk: {}", part.len());
        // Check that ID order is preserved; no model inference is performed here.
        combined.extend_from_slice(part);
    }
    assert_eq!(combined, image_ids);
    println!("Total images: {}", combined.len());

    // 0.04 and 0.01 are illustrative losses for averaging, not measured model performance.
    // The element counts come from 64 and 8 grayscale images of size 28 by 28.
    let batches = [(0.04, 64 * 28 * 28), (0.01, 8 * 28 * 28)];
    match weighted_epoch_mse(&batches) {
        Some(mean) => println!("Element-weighted MSE of the two example batches: {mean:.6}"),
        None => println!("There are no elements to average."),
    }
}
