// Rust counterpart for a four-channel autoencoder and per-image MSE.
// Written against the official tch 0.26.0 API, which requires libtorch 2.13.0.
// This Rust example was not compiled or executed in the article verification environment.
// Inputs in main are synthetic tensors for structural checks, not handwriting performance.

use tch::nn::{ModuleT, OptimizerConfig};
use tch::{nn, Device, Kind, Reduction, Tensor};

fn student_dae(path: &nn::Path<'_>) -> nn::SequentialT {
    // These settings play the same roles as in Python Conv2d/ConvTranspose2d.
    let encoder_config = nn::ConvConfig {
        stride: 2,
        padding: 1,
        bias: true,
        ..Default::default()
    };
    let decoder_config = nn::ConvTransposeConfig {
        stride: 2,
        padding: 1,
        bias: true,
        ..Default::default()
    };
    nn::seq_t()
        .add(nn::conv2d(path / "encoder", 1, 4, 3, encoder_config))
        .add_fn(|x| x.relu())
        .add(nn::conv_transpose2d(path / "decoder", 4, 1, 4, decoder_config))
        .add_fn(|x| x.sigmoid())
}

fn train_one_batch(
    model: &impl ModuleT,
    optimizer: &mut nn::Optimizer,
    noisy: &Tensor,
    clean: &Tensor,
) -> f64 {
    // &Tensor borrows a tensor; &mut Optimizer borrows an optimizer for modification.
    optimizer.zero_grad();
    let pred = model.forward_t(noisy, true); // true selects training behavior.
    assert_eq!(pred.size(), clean.size());
    let loss = pred.mse_loss(clean, Reduction::Mean);
    loss.backward();
    optimizer.step();
    // Extract a number like Python loss.item(); this is still the pre-update measurement.
    loss.double_value(&[])
}

fn per_image_mse(reconstruction: &Tensor, clean: &Tensor) -> Tensor {
    let shape = reconstruction.size();
    assert_eq!(shape, clean.size());
    assert_eq!(shape.len(), 4);
    assert!(shape.iter().all(|&size| size > 0));
    (reconstruction - clean)
        .square()
        .flatten(1, -1) // Keep image axis 0; combine axis 1 through the final axis.
        .mean_dim(
            &[1_i64][..], // Average the per-image pixel axis.
            false,       // Do not retain the averaged axis with size 1.
            Kind::Float, // This example uses float32 images.
        )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tch::manual_seed(17);
    let vs = nn::VarStore::new(Device::Cpu); // Stores the trainable parameters.
    let model = student_dae(&vs.root());
    let mut optimizer = nn::Adam::default().build(&vs, 0.01)?;
    let parameter_count: usize = vs.trainable_variables().iter().map(|p| p.numel()).sum();
    assert_eq!(parameter_count, 105);

    // Three synthetic images for structural checks, not actual handwriting data.
    let clean = Tensor::rand([3, 1, 28, 28], (Kind::Float, Device::Cpu));
    let noisy = (&clean + clean.randn_like() * 0.25).clamp(0.0, 1.0);
    let loss_before_update = train_one_batch(&model, &mut optimizer, &noisy, &clean);
    println!("loss measured before the update: {loss_before_update}");

    // false selects evaluation behavior; no_grad disables gradient tracking.
    let recon = tch::no_grad(|| model.forward_t(&noisy, false));
    let each = per_image_mse(&recon, &clean);
    assert_eq!(each.size(), vec![3]);

    let scalar_losses = vec![Tensor::from(0.1_f32), Tensor::from(0.2_f32), Tensor::from(0.3_f32)];
    let stacked = Tensor::stack(&scalar_losses, 0); // Three scalars -> [3].
    let expanded = stacked.unsqueeze(1); // Add a size-1 axis -> [3, 1].
    println!("stack: {:?}, unsqueeze: {:?}", stacked.size(), expanded.size());
    Ok(())
}
