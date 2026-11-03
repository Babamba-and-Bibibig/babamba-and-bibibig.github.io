// 4채널 오토인코더와 이미지별 MSE의 Rust 대응 예제.
// tch 0.26.0의 공식 API를 기준으로 작성했다. 이 버전은 libtorch 2.13.0이 필요하다.
// 이 글의 검수 환경에서는 Rust 컴파일·실행을 하지 않았다.
// main의 입력은 구조 확인용 인공 텐서이며 손글씨 복원 성능을 나타내지 않는다.

use tch::nn::{ModuleT, OptimizerConfig};
use tch::{nn, Device, Kind, Reduction, Tensor};

fn student_dae(path: &nn::Path<'_>) -> nn::SequentialT {
    // 각 설정값은 Python의 Conv2d/ConvTranspose2d와 같은 역할을 한다.
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
    // &Tensor는 빌린 텐서, &mut Optimizer는 수정할 수 있도록 빌린 옵티마이저다.
    optimizer.zero_grad();
    let pred = model.forward_t(noisy, true); // true는 학습 동작을 선택한다.
    assert_eq!(pred.size(), clean.size());
    let loss = pred.mse_loss(clean, Reduction::Mean);
    loss.backward();
    optimizer.step();
    // Python loss.item()처럼 숫자를 꺼낸다. 여전히 갱신 전 측정값이다.
    loss.double_value(&[])
}

fn per_image_mse(reconstruction: &Tensor, clean: &Tensor) -> Tensor {
    let shape = reconstruction.size();
    assert_eq!(shape, clean.size());
    assert_eq!(shape.len(), 4);
    assert!(shape.iter().all(|&size| size > 0));
    (reconstruction - clean)
        .square()
        .flatten(1, -1) // 이미지 축을 남기고 1번 축부터 마지막 축까지 합친다.
        .mean_dim(
            &[1_i64][..], // 이미지별 픽셀 축을 평균낸다.
            false,       // 평균낸 축을 크기 1로 남기지 않는다.
            Kind::Float, // 이 예제는 float32 이미지를 사용한다.
        )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tch::manual_seed(17);
    let vs = nn::VarStore::new(Device::Cpu); // 학습 파라미터를 보관하는 객체.
    let model = student_dae(&vs.root());
    let mut optimizer = nn::Adam::default().build(&vs, 0.01)?;
    let parameter_count: usize = vs.trainable_variables().iter().map(|p| p.numel()).sum();
    assert_eq!(parameter_count, 105);

    // 실제 손글씨 데이터가 아닌 구조 검산용 이미지 세 장이다.
    let clean = Tensor::rand([3, 1, 28, 28], (Kind::Float, Device::Cpu));
    let noisy = (&clean + clean.randn_like() * 0.25).clamp(0.0, 1.0);
    let loss_before_update = train_one_batch(&model, &mut optimizer, &noisy, &clean);
    println!("loss measured before the update: {loss_before_update}");

    // false는 평가 동작 선택, no_grad는 기울기 추적 중단이다.
    let recon = tch::no_grad(|| model.forward_t(&noisy, false));
    let each = per_image_mse(&recon, &clean);
    assert_eq!(each.size(), vec![3]);

    let scalar_losses = vec![Tensor::from(0.1_f32), Tensor::from(0.2_f32), Tensor::from(0.3_f32)];
    let stacked = Tensor::stack(&scalar_losses, 0); // 스칼라 세 개 → [3].
    let expanded = stacked.unsqueeze(1); // 크기 1인 축 추가 → [3, 1].
    println!("stack: {:?}, unsqueeze: {:?}", stacked.size(), expanded.size());
    Ok(())
}
