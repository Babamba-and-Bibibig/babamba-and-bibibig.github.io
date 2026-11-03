// Python의 split/cat과 평균 집계 원리를 Rust 표준 라이브러리로 확인한다.
// 이 코드는 신경망·다차원 텐서·자동 미분을 구현한 코드가 아니다.

// 각 배치의 (평균 MSE, 그 평균을 만든 원소 수)를 받아 전체 평균을 계산한다.
fn weighted_epoch_mse(batches: &[(f64, usize)]) -> Option<f64> {
    let mut sse = 0.0;
    let mut count: usize = 0;

    // Python의 for 패턴처럼 튜플 안의 값을 각각의 변수로 받는다.
    for &(mean_loss, elements) in batches {
        sse += mean_loss * elements as f64;
        count += elements;
    }

    // 비교한 원소가 없으면 평균을 만들 수 없으므로 None을 반환한다.
    if count == 0 {
        None
    } else {
        Some(sse / count as f64)
    }
}

fn main() {
    // 600은 분리·연결을 확인하기 위한 예시 ID 개수이며 실제 검증 이미지 수가 아니다.
    // 0..600의 끝값 600은 제외되므로 실제 ID는 0~599다.
    let image_ids: Vec<usize> = (0..600).collect();
    let mut combined: Vec<usize> = Vec::new();

    // chunks(256)는 원본을 빌려 보는 슬라이스를 최대 256개씩 제공한다.
    for part in image_ids.chunks(256) {
        println!("묶음의 이미지 수: {}", part.len());
        // 모델 추론 대신 ID의 순서 보존만 확인한다.
        combined.extend_from_slice(part);
    }
    assert_eq!(combined, image_ids);
    println!("전체 이미지 수: {}", combined.len());

    // 0.04와 0.01은 평균 집계를 설명하기 위해 정한 예시 손실이며 실험 성능이 아니다.
    // 원소 수는 각각 흑백 28×28 이미지 64장과 8장에서 계산했다.
    let batches = [(0.04, 64 * 28 * 28), (0.01, 8 * 28 * 28)];
    match weighted_epoch_mse(&batches) {
        Some(mean) => println!("예시 두 배치의 원소 수 가중 평균 MSE: {mean:.6}"),
        None => println!("평균을 계산할 원소가 없습니다."),
    }
}
