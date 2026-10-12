// 학습된 상태를 읽어 예측하는 흐름을 확인하는 독립 실행 예제입니다.
// 아래 저장값은 실제 학습 결과가 아닌 임의의 숫자입니다.
// origin 처리와 Ridge 학습은 포함하지 않습니다. 무게 단위는 lb입니다.
struct WeightModel {
    mean: f64,  // 학습에서 저장하는 무게의 평균입니다.
    scale: f64, // 학습에서 저장하는 변환용 나눗값입니다.
    w: f64,     // 변환된 무게에 곱하는 계수입니다.
    b: f64,     // 예측식에서 마지막에 더하는 절편입니다.
}

impl WeightModel {
    // &self는 모델을 읽기 위해 빌린 참조입니다. 저장값을 바꾸지 않습니다.
    // weight는 새 자동차의 무게이며 반환값은 예측 연비입니다.
    fn predict(&self, weight: f64) -> f64 {
        assert!(self.scale.is_finite() && self.scale > 0.0);
        let weight_scaled = (weight - self.mean) / self.scale;
        self.w * weight_scaled + self.b
    }
}

fn main() {
    // Python 마지막 예제와 같은 임의의 저장값으로 객체를 만듭니다.
    let model = WeightModel {
        mean: 3000.0,
        scale: 500.0,
        w: -4.0,
        b: 25.0,
    };
    let prediction = model.predict(2800.0);
    // 실수 계산의 작은 표현 오차를 허용하며 기대한 26.6과 비교합니다.
    assert!((prediction - 26.6).abs() < 1e-12);
    println!("임의의 저장값으로 확인한 예측(mpg): {prediction:.1}");
}
