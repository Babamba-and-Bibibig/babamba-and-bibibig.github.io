"""가상 자동차로 실행하는 전처리·교차검증·Ridge 복습 예제.

필요한 make_model 함수를 이 파일에서 직접 정의하므로 별도 보조 파일 없이 실행합니다.
16건의 값은 처리 순서를 확인하기 위해 직접 만든 가상 자료입니다.
weight의 단위는 lb(파운드), mpg는 갤런당 주행 마일 수입니다.
origin의 1·2·3은 가상 생산 지역 코드이며 크기나 순서를 뜻하지 않습니다.
mpg를 포함한 모든 값은 실제 자동차의 측정값이 아닙니다.
필요한 라이브러리: pandas, numpy, scikit-learn(1.2 이상).
"""

import inspect

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# 1. 한 행은 가상 자동차 한 대입니다. 정답 mpg도 예제용으로 정한 값입니다.
car_frame = pd.DataFrame(
    [
        (1900, 1, 34.0), (2050, 2, 35.0), (2200, 3, 34.5), (2300, 1, 29.8),
        (2450, 2, 30.8), (2550, 3, 30.2), (2700, 1, 25.9), (2800, 2, 28.0),
        (2900, 3, 27.5), (3050, 1, 22.8), (3150, 2, 24.5), (3300, 3, 23.4),
        (3450, 1, 19.5), (3600, 2, 20.5), (3750, 3, 19.8), (3900, 1, 16.6),
    ],
    columns=["weight", "origin", "mpg"],
    # index는 각 자동차를 확인하는 이름입니다. 모델의 입력에는 넣지 않습니다.
    index=pd.Index(range(101, 117), name="row_id"),
)
X = car_frame[["weight", "origin"]]  # 입력은 DataFrame이며 shape은 (16, 2)입니다.
y = car_frame["mpg"]  # 정답은 Series이며 shape은 (16,)입니다.

# train_test_split 함수는 입력과 정답을 같은 자동차 순서로 함께 나눕니다.
# test_size=4는 별도 평가용 4건, random_state=42는 이 분할의 재현용 설정입니다.
train_frame, valid_frame, y_train, y_valid = train_test_split(
    X, y, test_size=4, random_state=42
)
assert train_frame.shape == (12, 2) and valid_frame.shape == (4, 2)
assert train_frame.index.equals(y_train.index)
assert valid_frame.index.equals(y_valid.index)
assert train_frame.index.intersection(valid_frame.index).empty
print("가상 자료의 학습/별도 평가 shape:", train_frame.shape, valid_frame.shape)
print("입력과 정답의 index 대응을 확인했습니다.")

# 대괄호 하나는 Series, 컬럼명 목록을 전달한 이중 대괄호는 DataFrame입니다.
X_train_num = train_frame[["weight"]].to_numpy(dtype=float)
valid_row_ids = valid_frame.index.to_numpy()  # 원래 행을 확인하기 위한 정보입니다.
print("무게 Series / 표 / 배열의 shape:",
      train_frame["weight"].shape, train_frame[["weight"]].shape, X_train_num.shape)
print("별도 평가 자동차의 row_id:", valid_row_ids)

# 2. 수동 전처리는 변환 결과의 구조를 확인하는 실습입니다.
# 여기서 전체 학습 자료로 만든 X_train_ready를 아래 교차검증에 넣지 않습니다.
# StandardScaler 클래스는 평균과 분산을 저장할 스케일러 객체를 만듭니다.
scaler = StandardScaler()
fit_result = scaler.fit(train_frame[["weight"]])
assert fit_result is scaler  # fit 메서드의 반환값은 변환 배열이 아니라 객체 자신입니다.
train_weight = scaler.transform(train_frame[["weight"]])
valid_weight = scaler.transform(valid_frame[["weight"]])  # 같은 학습 기준을 씁니다.
print("표준화한 학습/별도 평가 배열 shape:", train_weight.shape, valid_weight.shape)

# MinMaxScaler도 클래스입니다. 기본 설정에서는 학습 최솟값을 0, 최댓값을 1로 만듭니다.
minmax_scaler = MinMaxScaler()
train_weight_minmax = minmax_scaler.fit_transform(train_frame[["weight"]])
valid_weight_minmax = minmax_scaler.transform(valid_frame[["weight"]])
print("학습 무게의 MinMax 범위:", train_weight_minmax.min(), train_weight_minmax.max())
print("별도 평가 무게의 MinMax 범위:", valid_weight_minmax.min(), valid_weight_minmax.max())

# CategoricalDtype 클래스에 사전에 정한 허용 코드 목록을 전달합니다.
# 이 목록은 검증 자료를 조사해 얻은 것이 아닙니다. ordered=False는 대소 순서가 없다는 뜻입니다.
origin_dtype = pd.CategoricalDtype(categories=[1, 2, 3], ordered=False)
train_cat = train_frame[["origin"]].astype({"origin": origin_dtype})
valid_cat = valid_frame[["origin"]].astype({"origin": origin_dtype})
assert not train_cat.isna().any().any() and not valid_cat.isna().any().any()

# get_dummies 함수는 지역별 0/1 컬럼을 만듭니다. dtype=int는 정수로 출력합니다.
# 어느 지역의 행이 없어도, 위에 지정한 세 범주에 대응하는 컬럼을 유지합니다.
train_onehot = pd.get_dummies(train_cat, columns=["origin"], dtype=int)
valid_onehot = pd.get_dummies(valid_cat, columns=["origin"], dtype=int)
assert train_onehot.columns.equals(valid_onehot.columns)

# 배열에는 원래 행 index가 없으므로 새 표를 만들 때 해당 자동차의 index를 붙입니다.
train_weight_frame = pd.DataFrame(
    train_weight, index=train_frame.index, columns=["weight_scaled"]
)
valid_weight_frame = pd.DataFrame(
    valid_weight, index=valid_frame.index, columns=["weight_scaled"]
)
# concat 함수의 axis=1은 같은 행 index를 맞추며 컬럼을 옆으로 붙입니다.
X_train_ready = pd.concat([train_weight_frame, train_onehot], axis=1)
X_valid_ready = pd.concat([valid_weight_frame, valid_onehot], axis=1)
assert X_train_ready.index.equals(y_train.index)
assert X_valid_ready.index.equals(y_valid.index)
assert X_train_ready.columns.equals(X_valid_ready.columns)
assert X_train_ready.shape == (12, 4) and X_valid_ready.shape == (4, 4)
print("수동 전처리 결과 shape:", X_train_ready.shape, X_valid_ready.shape)
print(X_train_ready.head())  # head 메서드는 앞부분을 보여 줍니다.

# 3. 반복해서 사용하는 전처리와 회귀 모델의 구성을 make_model 함수로 묶습니다.
# regressor에는 Ridge나 LinearRegression 객체를 전달합니다.
# include_origin은 지역을 입력에 포함할지 정하는 값이며, 기본값은 True입니다.
def make_model(regressor, include_origin=True):
    """전처리와 전달받은 회귀 모델을 묶은, 아직 학습하지 않은 Pipeline을 반환합니다.

    무게의 표준화는 항상 포함하고, include_origin=True일 때 지역 인코딩을 추가합니다.
    이 함수는 구성을 만들 뿐입니다. 기준과 계수의 학습은 이후 fit에서 실행합니다.
    """
    # 매번 새 목록과 스케일러를 만듭니다. 각 항목은 (이름, 변환 객체, 대상 컬럼)입니다.
    transformers = [("weight", StandardScaler(), ["weight"])]
    if include_origin:
        # append 메서드는 지역 변환 항목을 현재 목록의 끝에 추가합니다.
        transformers.append((
            "origin",
            OneHotEncoder(
                categories=[[1, 2, 3]],  # origin 한 컬럼의 사전 허용 코드 목록입니다.
                handle_unknown="error",  # 허용 목록 밖의 코드가 오면 오류로 알립니다.
                sparse_output=False,  # 일반적인 2차원 NumPy 배열을 출력합니다.
            ),
            ["origin"],
        ))
    # ColumnTransformer는 변환 결과를 결합합니다. remainder="drop"은 지정하지 않은 컬럼을 제외합니다.
    preprocess = ColumnTransformer(transformers, remainder="drop")
    # Pipeline은 전처리와 모델을 순서대로 실행합니다. 단계 이름은 이 함수에서 정합니다.
    return Pipeline([
        ("preprocess", preprocess),
        ("model", regressor),
    ])


# 함수 호출은 구성만 만듭니다. 실제 교차검증은 아래에서 원래 train_frame으로 실행합니다.
# alpha는 규제 강도이며 학습률이 아닙니다. include_origin=True는 표준화한 무게와 지역 표시를 함께 사용합니다.
cv_model = make_model(Ridge(alpha=1.0), include_origin=True)
# baseline은 비교 기준입니다. include_origin=False이면 지역을 제외하고 무게만 사용합니다.
baseline = make_model(LinearRegression(), include_origin=False)

# KFold 클래스는 3회 분할을 정합니다. 각 회차는 훈련 8건과 검증 4건을 사용합니다.
# shuffle=True는 분할 전 섞기, random_state=42는 같은 분할을 재현하기 위한 설정입니다.
# list는 분할 결과를 보관해 모든 후보에 같은 회차 구성을 적용하게 합니다.
cv_splits = list(KFold(3, shuffle=True, random_state=42).split(train_frame, y_train))

# cross_validate 함수는 회차별 점수와 시간을 담은 딕셔너리를 반환합니다.
# 매 회차 새 Pipeline을 학습하므로 전처리 기준도 해당 회차의 훈련 자료에서만 구합니다.
baseline_result = cross_validate(
    baseline, train_frame, y_train, cv=cv_splits,
    scoring="neg_mean_squared_error", return_train_score=True,
)
cv_result = cross_validate(
    cv_model, train_frame, y_train, cv=cv_splits,
    scoring="neg_mean_squared_error", return_train_score=True,
)
# scorer는 클수록 좋은 점수를 반환합니다. 음수 MSE에 -를 붙이면 실제 MSE가 됩니다.
# test_score는 각 회차의 검증 점수이며 별도 valid_frame의 점수가 아닙니다.
cv_table = pd.DataFrame({
    "학습 MSE": -cv_result["train_score"],
    "검증 MSE": -cv_result["test_score"],
    "기준 모델 검증 MSE": -baseline_result["test_score"],
})
print("교차검증 결과(가상 자료):")
print(cv_table)
print("무게만 쓰는 기준 모델의 평균 검증 MSE:", cv_table["기준 모델 검증 MSE"].mean())
print("무게와 지역을 쓰는 Ridge의 평균 검증 MSE:", cv_table["검증 MSE"].mean())
# 두 모델은 입력 특성도 다릅니다. 차이를 규제 하나의 효과로 해석하지 않습니다.

# 4. GridSearchCV 클래스는 후보별 교차검증과 좋은 후보의 최종 재학습을 구성합니다.
# model__alpha는 Pipeline의 "model" 단계에 있는 alpha 설정을 가리킵니다.
search = GridSearchCV(
    estimator=cv_model,
    param_grid={"model__alpha": [0.01, 0.1, 1, 10, 100]},
    cv=cv_splits, scoring="neg_mean_squared_error", refit=True,
)
search.fit(train_frame, y_train)  # 5후보×3회 학습 후, 선택된 설정으로 12건을 다시 학습합니다.
results = pd.DataFrame(search.cv_results_)  # cv_results_ 속성은 후보별 결과 딕셔너리입니다.
comparison = pd.DataFrame({
    "alpha": results["param_model__alpha"],
    "평균 검증 MSE": -results["mean_test_score"],
})
print("설정 비교(가상 자료):")
print(comparison.round(4).to_string(index=False))  # 출력만 반올림하며 모델은 바뀌지 않습니다.
print("선택된 설정:", search.best_params_)
print("선택된 후보의 평균 검증 MSE:", -search.best_score_)

# 5. best_estimator_는 refit=True에 따라 이미 전체 학습 자료로 다시 학습된 Pipeline입니다.
selected_model = search.best_estimator_  # 이 대입은 추가 학습을 실행하지 않습니다.
final_prediction = selected_model.predict(valid_frame)  # 반환 배열의 shape은 (4,)입니다.
final_mse = mean_squared_error(y_valid, final_prediction)  # 단일 출력 회귀의 MSE 숫자입니다.
print("별도 평가 MSE(가상 자료):", final_mse)
sample_request = pd.DataFrame({"weight": [2800], "origin": [2]})
sample_prediction = selected_model.predict(sample_request)  # 학습된 기준과 계수로 계산합니다.
print("가상 모델의 새 요청 예측(mpg):", sample_prediction)

# named_steps와 named_transformers_ 속성으로 이름에 해당하는 학습된 객체를 찾습니다.
fitted_preprocess = selected_model.named_steps["preprocess"]
fitted_scaler = fitted_preprocess.named_transformers_["weight"]
fitted_ridge = selected_model.named_steps["model"]
print("최종 학습에서 저장된 평균:", fitted_scaler.mean_)
print("최종 학습에서 저장된 변환용 표준편차:", fitted_scaler.scale_)
print("학습된 계수:", fitted_ridge.coef_)
print("학습된 절편:", fitted_ridge.intercept_)

# 아래 두 줄의 주석을 해제하면 이 파일에서 정의한 make_model의 위치와 본문을 확인합니다.
# print(inspect.getfile(make_model))  # 함수가 정의된 파일 위치를 확인합니다.
# print(inspect.getsource(make_model))  # 접근 가능한 Python 소스가 있을 때 함수 본문을 확인합니다.

# 6. Rust와 비교할 별도 계산입니다. 다음 값은 위 모델의 학습값이 아닌 임의의 저장값입니다.
# 무게 하나의 표준화와 예측만 확인하며, origin 인코딩과 Ridge 학습은 포함하지 않습니다.
saved_mean, saved_scale, saved_w, saved_b = 3000.0, 500.0, -4.0, 25.0
demo_weight = 2800.0
demo_weight_scaled = (demo_weight - saved_mean) / saved_scale
demo_prediction = saved_w * demo_weight_scaled + saved_b
assert np.isclose(demo_prediction, 26.6)
print("임의의 저장값으로 확인한 예측(mpg):", demo_prediction)
