"""Review preprocessing, cross-validation, and Ridge with fictional cars.

This code does not reconstruct the course functions or reproduce course results.
The 16 records are fictional data created to demonstrate the processing sequence.
weight is measured in lb (pounds), and mpg is miles traveled per gallon of fuel.
origin values 1, 2, and 3 are fictional region codes, not magnitudes or rankings.
None of the values, including mpg, are measurements of real cars.
Required libraries: pandas, numpy, scikit-learn (1.2 or later).
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

# 1. Each row is one fictional car. Target mpg values are also invented for this example.
car_frame = pd.DataFrame(
    [
        (1900, 1, 34.0), (2050, 2, 35.0), (2200, 3, 34.5), (2300, 1, 29.8),
        (2450, 2, 30.8), (2550, 3, 30.2), (2700, 1, 25.9), (2800, 2, 28.0),
        (2900, 3, 27.5), (3050, 1, 22.8), (3150, 2, 24.5), (3300, 3, 23.4),
        (3450, 1, 19.5), (3600, 2, 20.5), (3750, 3, 19.8), (3900, 1, 16.6),
    ],
    columns=["weight", "origin", "mpg"],
    # The index identifies each car. It is not included in the model inputs.
    index=pd.Index(range(101, 117), name="row_id"),
)
X = car_frame[["weight", "origin"]]  # Inputs are a DataFrame with shape (16, 2).
y = car_frame["mpg"]  # Targets are a Series with shape (16,).

# train_test_split splits inputs and targets together, preserving their car correspondence.
# test_size=4 holds out 4 evaluation records; random_state=42 reproduces this split.
train_frame, valid_frame, y_train, y_valid = train_test_split(
    X, y, test_size=4, random_state=42
)
assert train_frame.shape == (12, 2) and valid_frame.shape == (4, 2)
assert train_frame.index.equals(y_train.index)
assert valid_frame.index.equals(y_valid.index)
assert train_frame.index.intersection(valid_frame.index).empty
print("Fictional training/separate evaluation shapes:", train_frame.shape, valid_frame.shape)
print("Input and target index correspondence verified.")

# A single column name selects a Series; a list of names keeps a DataFrame, hence double brackets.
X_train_num = train_frame[["weight"]].to_numpy(dtype=float)
valid_row_ids = valid_frame.index.to_numpy()  # Information for identifying the original rows.
print("Weight Series / table / array shapes:",
      train_frame["weight"].shape, train_frame[["weight"]].shape, X_train_num.shape)
print("Separate evaluation car row_ids:", valid_row_ids)

# 2. Manual preprocessing is an exercise in inspecting the structure of transformed data.
# X_train_ready, constructed using all training data here, is not passed to cross-validation below.
# StandardScaler creates a scaler object that stores means and variances.
scaler = StandardScaler()
fit_result = scaler.fit(train_frame[["weight"]])
assert fit_result is scaler  # fit returns the object itself, not a transformed array.
train_weight = scaler.transform(train_frame[["weight"]])
valid_weight = scaler.transform(valid_frame[["weight"]])  # Reuse the same training statistics.
print("Standardized training/separate evaluation array shapes:", train_weight.shape, valid_weight.shape)

# MinMaxScaler is also a class. Its defaults map the training minimum to 0 and maximum to 1.
minmax_scaler = MinMaxScaler()
train_weight_minmax = minmax_scaler.fit_transform(train_frame[["weight"]])
valid_weight_minmax = minmax_scaler.transform(valid_frame[["weight"]])
print("MinMax range of training weights:", train_weight_minmax.min(), train_weight_minmax.max())
print("MinMax range of separate evaluation weights:", valid_weight_minmax.min(), valid_weight_minmax.max())

# Pass the previously defined list of allowed codes to the CategoricalDtype class.
# This list was not learned by inspecting validation data. ordered=False means no magnitude ordering.
origin_dtype = pd.CategoricalDtype(categories=[1, 2, 3], ordered=False)
train_cat = train_frame[["origin"]].astype({"origin": origin_dtype})
valid_cat = valid_frame[["origin"]].astype({"origin": origin_dtype})
assert not train_cat.isna().any().any() and not valid_cat.isna().any().any()

# get_dummies creates 0/1 columns for regions. dtype=int requests integer output.
# The three specified categories retain their columns even when a region has no rows.
train_onehot = pd.get_dummies(train_cat, columns=["origin"], dtype=int)
valid_onehot = pd.get_dummies(valid_cat, columns=["origin"], dtype=int)
assert train_onehot.columns.equals(valid_onehot.columns)

# Arrays lack the original row index, so attach each car's index when constructing the new tables.
train_weight_frame = pd.DataFrame(
    train_weight, index=train_frame.index, columns=["weight_scaled"]
)
valid_weight_frame = pd.DataFrame(
    valid_weight, index=valid_frame.index, columns=["weight_scaled"]
)
# axis=1 in concat aligns matching row index labels and adds columns side by side.
X_train_ready = pd.concat([train_weight_frame, train_onehot], axis=1)
X_valid_ready = pd.concat([valid_weight_frame, valid_onehot], axis=1)
assert X_train_ready.index.equals(y_train.index)
assert X_valid_ready.index.equals(y_valid.index)
assert X_train_ready.columns.equals(X_valid_ready.columns)
assert X_train_ready.shape == (12, 4) and X_valid_ready.shape == (4, 4)
print("Manual preprocessing result shapes:", X_train_ready.shape, X_valid_ready.shape)
print(X_train_ready.head())  # The head method shows the first few rows.

# 3. Pass a Pipeline receiving the original train_frame to actual cross-validation.
# ColumnTransformer combines results specified as (name, transformer, target columns).
preprocess = ColumnTransformer([
    ("weight", StandardScaler(), ["weight"]),
    ("origin", OneHotEncoder(
        categories=[[1, 2, 3]],  # Previously allowed codes for the single origin column.
        handle_unknown="error",  # Report an error for codes outside the allowed list.
        sparse_output=False,  # Return an ordinary two-dimensional NumPy array.
    ), ["origin"]),
])
# Pipeline creates an object that runs named preprocessing steps and a model in order.
# "preprocess", "model", and "weight" are names chosen for this example.
cv_model = Pipeline([
    ("preprocess", preprocess),
    ("model", Ridge(alpha=1.0)),  # alpha is regularization strength, not a learning rate.
])
# The baseline is the reference model: linear regression using only weight here.
baseline = Pipeline([
    ("preprocess", ColumnTransformer([
        ("weight", StandardScaler(), ["weight"]),
    ])),
    ("model", LinearRegression()),
])

# KFold defines 3 splits. Each uses 8 training records and 4 validation records.
# shuffle=True shuffles before splitting; random_state=42 reproduces the same split.
# list stores the splits so every candidate uses the same car groups in each split.
cv_splits = list(KFold(3, shuffle=True, random_state=42).split(train_frame, y_train))

# cross_validate returns a dictionary of scores and times for each split.
# Each split fits a new Pipeline, so preprocessing statistics use only that split's training data.
baseline_result = cross_validate(
    baseline, train_frame, y_train, cv=cv_splits,
    scoring="neg_mean_squared_error", return_train_score=True,
)
cv_result = cross_validate(
    cv_model, train_frame, y_train, cv=cv_splits,
    scoring="neg_mean_squared_error", return_train_score=True,
)
# Scorers return higher values for better models. Negating negative MSE restores actual MSE.
# test_score evaluates each split's validation portion, not the separate valid_frame.
cv_table = pd.DataFrame({
    "Training MSE": -cv_result["train_score"],
    "Validation MSE": -cv_result["test_score"],
    "Baseline validation MSE": -baseline_result["test_score"],
})
print("Cross-validation results (fictional data):")
print(cv_table)
print("Mean validation MSE of the weight-only baseline:", cv_table["Baseline validation MSE"].mean())
print("Mean validation MSE of Ridge using weight and region:", cv_table["Validation MSE"].mean())
# The models also use different input features. Do not attribute the difference to regularization alone.

# 4. GridSearchCV configures candidate cross-validation and final refitting of the selected candidate.
# model__alpha refers to alpha in the Pipeline's "model" step.
search = GridSearchCV(
    estimator=cv_model,
    param_grid={"model__alpha": [0.01, 0.1, 1, 10, 100]},
    cv=cv_splits, scoring="neg_mean_squared_error", refit=True,
)
search.fit(train_frame, y_train)  # After 5 candidates × 3 fits, refit on 12 records with the selected setting.
results = pd.DataFrame(search.cv_results_)  # The cv_results_ attribute is a dictionary of candidate results.
comparison = pd.DataFrame({
    "alpha": results["param_model__alpha"],
    "Mean validation MSE": -results["mean_test_score"],
})
print("Setting comparison (fictional data):")
print(comparison.round(4).to_string(index=False))  # Round only the display; the model is unchanged.
print("Selected setting:", search.best_params_)
print("Selected candidate's mean validation MSE:", -search.best_score_)

# 5. With refit=True, best_estimator_ is already a Pipeline refitted on all training data.
selected_model = search.best_estimator_  # This assignment does not run additional training.
final_prediction = selected_model.predict(valid_frame)  # The returned array has shape (4,).
final_mse = mean_squared_error(y_valid, final_prediction)  # A scalar MSE for single-output regression.
print("Separate evaluation MSE (fictional data):", final_mse)
sample_request = pd.DataFrame({"weight": [2800], "origin": [2]})
sample_prediction = selected_model.predict(sample_request)  # Use fitted statistics and coefficients.
print("Fictional model's prediction for the new request (mpg):", sample_prediction)

# Use named_steps and named_transformers_ to locate fitted objects by name.
fitted_preprocess = selected_model.named_steps["preprocess"]
fitted_scaler = fitted_preprocess.named_transformers_["weight"]
fitted_ridge = selected_model.named_steps["model"]
print("Mean stored during final fitting:", fitted_scaler.mean_)
print("Standard deviation for transformation stored during final fitting:", fitted_scaler.scale_)
print("Learned coefficients:", fitted_ridge.coef_)
print("Learned intercept:", fitted_ridge.intercept_)

# Uncomment the next two lines only in a course environment where run_gd has been imported.
# print(inspect.getfile(run_gd))  # Inspect the file containing the function definition.
# print(inspect.getsource(run_gd))  # Inspect the implementation when Python source is accessible.

# 6. A separate calculation for comparison with Rust. These are arbitrary saved values, not fitted values above.
# This checks weight-only standardization and prediction; origin encoding and Ridge fitting are excluded.
saved_mean, saved_scale, saved_w, saved_b = 3000.0, 500.0, -4.0, 25.0
demo_weight = 2800.0
demo_weight_scaled = (demo_weight - saved_mean) / saved_scale
demo_prediction = saved_w * demo_weight_scaled + saved_b
assert np.isclose(demo_prediction, 26.6)
print("Prediction checked with arbitrary saved values (mpg):", demo_prediction)
