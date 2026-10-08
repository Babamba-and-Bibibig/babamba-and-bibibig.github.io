"""Supervised Learning: Core Concepts and Basic Coding, from delivery fees to real data.

Required packages: numpy, matplotlib, scikit-learn
In Jupyter, install them in a separate cell with %pip install numpy matplotlib scikit-learn
Run the code below from top to bottom.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=6, suppress=True)

# These fictional, simplified data describe delivery fees.
# x_one: additional distance beyond the distance included in the base fee (km); y_one: fee (thousand won).
# b is the base fee, and w is the fee per additional km. Start with b=0 and w=1.
x_one = np.array([0.0, 1.0, 2.0])
y_one = np.array([1.0, 3.0, 5.0])
b_initial, w_initial = 0.0, 1.0
pred_one = b_initial + w_initial * x_one
e_one = pred_one - y_one
sse = np.sum(e_one ** 2)
mse = np.mean(e_one ** 2)
j_before = mse / 2
mae = np.mean(np.abs(e_one))
rmse = np.sqrt(mse)
print("Initial delivery fee predictions:", pred_one)
print("Error = prediction - target:", e_one)
print("SSE, MSE, J, MAE, RMSE:", sse, mse, j_before, mae, rmse)
assert np.allclose(e_one, [-1.0, -2.0, -3.0], atol=1e-12, rtol=0)
assert np.isclose(j_before, 7 / 3, atol=1e-12, rtol=0)

# Increase only the base fee by 0.1 thousand won; keep the original fee per km.
b_trial = b_initial + 0.1
pred_b_trial = b_trial + w_initial * x_one
e_b_trial = pred_b_trial - y_one
j_b_trial = np.mean(e_b_trial ** 2) / 2
print("Predictions / errors / J after increasing only b by 0.1:", pred_b_trial, e_b_trial, j_b_trial)
assert np.allclose(pred_b_trial, [0.1, 1.1, 2.1], atol=1e-12, rtol=0)
assert np.allclose(e_b_trial, [-0.9, -1.9, -2.9], atol=1e-12, rtol=0)
assert np.isclose(j_b_trial, 1283 / 600, atol=1e-12, rtol=0)

# Start again from the original b=0, w=1, and increase only the fee per km by 0.1.
# Do not carry over b_trial from the previous experiment.
w_trial = w_initial + 0.1
pred_w_trial = b_initial + w_trial * x_one
e_w_trial = pred_w_trial - y_one
j_w_trial = np.mean(e_w_trial ** 2) / 2
print("Predictions / errors / J after increasing only w by 0.1:", pred_w_trial, e_w_trial, j_w_trial)
assert np.allclose(pred_w_trial, [0.0, 1.1, 2.2], atol=1e-12, rtol=0)
assert np.allclose(e_w_trial, [-1.0, -1.9, -2.8], atol=1e-12, rtol=0)
assert np.isclose(j_w_trial, 83 / 40, atol=1e-12, rtol=0)

# Use the derivatives to update the base fee and the fee per km together once.
# First calculate both derivatives from the errors at the original b=0, w=1.
db = np.mean(e_one)
dw = np.mean(e_one * x_one)
lr_one = 0.1
b = b_initial - lr_one * db
w = w_initial - lr_one * dw
pred_after = b + w * x_one
mse_after = np.mean((pred_after - y_one) ** 2)
j_after = mse_after / 2
print("db, dw before the update:", db, dw)
print("b, w, J after one update:", b, w, j_after)
assert np.allclose([db, dw], [-2.0, -8 / 3], atol=1e-12, rtol=0)
assert np.allclose([b, w], [0.2, 19 / 15], atol=1e-12, rtol=0)
assert np.isclose(j_after, 1829 / 1350, atol=1e-12, rtol=0)

# Writing the same three deliveries as a matrix gives the same predictions and derivatives.
# The leading column of ones adds the base fee b once to each delivery.
Xb_one = np.column_stack([np.ones(len(y_one)), x_one])
theta_one = np.array([b_initial, w_initial])
pred_matrix_one = Xb_one @ theta_one
gradient_one = Xb_one.T @ (pred_matrix_one - y_one) / len(y_one)
theta_after_matrix = theta_one - lr_one * gradient_one
print("Three-row delivery matrix Xb:\n", Xb_one)
print("Matrix predictions / [db, dw]:", pred_matrix_one, gradient_one)
print("[b, w] after one matrix update:", theta_after_matrix)
assert np.array_equal(Xb_one, [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
assert np.allclose(pred_matrix_one, pred_one, atol=1e-12, rtol=0)
assert np.allclose(gradient_one, [db, dw], atol=1e-12, rtol=0)
assert np.allclose(theta_after_matrix, [b, w], atol=1e-12, rtol=0)

# Select deliveries or the distance column and check the resulting array shapes.
print("Second delivery row:", Xb_one[1], "shape =", Xb_one[1].shape)
print("Keep the second delivery as one row:", Xb_one[[1]], "shape =", Xb_one[[1]].shape)
print("Distance column:", Xb_one[:, 1], "shape =", Xb_one[:, 1].shape)
print("Keep distance as a single column:", Xb_one[:, [1]], "shape =", Xb_one[:, [1]].shape)

# Subtract the target from the prediction for the same delivery.
# Turning only one array into a column produces an unintended 3-by-3 set of differences.
print("Prediction and target shapes:", pred_one.shape, y_one.shape)
print("Shape after turning targets into a column:", y_one[:, None].shape)
print("Correct errors for matching deliveries:", pred_one - y_one)
print("Unintended 3-by-3 result:\n", pred_one - y_one[:, None])
assert pred_one.shape == y_one.shape

# Compare learning rates on the same deliveries by fixing w=1 and updating only the base fee.
# Here db is b-2. Restart every experiment from the same b=0.
learning_rate_results = []
for rate in [0.2, 0.8, 1.5, 2.2]:
    b_rate = 0.0
    db_rate = b_rate - 2.0
    b_rate = b_rate - rate * db_rate
    pred_rate = b_rate + 1.0 * x_one
    j_rate = np.mean((pred_rate - y_one) ** 2) / 2
    learning_rate_results.append([rate, b_rate, j_rate])
    print("Learning rate / b after the first update / J:", rate, b_rate, j_rate)
assert np.allclose(np.array(learning_rate_results)[:, 1], [0.4, 1.6, 3.0, 4.4],
                   atol=1e-12, rtol=0)
assert np.allclose(np.array(learning_rate_results)[:, 2],
                   [121 / 75, 31 / 75, 5 / 6, 241 / 75], atol=1e-12, rtol=0)

# Add a second input: u is additional distance (km); v indicates special packaging (no=0, yes=1).
# Fees for these four deliveries are still measured in thousand won.
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = np.array([1.0, 3.0, 4.0, 6.0])
n = len(y)
Xb = np.column_stack([np.ones(n), X])
theta = np.zeros(3)              # [base fee b, fee per km w_u, special packaging fee w_v]
e0 = Xb @ theta - y
g0 = Xb.T @ e0 / n
j0 = np.mean(e0 ** 2) / 2
print("Four-row delivery matrix Xb:\n", Xb)
print("Initial J, gradient:", j0, g0)

# Train all three fees from zero with learning rate 0.2 for 2000 updates.
lr = 0.2
n_steps = 2000
history = []
for _ in range(n_steps):
    pred = Xb @ theta
    e = pred - y
    history.append(float(np.mean(e ** 2) / 2))  # Loss immediately before this update
    gradient = Xb.T @ e / n
    theta = theta - lr * gradient

theta_gd = theta.copy()
pred_gd = Xb @ theta_gd
final_j = float(np.mean((pred_gd - y) ** 2) / 2)
history_complete = np.array(history + [final_j])
print("GD coefficients [b, w_u, w_v]:", theta_gd)
print("GD delivery fee predictions:", pred_gd)
print("Initial / last stored / final loss:", history[0], history[-1], final_j)
print("History length including the loss after 2000 updates:", len(history_complete))

fig_loss, ax_loss = plt.subplots(figsize=(7, 4))
shown = min(81, len(history_complete))  # Show the starting loss and the first 80 completed updates.
ax_loss.plot(np.arange(shown), history_complete[:shown])
ax_loss.set(xlabel="Completed updates", ylabel="J = MSE / 2",
            title="Gradient descent on four delivery samples")
ax_loss.grid(alpha=0.25)
fig_loss.tight_layout()

# Fit the same four deliveries with least squares and scikit-learn as well.
# The first item in the least-squares result is the coefficient array we need.
result = np.linalg.lstsq(Xb, y)
theta_ls = result[0]
pred_ls = Xb @ theta_ls
model_small = LinearRegression()
fit_return = model_small.fit(X, y)  # The model handles the base fee, so use X without a leading column of ones.
pred_sk = model_small.predict(X)
print("Least-squares coefficients:", theta_ls)
print("Does fit return the same model:", fit_return is model_small)
print("sklearn base fee / distance and packaging fees:", model_small.intercept_, model_small.coef_)
print("Least-squares / GD / sklearn predictions as columns:\n", np.column_stack([pred_ls, pred_gd, pred_sk]))
new_input = np.array([1.0, 3.0, 1.0])  # [1 for the base fee, 3 additional km, special packaging]
new_prediction = new_input @ theta_gd
print("Fee for 3 additional km with special packaging (thousand won):", new_prediction)

# Compare arrays of matching shapes to check that all three methods predict the same fees.
assert theta_gd.shape == (3,)
assert pred_gd.shape == y.shape
assert np.isfinite(theta_gd).all()
assert np.isfinite(history_complete).all()
assert np.isclose(j0, 7.75, atol=1e-12, rtol=0)
assert np.allclose(g0, [-3.5, -2.25, -2.5], atol=1e-12, rtol=0)
assert np.allclose(theta_gd, [1.0, 2.0, 3.0], atol=1e-8, rtol=0)
assert np.allclose(pred_gd, pred_ls, atol=1e-8, rtol=0)
assert np.allclose(pred_gd, pred_sk, atol=1e-8, rtol=0)
assert np.isclose(new_prediction, 10.0, atol=1e-8, rtol=0)
assert final_j < history_complete[0]
print("Implementation checks passed. Largest GD/least-squares prediction difference:", np.max(np.abs(pred_gd - pred_ls)))

# Move to real data, using stored feature values without the loader's additional scaling.
X_real, y_real = load_diabetes(return_X_y=True, scaled=False)
# Apply the same row indices to inputs and targets to preserve their pairing.
all_ids = np.arange(len(y_real))
train_ids, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
X_train, X_test = X_real[train_ids], X_real[test_ids]
y_train, y_test = y_real[train_ids], y_real[test_ids]
assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]
print("Full / training / test input shapes:", X_real.shape, X_train.shape, X_test.shape)

# Build a baseline that predicts the mean training target for every test row.
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
pred_baseline = baseline.predict(X_test)

# Train separate models using BMI alone and all ten features.
model_bmi = LinearRegression()
model_bmi.fit(X_train[:, [2]], y_train)
pred_bmi = model_bmi.predict(X_test[:, [2]])
model_all = LinearRegression()
model_all.fit(X_train, y_train)
pred_all = model_all.predict(X_test)
predictions = {"baseline": pred_baseline, "bmi": pred_bmi, "all": pred_all}
scores = {}
for name, prediction in predictions.items():
    assert prediction.shape == y_test.shape
    assert np.isfinite(prediction).all()
    scores[name] = mean_squared_error(y_test, prediction)
    print(name, "Test MSE:", scores[name])
print("Mean training target used by the baseline:", y_train.mean())

# Residuals are target minus prediction, the opposite sign of the earlier errors e.
residual = y_test - pred_all
print("Original row index of the first test sample:", test_ids[0])
print("First test sample BMI:", X_test[0, 2])
print("First test target / all-feature prediction / residual:", y_test[0], pred_all[0], residual[0])
assert test_ids[0] == 287
assert np.isclose(X_test[0, 2], 25.8, atol=1e-12, rtol=0)
assert y_test[0] == 219.0
fig_residual, ax_residual = plt.subplots(figsize=(7, 4))
ax_residual.scatter(pred_all, residual, alpha=0.65, label="Test samples")
ax_residual.scatter([pred_all[0]], [residual[0]], color="crimson",
                    s=90, label="First test sample")
ax_residual.axhline(0, color="black", linewidth=1)
ax_residual.set(xlabel="Prediction", ylabel="Residual = target - prediction",
                title="Residuals on 89 held-out samples")
ax_residual.legend()
fig_residual.tight_layout()
plt.show()
