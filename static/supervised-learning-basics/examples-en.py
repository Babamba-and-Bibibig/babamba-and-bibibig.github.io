"""Supervised Learning: Core Concepts and Basic Coding — runnable Python examples.

Required packages: numpy, matplotlib, scikit-learn
In Jupyter, install in a separate cell: %pip install numpy matplotlib scikit-learn
Run the code below from top to bottom in one pass.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Change only the number of displayed decimal places, not the calculation precision.
np.set_printoptions(precision=6, suppress=True)

# 1. Errors, sum of squares, mean squared error, and training loss J
y_demo = np.array([10.0, 20.0, 30.0])
pred_demo = np.array([9.0, 18.0, 27.0])
e_demo = pred_demo - y_demo
sse = np.sum(e_demo ** 2)       # ** 2: square each element
mse = np.mean(e_demo ** 2)      # mean: divide the sum of the elements by their count
j_demo = mse / 2                # The factor 1/2 is not part of the definition of MSE itself
mae = np.mean(np.abs(e_demo))    # abs: magnitude without the sign
rmse = np.sqrt(mse)             # sqrt: square root
print("Errors:", e_demo)
print("SSE, MSE, J, MAE, RMSE:", sse, mse, j_demo, mae, rmse)

# 2. Indexing: select rows and columns, and check the resulting shape
A = np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
row_ids = [0, 2]
index_examples = {
    "A[1]": A[1],
    "A[[1]]": A[[1]],
    "A[:, 2]": A[:, 2],
    "A[:, [2]]": A[:, [2]],
    "A[row_ids][1]": A[row_ids][1],
    "A[row_ids][:, [2]]": A[row_ids][:, [2]],
    "A[[0, 2], [1, 2]]": A[[0, 2], [1, 2]],
}
# items(): retrieve each name and result as a pair from the dictionary
for label, values in index_examples.items():
    print(label, "=", values, "shape =", values.shape)
print("A[1][2] =", A[1][2])

# 3. Adding axes, reshaping, and possible broadcasting mistakes
v = np.array([1.0, 3.0, 5.0])
p = np.array([0.0, 2.0, 4.0])
print("Shapes of v and v.T:", v.shape, v.T.shape)
print("v[:, None]:", v[:, None], "shape =", v[:, None].shape)
print("v[None, :]:", v[None, :], "shape =", v[None, :].shape)
print("reshape(-1, 1):", v.reshape(-1, 1))
print("Correct errors at matching positions:", p - v)
print("Unintended 3×3 result:\n", p - v[:, None])
assert p.shape == v.shape

# 4. Use derivatives to update a model with one input feature once
x_one = np.array([0.0, 1.0, 2.0])
y_one = np.array([1.0, 3.0, 5.0])
w, b = 1.0, 0.0
lr_one = 0.1
pred_one = b + w * x_one
e_one = pred_one - y_one
j_before = np.mean(e_one ** 2) / 2
dw = np.mean(e_one * x_one)      # Current value of the derivative of J with respect to w
db = np.mean(e_one)              # Current value of the derivative of J with respect to b
# Compute both derivatives at the existing w and b before updating either coefficient.
w = w - lr_one * dw
b = b - lr_one * db
pred_after = b + w * x_one
mse_after = np.mean((pred_after - y_one) ** 2)
print("J, dw, db before the single update:", j_before, dw, db)
print("w, b, MSE after the update:", w, b, mse_after)
assert np.isclose(j_before, 7 / 3, atol=1e-12, rtol=0)
assert np.isclose(mse_after, 1829 / 675, atol=1e-12, rtol=0)

# 5. A separate dataset with four rows and two input features
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
y = np.array([1.0, 3.0, 4.0, 6.0])
n = len(y)
# ones(n): create n ones. c_: join arrays side by side as columns.
Xb = np.c_[np.ones(n), X]
theta = np.zeros(3)              # Coefficient order: [b, w1, w2]
e0 = Xb @ theta - y              # @: matrix multiplication of each row by the coefficient vector
g0 = Xb.T @ e0 / n              # .T: swap the rows and columns of this two-dimensional array
j0 = np.mean(e0 ** 2) / 2
trial_theta = theta - 0.1 * g0   # Store the result under another name; theta is still zero
print("Xb:\n", Xb)
print("Initial J, gradient:", j0, g0)
print("Trial update with learning rate 0.1:", trial_theta)
print("Original theta after the trial calculation:", theta)

# 6. Run the actual iterations with learning rate 0.2, starting from zero coefficients
lr = 0.2
n_steps = 2000
history = []                     # List that stores each loss immediately before the corresponding update
for _ in range(n_steps):         # range(2000): 0 through 1999, for a total of 2000 iterations
    pred = Xb @ theta
    e = pred - y
    history.append(float(np.mean(e ** 2) / 2))
    gradient = Xb.T @ e / n
    theta -= lr * gradient      # In-place update that changes the contents of the existing array

theta_gd = theta.copy()          # Independent copy that preserves the result if theta changes later
pred_gd = Xb @ theta_gd
final_j = float(np.mean((pred_gd - y) ** 2) / 2)
history_complete = np.array(history + [final_j])
print("GD coefficients:", theta_gd)
print("GD predictions:", pred_gd)
print("First loss / last stored loss / final loss:", history[0], history[-1], final_j)
print("Length including the loss after 2000 updates:", len(history_complete))

# subplots returns the entire figure, fig, and the plotting area, ax.
fig_loss, ax_loss = plt.subplots(figsize=(7, 4))  # figsize is measured in inches
shown = min(81, len(history_complete))           # 0 through 80 completed updates
ax_loss.plot(np.arange(shown), history_complete[:shown])
ax_loss.set(xlabel="Completed updates", ylabel="J = MSE / 2",
            title="Gradient descent on four samples")
ax_loss.grid(alpha=0.25)           # alpha: opacity of the grid lines
fig_loss.tight_layout()           # Adjust margins to prevent overlap of the title and axis labels

# 7. Compare learning rates using the separate function J(t)=t²/2
# Here, t is a single number, separate from theta in the original regression model.
for rate in [0.2, 0.8, 1.5, 2.2]:
    t = 3.0
    t_history = [t]              # Start recording at the state with zero completed updates
    for _ in range(8):
        t -= rate * t           # The derivative of this function at t is t
        t_history.append(t)
    losses = np.array(t_history) ** 2 / 2
    print("Learning rate", rate, "t/J after the first update:", t_history[1], losses[1],
          "J after 8 updates:", losses[-1])

# 8. Apply least squares and scikit-learn to the same small dataset
# Four returned values: coefficients, array of residual sums of squares, rank, singular values.
theta_ls, residual_sums, rank, singular_values = np.linalg.lstsq(
    Xb, y, rcond=None
)
pred_ls = Xb @ theta_ls
model_small = LinearRegression()  # Default: fit_intercept=True
fit_return = model_small.fit(X, y)  # Use X before adding the column of ones
pred_sk = model_small.predict(X)
print("Least-squares coefficients:", theta_ls)
print("Other values returned by lstsq:", residual_sums, rank, singular_values)
print("Does fit return the same model object:", fit_return is model_small)
print("sklearn intercept / weights:", model_small.intercept_, model_small.coef_)
print("Compare least-squares / GD / sklearn predictions by column:\n", np.c_[pred_ls, pred_gd, pred_sk])
new_input = np.array([1.0, 3.0, 4.0])  # [1 for the intercept, x1, x2]
print("GD prediction for x1=3, x2=4:", new_input @ theta_gd)

# 9. Check the implementation: verify shapes first, then compare numbers within a tolerance
assert theta_gd.shape == (3,)
assert pred_gd.shape == y.shape
assert np.isfinite(theta_gd).all()  # Check that every coefficient is neither NaN nor infinite
assert np.isfinite(history_complete).all()
assert np.isclose(j0, 7.75, atol=1e-12, rtol=0)
assert np.allclose(g0, [-3.5, -2.25, -2.5], atol=1e-12, rtol=0)
assert np.allclose(theta_gd, [1.0, 2.0, 3.0], atol=1e-8, rtol=0)
assert np.allclose(pred_gd, pred_ls, atol=1e-8, rtol=0)
assert np.allclose(pred_gd, pred_sk, atol=1e-8, rtol=0)
assert final_j < history_complete[0]
print("Implementation checks passed. Maximum GD/least-squares prediction difference:", np.max(np.abs(pred_gd - pred_ls)))

# 10. Real data: stored feature values without the additional scaling offered by the loader
X_real, y_real = load_diabetes(return_X_y=True, scaled=False)
# Split the row indices and apply the same indices to both arrays.
all_ids = np.arange(len(y_real))
train_idx, test_idx = train_test_split(
    all_ids, test_size=0.2, random_state=42
)
X_train, X_test = X_real[train_idx], X_real[test_idx]
y_train, y_test = y_real[train_idx], y_real[test_idx]
assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0] == y_test.shape[0]
print("Full / training / test input shapes:", X_real.shape, X_train.shape, X_test.shape)

# 11. Baseline: predict the mean of the training targets for every test row
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)
pred_baseline = baseline.predict(X_test)

# 12. Regression using only the BMI feature. Select with [2] to retain the column dimension.
model_bmi = LinearRegression()
model_bmi.fit(X_train[:, [2]], y_train)
pred_bmi = model_bmi.predict(X_test[:, [2]])

# 13. Regression using all ten features. This is a separate object from the BMI model above.
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

# 14. Residual = actual value - prediction, the opposite sign from e used during training.
residual = y_test - pred_all
print("Original row index of the first test sample:", test_idx[0])
print("First test target / prediction / residual:", y_test[0], pred_all[0], residual[0])
fig_residual, ax_residual = plt.subplots(figsize=(7, 4))
ax_residual.scatter(pred_all, residual, alpha=0.65, label="Test samples")
ax_residual.scatter([pred_all[0]], [residual[0]], color="crimson",
                    s=90, label="First test sample")  # s: marker area
ax_residual.axhline(0, color="black", linewidth=1)  # Reference line at residual 0
ax_residual.set(xlabel="Prediction", ylabel="Residual = target - prediction",
                title="Residuals on 89 held-out samples")
ax_residual.legend()
fig_residual.tight_layout()
plt.show()                       # Display the two prepared figures
