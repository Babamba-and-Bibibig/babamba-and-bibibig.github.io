# Reading a neural network through matrix operations: XOR and additional examples.
# NumPy is the only external library required.
# A person chose these weights to make the calculations easy to inspect.
# There is no backpropagation or parameter update, so this script does not train a model.

import numpy as np  # Use the array library under the name np.

np.set_printoptions(precision=4, suppress=True)  # Only the displayed precision changes.

# np.array() creates an array from the supplied values; dtype=float stores real numbers.
# Each row is one sample. The two columns are input features x1 and x2.
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)  # (4, 2): four samples and two features.
y = np.array([0, 1, 1, 0])  # (4,): the true class index for each corresponding row.
class_names = np.array(["same", "different"])  # Define the meaning of columns 0 and 1.

# W1 rows identify input features; W1 columns identify destination hidden neurons.
# The first neuron computes x1 - x2; the second computes -x1 + x2.
W1 = np.array([
    [1, -1],
    [-1, 1],
], dtype=float)  # (2, 2): two input features and two hidden neurons.
b1 = np.zeros(2)  # np.zeros() creates a zero-filled array. Two biases, shape (2,).

# @ is matrix multiplication: pair an input row with a W1 column, multiply, then sum.
# Add the same two biases to every sample row.
Z1 = X @ W1 + b1  # (4, 2) @ (2, 2) + (2,) -> (4, 2).
# np.maximum() chooses the larger value element by element. This ReLU replaces negatives.
H = np.maximum(0, Z1)  # (4, 2) -> (4, 2). The original X remains unchanged.

# Both output neurons receive both hidden values, h1 and h2.
# The first column computes the 'same' score; the second computes the 'different' score.
W2 = np.array([
    [-1, 1],
    [-1, 1],
], dtype=float)  # (2, 2): two hidden outputs and two output neurons.
b2 = np.array([1, 0], dtype=float)  # (2,): add 1 to the first score and 0 to the second.
S = H @ W2 + b2  # (4, 2). These are class scores, not probabilities.

# argmax() returns the index of the largest value, rather than the value itself.
# axis=1 compares the class scores within each sample. (4, 2) -> (4,).
pred = S.argmax(axis=1)
pred_names = class_names[pred]  # Retrieve the four names at the predicted indices.

# max() reduces two class scores to one maximum for each sample.
# keepdims=True retains the reduced class axis with size one.
row_max = S.max(axis=1, keepdims=True)  # (4, 2) -> (4, 1).
shifted = S - row_max  # (4, 2) - (4, 1) -> (4, 2). Subtract each row's maximum.
# np.exp() applies e**value to every element. exp(0)=1 and exp(-1)≈0.3679.
exp_scores = np.exp(shifted)  # (4, 2). Intermediate values, not learned weights.
# sum(axis=1) adds the class values in each row; keepdims=True retains the sum's column.
row_sums = exp_scores.sum(axis=1, keepdims=True)  # (4, 2) -> (4, 1).
probabilities = exp_scores / row_sums  # Divide each element by its row's sum: (4, 2).

# len(y) is the number of labels, 4. np.arange(4) creates the integers [0, 1, 2, 3].
row_indices = np.arange(len(y))  # (4,): the sample row indices.
# Pair the indices at matching positions to select [0,0], [1,1], [2,1], and [3,0].
# Use the true labels y, rather than the model's predictions pred, to select probabilities.
correct_p = probabilities[row_indices, y]  # Select one value per row: (4, 2) -> (4,).
losses = -np.log(correct_p)  # np.log() applies the natural logarithm. Four losses, (4,).
mean_ce = losses.mean()  # mean() returns the arithmetic mean of four losses: one number.

# This calculation reduces problems caused by tiny probabilities rounding to zero.
# Compute log probabilities from scores, rather than taking log of stored probabilities.
log_probabilities = shifted - np.log(row_sums)  # (4, 2) - (4, 1) -> (4, 2).
stable_losses = -log_probabilities[row_indices, y]  # Select true-class log probabilities: (4,).

# .size is an array's element count. sum() adds the counts for all parameter arrays.
parameter_count = sum(a.size for a in [W1, b1, W2, b2])  # 4 + 2 + 4 + 2 = 12.

# assert stops if a condition is false. allclose() allows small floating-point differences.
# array_equal() checks array shapes and values; isclose() checks whether two values are close.
assert X.shape == (4, 2) and y.shape == (4,)
assert H.shape == (4, 2) and S.shape == (4, 2)
assert row_max.shape == (4, 1) and row_sums.shape == (4, 1)
assert np.array_equal(Z1, [[0, 0], [-1, 1], [1, -1], [0, 0]])
assert np.array_equal(H, [[0, 0], [0, 1], [1, 0], [0, 0]])
assert np.array_equal(S, [[1, 0], [0, 1], [0, 1], [1, 0]])
assert np.allclose(probabilities.sum(axis=1), 1.0)  # Each sample's probabilities sum to 1.
assert np.array_equal(pred, y)  # Every sample's predicted class must match its true class.
assert np.allclose(stable_losses, losses)
assert np.isclose(mean_ce, 0.3132616875182228) and parameter_count == 12

# Unpack each name-and-array pair, then print .shape (axis lengths) and the actual values.
for name, values in [
    ("X", X), ("Z1", Z1), ("H", H), ("S", S),
    ("row_max", row_max), ("shifted", shifted), ("exp_scores", exp_scores),
    ("row_sums", row_sums), ("probabilities", probabilities),
    ("correct_p", correct_p), ("losses", losses),
    ("log_probabilities", log_probabilities), ("stable_losses", stable_losses),
]:
    print(name, "shape=", values.shape, "\n", values)
print("Predictions:", pred, pred_names)
print("Mean cross-entropy:", mean_ce, "Parameter count:", parameter_count)
print("All XOR checks passed.")

# === Additional shape and API examples / Separate calculations ===
# The following examples are separate calculations that leave the XOR results unchanged.

# 1. max returns scores; argmax returns column indices. Map names separately.
scores3 = np.array([[2.1, 0.4, 5.0], [1.3, 4.2, 0.7]])  # (2, 3).
names3 = np.array(["cat", "dog", "rabbit"])
print("Maximum scores:", scores3.max(axis=1))  # [5.0, 4.2].
print("Maximum columns:", scores3.argmax(axis=1))  # [2, 1].
print("Predicted names:", names3[scores3.argmax(axis=1)])  # ['rabbit', 'dog'].
assert np.array_equal(scores3.argmax(axis=1), [2, 1])

# 2. Subtracting the same maximum from scores [1000, 1002] preserves softmax probabilities.
# Dividing the scores or mapping them to 0..1 changes their differences and probabilities.
comparison_scores = np.array([
    [-2.0, 0.0],  # Subtract 1002 from each score.
    [1.0, 1.002],  # Divide each score by 1000.
    [0.0, 1.0],  # Map the minimum to 0 and the maximum to 1.
])
comparison_exp = np.exp(comparison_scores)
comparison_p = comparison_exp / comparison_exp.sum(axis=1, keepdims=True)
print("Softmax after each transformation:\n", comparison_p)
# [[0.1192, 0.8808], [0.4995, 0.5005], [0.2689, 0.7311]].
assert np.allclose(comparison_p.sum(axis=1), 1.0)

# 3. Four input features -> five hidden neurons -> six hidden neurons -> three outputs.
# Both samples use the same weights at every layer. A person also chose these weights.
X_demo = np.array([[2, 3, 5, 7], [11, 13, 17, 19]], dtype=float)  # (2, 4).
W1_demo = np.array([
    [0.2, 0.4, -0.3, 0.8, 0.1],
    [-0.5, 0.2, 0.6, 0.3, -0.2],
    [0.7, -0.1, 0.5, 0.4, 0.6],
    [0.1, 0.9, -0.4, 0.2, 0.5],
])  # (4, 5): each column holds the four weights for one neuron in the first hidden layer.
b1_demo = np.zeros(5)
W2_demo = np.array([
    [0.1, -0.2, 0.3, 0.4, -0.1, 0.2],
    [0.2, 0.3, -0.1, 0.1, 0.4, -0.2],
    [-0.3, 0.1, 0.2, -0.2, 0.1, 0.3],
    [0.4, -0.1, 0.1, 0.2, 0.3, 0.1],
    [0.1, 0.2, 0.3, -0.1, -0.2, 0.4],
])  # (5, 6): connect all five hidden outputs to all six neurons in the next layer.
b2_demo = np.zeros(6)
W3_demo = np.array([
    [0.1, -0.2, 0.3],
    [0.2, 0.1, -0.1],
    [-0.1, 0.3, 0.2],
    [0.3, -0.1, 0.1],
    [0.1, 0.2, 0.4],
    [-0.2, 0.4, 0.1],
])  # (6, 3): connect all six hidden outputs to all three output neurons.
b3_demo = np.zeros(3)

H1_demo = np.maximum(0, X_demo @ W1_demo + b1_demo)  # (2, 4) @ (4, 5) -> (2, 5).
H2_demo = np.maximum(0, H1_demo @ W2_demo + b2_demo)  # (2, 5) @ (5, 6) -> (2, 6).
S_demo = H2_demo @ W3_demo + b3_demo  # (2, 6) @ (6, 3) -> (2, 3).

# Expand the calculation for the first sample's third hidden neuron (column index 2).
z_demo = (
    X_demo[0, 0] * W1_demo[0, 2]
    + X_demo[0, 1] * W1_demo[1, 2]
    + X_demo[0, 2] * W1_demo[2, 2]
    + X_demo[0, 3] * W1_demo[3, 2]
    + b1_demo[2]
)  # 2*(-0.3) + 3*0.6 + 5*0.5 + 7*(-0.4) + 0 = 0.9.
# H1 column 2 matches row 2 of the next W2, contributing to every neuron in that layer.
# ':' in W2_demo[2, :] selects all columns from that row; the result has shape (6,).
next_layer_contributions = H1_demo[0, 2] * W2_demo[2, :]
demo_parameter_count = sum(
    a.size for a in [W1_demo, b1_demo, W2_demo, b2_demo, W3_demo, b3_demo]
)  # (4*5 + 5) + (5*6 + 6) + (6*3 + 3) = 82.

assert H1_demo.shape == (2, 5) and H2_demo.shape == (2, 6)
assert S_demo.shape == (2, 3) and demo_parameter_count == 82
assert np.isclose(H1_demo[0, 2], max(0, z_demo))  # isclose() compares two values.
assert np.isclose(H1_demo[0, 2], 0.9)
for name, values in [("H1_demo", H1_demo), ("H2_demo", H2_demo), ("S_demo", S_demo)]:
    print(name, "shape=", values.shape, "\n", values)
print("H1_demo[0, 2]:", H1_demo[0, 2])
print("This value's contributions to the six next neurons:", next_layer_contributions)
print("Additional model parameter count:", demo_parameter_count)
print("All example checks passed.")
