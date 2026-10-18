"""Check decision trees, bagging, random forests, and regression boosting with four houses.

Required packages: numpy, scikit-learn.
The data are fictional and illustrate the calculations. Prices are in units of KRW 100 million.
All rows have equal weight, and the regression trees are trained using squared error.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

# Each row is one house. Column order: [near a station, renovated, parking].
# For each input, 0 means no and 1 means yes.
X = np.array([
    [0, 0, 1],  # House A
    [0, 1, 0],  # House B
    [1, 0, 0],  # House C
    [1, 1, 1],  # House D
])
y = np.array([5.0, 7.0, 11.0, 17.0])  # Actual prices in the same order.
new_X = np.array([[1, 0, 1]])  # New house E: 1 row, 3 columns. Its actual price is unknown.

print("Input and target shapes:", X.shape, y.shape)  # (4, 3), (4,)

# 1. Create one regression tree that learns the house prices themselves.
single_tree = DecisionTreeRegressor(
    criterion="squared_error",  # Evaluate candidate splits using squared error.
    max_depth=1,                # Allow only depths 0 (the root) and 1.
    random_state=42,            # Make random choices reproducible.
)
fit_result = single_tree.fit(X, y)  # Learn all the tree's conditions and leaf values.
assert fit_result is single_tree   # fit returns the model object itself.
print("Tree trained directly on prices:", single_tree.predict(X))  # [6, 6, 14, 14]

# 2. Bagging trains three models on different samples of the training data.
bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=1),
    n_estimators=3,
    bootstrap=True,    # Sample training rows with replacement.
    max_features=1.0,   # Pass all columns to each model.
    random_state=42,
)
bagging.fit(X, y)       # Each model learns the original prices of its sampled houses.
print("Number of bagging models:", len(bagging.estimators_))  # 3

# 3. In a random forest, max_features sets the number of candidate columns at a node.
forest = RandomForestRegressor(
    n_estimators=3,
    max_depth=1,
    bootstrap=True,
    max_features=2,    # Select two candidate columns for each split in this example.
    random_state=42,
)
forest.fit(X, y)

# Each tree predicts a price for the same new house. Their mean is the final prediction.
forest_outputs = np.array([tree.predict(new_X) for tree in forest.estimators_])
print("Individual forest price predictions:", forest_outputs)
print("Mean forest price prediction:", forest.predict(new_X))
np.testing.assert_allclose(forest.predict(new_X), forest_outputs.mean(axis=0))

# 4. Write the outer boosting loop directly. Train it separately from the models above.
initial_value = y.mean()  # Calculate the mean training target, 10.0.
learning_rate = 1.0      # Apply each new correction in full for this calculation.
current_prediction = np.full(y.shape, initial_value)  # [10, 10, 10, 10]
trees = []              # A list to store the trained trees.

print("Initial overall predictions:", current_prediction)
print("Initial MSE:", np.mean((y - current_prediction) ** 2))  # 21.0

# range(1, 4) provides 1, 2, and 3 in order. The final value, 4, is excluded.
for step in range(1, 4):
    residual = y - current_prediction  # Calculate each house's current residual.

    # Create a separate new tree on every iteration.
    tree = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=1,
        random_state=42,
    )
    tree.fit(X, residual)       # Train this entire new tree on the residuals.
    correction = tree.predict(X)  # Calculate four corrections, one per house.

    # Add the new tree's predictions to the existing overall predictions.
    current_prediction = current_prediction + learning_rate * correction
    trees.append(tree)         # Store the completed tree. Earlier trees are unchanged.
    mse = np.mean((y - current_prediction) ** 2)

    print(f"\nBoosting stage {step}")
    print("Residual training targets:", residual)
    print("New tree's predicted corrections:", correction)
    print("Updated overall price predictions:", current_prediction)
    print("Overall MSE:", mse)    # 5.0, 1.0, and 0.0 across the three stages.

np.testing.assert_allclose(current_prediction, y)

# 5. Inspect a trained tree's structure. Index 0 refers to the first element.
first_tree = trees[0]
print("First tree's actual depth:", first_tree.get_depth())       # 1
print("First tree's node count:", first_tree.tree_.node_count)     # 3
print("Leaf node index for each house:", first_tree.apply(X))      # [1, 1, 2, 2]

# 6. Predict a new house without fitting again or calculating its actual residual.
new_prediction = np.full(new_X.shape[0], initial_value)  # Starting value [10] for one house.
for tree in trees:
    new_prediction = new_prediction + learning_rate * tree.predict(new_X)

print("\nBoosting prediction for new house E:", new_prediction)  # [13.]
np.testing.assert_allclose(new_prediction, [13.0])

# 7. Train the library's complete boosting model with the same settings.
boosting = GradientBoostingRegressor(
    loss="squared_error",  # Set the loss that the complete model will reduce.
    n_estimators=3,        # Train three correction trees in sequence.
    max_depth=1,           # Apply this depth limit to each individual tree.
    learning_rate=1.0,
    random_state=42,
)
boosting.fit(X, y)  # Supply actual prices. The library calculates residuals internally.

print("Library prediction for the new house:", boosting.predict(new_X))  # [13.]
print("Internal tree array shape:", boosting.estimators_.shape)           # (3, 1)

# staged_predict provides the accumulated overall predictions at each stage in sequence.
for stage_prediction in boosting.staged_predict(X):
    print("Library's overall predictions at this stage:", stage_prediction)

# Check that both implementations produce the same results with these data and settings.
# assert_allclose passes if values match within tolerance and raises an error otherwise.
np.testing.assert_allclose(boosting.predict(X), current_prediction)
np.testing.assert_allclose(boosting.predict(new_X), new_prediction)
