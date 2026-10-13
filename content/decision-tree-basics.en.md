+++
title = "How a Decision Tree Learns and Predicts: From Gini Impurity to Pruning"
description = "Follow how a decision tree chooses split conditions and makes predictions. Calculate Gini and entropy with six samples, then interpret depth limits, pruning, validation accuracy, and feature importance in a penguin classification experiment."
date = 2026-10-13T18:00:00+09:00
slug = "decision-tree-basics"

[extra]
katex = true
toc = true
styles = ["decision-tree-basics/article.css"]
+++

Suppose we want to identify a penguin's species from measurements such as bill length and body mass. A decision tree applies conditions to these inputs in sequence. During training, it uses labeled data to decide **which feature to compare with which value**. During prediction, it **applies the learned conditions to a new input**.

Creating ever finer partitions is not the objective by itself. A tree that perfectly separates its training data can still misclassify a new penguin. We therefore need to distinguish the calculation that selects a condition from the evaluation of a trained model.

We begin with inputs, training, prediction, and tree structure. Next, we calculate splits using six fictional samples, then read a penguin classification experiment to see how tree size and validation accuracy change. The final section provides the complete calculation dataset and executable code.

## 1. Separate inputs from targets, then training from validation

A **sample** is one observed subject. In penguin data, one row represents one penguin. A **feature** is information supplied to the model, a **class** is a category to predict, and a **label** is the recorded class assigned to a sample.

Labels stored as 0, 1, and 2 are category codes. They do not mean incorrect and correct. The mapping from each code to a real category must come from the dataset's definition.

We use `X` for inputs and `y` for actual targets, adding `train` and `val` to distinguish their roles. With 204 training samples, 69 validation samples, and four features per sample:

| Name | Contents | Shape |
| --- | --- | --- |
| `X_train` | Features of the training penguins | `(204, 4)` |
| `y_train` | Actual labels of those same penguins | `(204,)` |
| `X_val` | Features of the validation penguins | `(69, 4)` |
| `y_val` | Actual labels of those same validation penguins | `(69,)` |

**`shape` gives the size of each dimension.** `(204, 4)` is a two-dimensional input with 204 rows and four columns. `(204,)` is a one-dimensional array of 204 labels; the trailing comma does not mean an omitted second dimension.

`X_train` and `y_train` are not training data and validation data. They are **the inputs and targets of the same training subjects**. The model partitions samples using inputs, then checks their labels to assess how well a split separates classes. That is why `fit(X_train, y_train)` needs both.

The first input row and first target must refer to the same penguin. Equal lengths are not sufficient. Sorting targets independently can attach another penguin's label to an input. Converting the data to an array does not repair such a mismatch.

### Fix the purpose of each dataset first

**Training** determines the values or structure needed for prediction from data. **Validation** evaluates candidate models to help choose a configuration. **Testing** is the final evaluation on separately held data after the configuration has been chosen.

```text
Prepare inputs X and actual targets y.
    ↓
Separate training, validation, and final test data.
    ↓
Create a model and specify its settings.
    ↓
fit: learn split conditions and tree structure from training data.
    ↓
predict: apply those conditions to validation inputs.
    ↓
accuracy_score: compare predictions with validation targets.
    ↓
Compare validation performance and tree size to choose a configuration.
    ↓
Evaluate final performance on the separate test set.
```

Validation inputs must contain the same features in the same column order as training inputs. When preprocessing is involved, apply the criteria learned from training data to validation data. Learning preprocessing criteria from held-out evaluation data introduces **data leakage**: evaluation information enters the training process.[^pitfalls]

## 2. Model creation, training, and prediction are different operations

`DecisionTreeClassifier` is a **Python class** that creates a classification-tree object. An object is an individual model that we actually use. A function belonging to an object is a **method**, and a value it exposes is an **attribute**. A Python class is different from a prediction class such as a species.

| Code | Operation | Return value |
| --- | --- | --- |
| `DecisionTreeClassifier(...)` | Create a configured model | An untrained model object |
| `model.fit(X_train, y_train)` | Build a tree from training inputs and targets | The fitted model itself |
| `model.predict(X_val)` | Predict validation labels using learned conditions | An array of predicted labels |
| `accuracy_score(y_val, val_pred)` | Compare actual and predicted labels | An accuracy number with default settings |

No training data are supplied when the model is merely created. `fit` uses data to determine conditions. Its return value is the model itself, not an evaluation result. Predictions come from `predict` and are passed separately to an evaluation function.[^classifier]

### A decision tree does not learn through gradient descent

**Gradient descent** gradually updates learnable numbers using the gradient of a function that measures error. It is not the method used for all machine learning.

A decision tree compares feature-and-threshold combinations to select split conditions. The trained model stores feature indices, thresholds, connections to subsequent conditions, and class composition in each part of the data. **What it learns is a set of input-partitioning conditions and their connections.**[^tree]

`random_state=41` controls the reproducibility of random choices during training. It is useful when comparing experiments while keeping the data and other conditions fixed. `criterion="gini"` selects Gini impurity as the split criterion and is the default. Section 5 explains the calculation.

### Accuracy is a fraction obtained by comparing two sets of labels

`accuracy_score` does not receive a model object. It receives actual labels and **predictions that have already been computed**, comparing matching subjects. With default settings:[^accuracy]

$$
\text{accuracy}=\frac{\text{correct predictions}}{\text{evaluated samples}}
$$

Dividing the correct count by the total converts it to a fraction. For 67 correct predictions among 69 validation samples:

$$
\frac{67}{69}\approx0.971014\approx97.10\%
$$

Comparing target and prediction arrays with `==` produces `True` or `False` at each position. Summing the result counts correct predictions because `True` contributes 1 and `False` contributes 0.

**Training accuracy** is calculated by predicting the training inputs again. **Validation accuracy** is calculated on validation inputs not used to train that model. Their difference can be recorded as `gap = train_accuracy - validation_accuracy`. A gap of 0.05 means five percentage points. A small gap does not establish good performance when both scores are low.

## 3. Nodes, leaves, and depth count different things

A **node** is a component of a tree that stores a condition or class distribution. The starting node is the **root**. A **split node** checks a condition and selects another node. A **leaf** has no children and ends the prediction path.

A **binary tree** gives each node at most two children. Split nodes have two children and leaves have none. “Binary” describes the two-way split, not the number of target classes. A binary tree can classify three or more classes.

Consider the following tree to understand a prediction path. `x1` is one numerical input feature, and classes 0 and 1 are two target categories. The numbers assigned to nodes are **node IDs** that distinguish individual nodes.

```text
node 0: x1 <= 0.5 ?                            depth 0
│
├─ true  → node 1: predict class 0 [leaf]        depth 1
│
└─ false → node 2: x1 <= 1.5 ?                  depth 1
             │
             ├─ true  → node 3: class 0 [leaf]  depth 2
             │
             └─ false → node 4: class 1 [leaf]  depth 2
```

Node 1 is already a leaf. It is node 2 that connects to nodes 3 and 4. The three leaves are nodes 1, 3, and 4; they do not need to have the same depth. Multiple leaves can predict class 0, so **the number of leaves differs from the number of classes**.

| Quantity | Value in this tree | Meaning |
| --- | --- | --- |
| Node IDs | 0, 1, 2, 3, 4 | Identifiers for individual nodes |
| Predicted classes | 0, 1 | Target categories |
| Number of leaves | 3 | Number of terminal nodes without children |
| Actual maximum depth | 2 | Number of connections from the root to the deepest leaf |

The number of nodes or leaves does not reveal the number of training samples. Multiple samples can belong to the same leaf.

### Depth starts at zero for the root

Depth is the **number of connections** from the root to a node. The root has depth zero because no connection has been crossed yet.

`max_depth=2` permits up to three levels including the root; `max_depth=3` permits up to four. Each extra depth can split every node into at most two children, so the maximum possible leaf count grows as `1 → 2 → 4 → 8`. For maximum depth $d$, the upper bound is $2^d$ leaves.

`max_depth` is an **allowed upper bound**, while `model.get_depth()` returns the **actual maximum depth**. Setting the limit to 4 does not force the model to reach it. `model.get_n_leaves()` returns the actual leaf count.[^classifier]

`max_depth=None` means no explicit depth limit. Splitting still stops at pure nodes, where no valid split exists, or when minimum-sample requirements prevent a split. The actual depth can be zero, but the `max_depth` parameter accepts positive integers or `None`.

## 4. Follow the prediction path for one input

Putting `x1=1` into the preceding tree gives this sequence:

| Step | Condition | Result |
| --- | --- | --- |
| 1 | At node 0, `1 <= 0.5` | False: select right child, node 2. |
| 2 | At node 2, `1 <= 1.5` | True: select left child, node 3. |
| 3 | Node 3 is a leaf. | Finish with its stored class distribution. |

Moving to a node means **selecting the node used for the next calculation**, not relocating the stored input data. Going left does not always imply class 0. The direction follows the condition; the prediction comes from the destination leaf.

A class distribution of `{0: 0.5, 1: 0.5}` at node 3 means the two classes have equal proportions there. Without training weights, these are the proportions of training samples reaching that leaf. **It does not mean 50% validation accuracy.** Proportions alone also do not reveal the leaf's sample count.

When the highest proportions tie, this classifier selects the earlier class in `classes_`. With class order `[0, 1]`, it predicts 0. For an input whose actual label is 1, that prediction would be incorrect.[^classifier]

### Predicted classes and node information are different outputs

| Operation | What it returns or displays |
| --- | --- |
| `export_text(model, ...)` | A string describing the whole tree's conditions and leaf predictions |
| `model.predict(X)` | A predicted class for each input |
| `model.predict_proba(X)` | Class probabilities for each input |
| `model.apply(X)` | The destination leaf's node ID for each input |
| `model.decision_path(X)` | A matrix showing which nodes each input uses |

An `apply` result of `[3]` identifies **leaf node 3**, not prediction class 3. `predict_proba` returns shape `(number of samples, number of classes)`, with class order given by `classes_`.[^structure]

In the `decision_path` matrix, rows are input samples and columns are node IDs. A used node is marked 1; an unused node is marked 0. The path `0 → 2 → 3` produces:

```text
node ID        0  1  2  3  4
node used     [1, 0, 1, 1, 0]
matrix shape  (1, 5)
```

A matrix with many zeros is a **sparse matrix**. `decision_path` uses the CSR format, which stores nonzero values and their locations. For one input, `.indices` identifies the nodes recorded in that row. With multiple inputs, the entire `.indices` array must not be read as one input's path. The column-index list is not, by itself, a chronological execution log.[^csr]

`.toarray()` expands it into an ordinary array. This is convenient for small examples but can consume substantial memory for large matrices. Child connections can be inspected through `model.tree_.children_left[node_id]` and `children_right`. A child ID of `-1` indicates no child.

### Keep even a single input two-dimensional

pandas `.iloc` selects by integer **position**. Positions begin at zero, and slices include their start but exclude their end. Selecting the second sample from validation data with four features gives these differences:[^indexing]

| Selection | Result | Shape |
| --- | --- | --- |
| `X_val.iloc[1]` | A Series containing the second row's values | `(4,)` |
| `X_val.iloc[1:2]` | A DataFrame with the second row only | `(1, 4)` |
| `val_pred[1:2]` | An array with the corresponding prediction | `(1,)` |

`predict` expects a two-dimensional input shaped `(samples, features)`, so keep one row as `(1, 4)`. Once all predictions have been computed, the matching prediction can be selected directly, provided the model has not been refitted and the input order has not changed.

A pandas Series index label is different from an array position. In a Series with integer labels, `y_val[1]` selects label 1, while `y_val.iloc[1]` selects the second position. In a NumPy array, `[1]` selects the second position.

The earlier path calculation assumes no missing values. When an input contains `NaN`, or displayed thresholds have been rounded, do not infer the actual branch solely from a displayed comparison. Check the model's inspection results.

## 5. Gini impurity measures class mixing

Prediction followed stored conditions. We now return to training and ask **which conditions should be stored**. The model needs a numerical criterion to compare the class composition produced by candidate splits.

**Gini impurity** measures whether actual labels at a node are concentrated in one class or mixed across several. It is not an error rate computed by comparing predictions with targets.

Let $K$ be the number of classes and $k$ a class index from 0 through $K-1$. Let $n_k$ be the number of class-$k$ samples at the current node, and $n$ its total sample count. Assume $n>0$ and no sample or class weights. The symbol $\sum$ means to add the specified terms.

The class proportion $p_k$ and Gini impurity $G$ are:[^tree]

$$
p_k=\frac{n_k}{n},\qquad
G=1-\sum_{k=0}^{K-1}p_k^2
$$

**There is one term per class, not one term per sample.** Two classes produce two terms; three classes produce three.

### Why square the proportions instead of simply adding them?

Class proportions always sum to 1. Their sum alone cannot distinguish a mostly single-class node from a balanced mixture.

Squaring multiplies a value by itself. For $0<p<1$, $p^2<p$, so the proportion itself becomes smaller. For example, $9/10=90/100$, but its square is $81/100$. Both numerator and denominator change.

At the same time, the relative ratio between unequal positive numbers becomes more pronounced. For $a>b>0$:

$$
\frac{a^2}{b^2}=\left(\frac{a}{b}\right)^2>\frac{a}{b}
$$

Because $a/b>1$, squaring that ratio makes it larger. This is different from claiming that the absolute difference between the two numbers always grows.

Compare class proportions constrained to sum to 1:

| Class proportions | Sum of squares | Gini: one minus that sum |
| --- | --- | --- |
| `[0.9, 0.1]` | $0.81+0.01=0.82$ | $0.18$ |
| `[0.5, 0.5]` | $0.25+0.25=0.50$ | $0.50$ |

The more concentrated distribution has the larger sum of squares. Subtracting it from 1 gives a smaller score for less mixing. Gini is zero when every sample belongs to one class. With $K$ equally represented classes, its maximum is $1-K(1/K)^2=1-1/K$. Thus **0.5 is the maximum for two classes**, not for every classification problem.

**Mean squared error, or MSE**, also uses squaring, but squares the difference between predictions and actual values. Gini squares class proportions. Using the same operation does not make them evaluations of the same quantity. Gini is also not the only impurity criterion; entropy provides an alternative.

### Split six fictional samples

The following fictional data make the split calculation explicit. `x1` and `x2` are numerical features without assigned physical units; 0 and 1 are the two target categories. We will calculate splits of this table separately from the illustrative prediction tree above.

| Sample ID | x1 | x2 | Actual class |
| --- | --- | --- | --- |
| 1 | 1 | 1 | 0 |
| 2 | 2 | 2 | 0 |
| 3 | 3 | 4 | 0 |
| 4 | 4 | 3 | 1 |
| 5 | 5 | 5 | 1 |
| 6 | 6 | 6 | 1 |

For `x2 <= 2.5`, samples 1 and 2 go left; samples 3, 4, 5, and 6 go right. Writing counts in class order 0, 1 gives `[2, 0]` on the left and `[1, 3]` on the right.

The right side's total is 4, but Gini requires the **per-class counts `[1, 3]`**:

$$
G_R
=1-\left[\left(\frac14\right)^2+\left(\frac34\right)^2\right]
=1-\left(\frac1{16}+\frac9{16}\right)
=\frac6{16}
=0.375
$$

The fractions $1/4$ and $3/4$ are the proportions of classes 0 and 1. The three class-1 samples are already accounted for by the numerator of $3/4$. Adding $(3/4)^2$ three times would count the same class proportion repeatedly.

Both left-hand samples are class 0, so:

$$
G_L=1-\left[\left(\frac22\right)^2+\left(\frac02\right)^2\right]=0
$$

## 6. Weight child impurities by their sample counts

There are two samples on the left and four on the right. Adding the two Gini values and dividing by 2 would give equal influence to differently sized groups. Evaluating the overall mixture requires accounting for how much data each group contains.

A **weighted average** incorporates the appropriate contribution of each value. Let $n_L$ and $n_R$ be the left and right sample counts, with $n=n_L+n_R$. Weight the child impurities $G_L$ and $G_R$ by those counts and divide by the total:

$$
\text{weighted Gini}
=\frac{n_LG_L+n_RG_R}{n}
=\frac{n_L}{n}G_L+\frac{n_R}{n}G_R
$$

Division determines each child's share of the data. Multiplication applies that share to its impurity. Addition combines the contributions.

For the six samples just partitioned:

$$
\text{weighted Gini}
=\frac26\times0+\frac46\times0.375
=0.25
$$

The order is **class counts → class proportions → child impurities → sample-weighted average**. The right-hand term uses Gini, not the sum of squared proportions $1/16+9/16$. Subtract that sum from 1 first.

Another candidate, `x2 <= 3.5`, gives `[2, 1]` on the left and `[1, 2]` on the right. Both impurities are $1-(4/9+1/9)=4/9$, and both children contain three samples. Their weighted average is also $4/9\approx0.4444$.

For the same parent, **`x2 <= 2.5`, with 0.25, reduces mixing more than the candidate with 0.4444**. This compares training split criteria, not validation accuracy.

## 7. Compare distinct partitions rather than infinitely many real numbers

A threshold could be changed from 2.1 to 2.11 to 2.111 indefinitely. A decision tree does not test every real number. It compares **ways of partitioning the current training samples differently**.

Consider a numerical feature without missing values and the default `splitter="best"`. Sorting the table's `x2` values gives `[1, 2, 3, 4, 5, 6]`. Both `x2 <= 2.1` and `x2 <= 2.9` send values 1 and 2 left, producing the same training partition and impurity.

For distinct adjacent values $a<b$, thresholds satisfying $a\le t<b$ produce the same training partition under $x\le t$. The midpoint $(a+b)/2$ is usually used as a representative. With $m$ distinct values, there are at most $m-1$ boundaries that leave both sides nonempty.

Identical training partitions do not imply identical predictions for every future input. A new value between 2.1 and 2.9 may go to a different side depending on which threshold was stored.

### Move the boundary and update counts

Sorting by `x2` changes the sample order to `1, 2, 4, 3, 5, 6`. Labels must follow those samples, becoming `[0, 0, 1, 0, 1, 1]`. Sorting the feature while leaving labels unchanged would calculate a different dataset.

Start with left counts `[0, 0]` and right counts `[3, 3]`. Each time the boundary moves right, update the class count of the sample newly added to the left.

| Threshold | Left class counts | Right class counts | Weighted Gini |
| --- | --- | --- | --- |
| 1.5 | `[1, 0]` | `[2, 3]` | 0.4000 |
| 2.5 | `[2, 0]` | `[1, 3]` | 0.2500 |
| 3.5 | `[2, 1]` | `[1, 2]` | 0.4444 |
| 4.5 | `[3, 1]` | `[0, 2]` | 0.2500 |
| 5.5 | `[3, 2]` | `[0, 1]` | 0.4000 |

This avoids recounting all labels from scratch for every candidate. Computing class-dependent scores still has a cost. The implementation also applies further constraints and optimizations, and sorting does not occur only once throughout the entire training process.[^implementation]

The table compares candidates for `x2` alone. The model compares the **feature-and-threshold combinations** it examines. On the same data, `x1 <= 3.5` creates `[3, 0]` and `[0, 3]`, giving weighted Gini 0. The best score of 0.25 for `x2` is therefore not the best across all features.

The tree chooses a condition at the current node, then repeats the process at its children. It does not exhaustively compare every possible future tree structure to guarantee a global optimum.[^tree]

## 8. Entropy and information gain provide another split criterion

**Entropy** is another measure of uncertainty in class composition. `criterion="gini"` compares splits using Gini, while `criterion="entropy"` compares them by entropy reduction. These are not two stages in which entropy grades an already calculated Gini score.

To understand entropy, first connect logarithms with information. A **logarithm** reverses exponentiation: $\log_2(p)=r$ means $2^r=p$. Define the **information content** of an outcome with probability $p$ as:[^entropy]

$$
I(p)=-\log_2(p),\qquad 0<p\le1
$$

A certain outcome has $p=1$ and information content zero; rarer outcomes have more information. For $0<p<1$, the logarithm is negative, so the minus sign makes information nonnegative. When independent outcomes have a joint probability expressed as a product, logarithms express their information as a sum:

$$
-\log_2(pq)=-\log_2(p)-\log_2(q)
$$

Entropy $H$ is the **weighted average** of class information, weighted by each class's proportion. Using the previously defined $p_k$ and class indices:

$$
H=\sum_{k=0}^{K-1}p_kI(p_k)
=-\sum_{k=0}^{K-1}p_k\log_2(p_k)
$$

The logarithm converts probability to information. Multiplication by proportions and summation average that information. A term with $p_k=0$ contributes zero; do not evaluate `log2(0)` directly.

### Information gain is the difference before and after splitting

**Information gain** is parent entropy minus the sample-weighted average of child entropies. It is larger when the split removes more uncertainty.

The six-sample parent has counts `[3, 3]`. Both class proportions are $1/2$, and $\log_2(1/2)=-1$, so parent entropy is 1. For `x2 <= 2.5`, the left child `[2, 0]` has entropy zero. The right child `[1, 3]` has:

$$
H_R=-\left[\frac14\log_2\left(\frac14\right)+\frac34\log_2\left(\frac34\right)\right]
\approx0.811278
$$

Weighted entropy is $(2/6)\times0+(4/6)\times0.811278\approx0.540852$. Information gain is $1-0.540852\approx0.459148$.

| Condition | Weighted Gini | Weighted entropy | Information gain |
| --- | --- | --- | --- |
| `x2 <= 3.5` | 0.4444 | 0.9183 | 0.0817 |
| `x2 <= 2.5` | 0.2500 | 0.5409 | 0.4591 |

Lower post-split Gini or entropy is better; higher information gain is better. Both criteria prefer the same candidate in this comparison, but they do not always select the same conditions on other data.

## 9. Depth limits and pruning simplify trees differently

A **hyperparameter** specifies how training operates or what limits it obeys. `max_depth` and `ccp_alpha` are hyperparameters. Actual split features, thresholds, connections, and class distributions are learned from data by `fit`.

`max_depth` limits growth while the tree is being built. **Pruning** removes the structure below a split node and turns that node into a leaf. It does not delete the original training samples.

`ccp_alpha` controls **cost-complexity pruning**. This method assigns a cost to leaf count as well as training impurity. Let $T$ be a candidate tree, $R(T)$ the leaf impurities summed with weights based on their shares of all training samples, and $L(T)$ the number of leaves. Let $\alpha$ be the complexity cost per leaf.[^pruning]

$$
\text{cost}(T)=R(T)+\alpha L(T)
$$

Adding the same cost $\alpha$ for every leaf produces the multiplication $\alpha L(T)$. Adding training impurity gives the total comparison criterion. For the same starting tree, larger $\alpha$ places more cost on a many-leaf structure and favors simpler trees.

`ccp_alpha=0` does not perform this pruning by default. Setting it to 0.01 sets alpha in the cost expression to 0.01; it does not mean removing one percent of branches. Structure changes at particular boundaries, so a small increase can leave the result unchanged.

**Training does not inspect validation accuracy to decide which branches to remove.** Pruning uses training impurity and leaf count. Comparing alpha candidates on validation data happens outside that procedure.

## 10. Compare validation accuracy together with leaf count

We now move from the six calculation samples to a penguin classification experiment. It uses 204 training samples, 69 validation samples, and four features. The data split and `random_state=41` remain fixed while the settings change.

These tables are a case study in interpreting recorded experiment results. Original rows and split information are not included, so the fictional-data code in Section 12 does not reproduce these scores.

### Changing the depth limit

| Depth limit | Actual depth | Leaves | Validation accuracy |
| --- | --- | --- | --- |
| 1 | 1 | 2 | 0.782609 |
| 2 | 2 | 4 | 0.913043 |
| 4 | 4 | 9 | 0.971014 |
| `None` | 4 | 9 | 0.971014 |

Limits of 4 and `None` tie for the best validation accuracy among these candidates. Removing the explicit limit still produces a tree of depth 4. On these data, permitting more depth did not make the tree deeper. This does not imply that depth 4 is sufficient for every problem.

For single-output classification without sample or class weights, exactly 100% training accuracy means that each leaf correctly predicts all training samples reaching it with one class. Those leaves contain no class mixture. The model did not stop because validation accuracy was about 97%; it stopped according to training data and stopping conditions.

### Changing the pruning alpha

| Alpha | Actual depth | Leaves | Training accuracy | Validation accuracy |
| --- | --- | --- | --- | --- |
| 0 | 4 | 9 | 1.000000 | 0.971014 |
| 0.005 | 4 | 9 | 1.000000 | 0.971014 |
| 0.01 | 4 | 6 | 0.980392 | 0.927536 |
| 0.02 | 2 | 4 | 0.965686 | 0.913043 |
| 0.05 | 2 | 3 | 0.955882 | 0.913043 |

Increasing alpha from 0 to 0.01 reduces leaves from 9 to 6. Convert validation accuracy back to counts:

$$
0.971014\approx\frac{67}{69},\qquad
0.927536\approx\frac{64}{69}
$$

**The tree loses three leaves, but also makes three fewer correct validation predictions.** The accuracy difference is approximately $(3/69)\times100=4.35$ percentage points. Structure became simpler while accuracy on this validation set decreased.

Alpha 0 and 0.005 tie for the highest validation accuracy; zero is not uniquely best. Alpha 0.02 and 0.05 also tie in validation accuracy, but 0.05 has fewer leaves. Between those two candidates, there is a basis for preferring the simpler model at the same observed validation performance.

Matching depth, leaf count, and accuracy does not prove that all split conditions are identical. Summary metrics and the complete tree structure are different information.

### Do not generalize beyond a single validation result

**Overfitting** occurs when detailed rules adapted to training data do not work well on new data. This is why 100% training accuracy does not guarantee 100% accuracy on new samples. A training-validation gap alone does not establish the cause either.

With 69 validation samples, one extra correct prediction changes accuracy by roughly $(1/69)\times100=1.45$ percentage points. Check whether the result holds under other appropriate splits.

**Cross-validation** repeatedly changes the portions used for training and evaluation. `GridSearchCV` compares candidate setting combinations this way. Time-ordered data and repeated observations of the same subject require splits that reflect actual use. Repeatedly changing settings after inspecting test scores also turns the test set into selection data.[^selection]

Candidate alphas can be calculated directly. `cost_complexity_pruning_path(X_train, y_train)` computes pruning boundaries for the supplied training data and settings; `ccp_alphas` in the result contains the candidate array. It does not return the alpha with the best validation accuracy. Candidates must be trained and evaluated separately, and changing other settings such as depth can change the candidates.[^pruning]

For this experiment, prioritizing validation accuracy favors alpha 0 or 0.005 among the tested candidates. That conclusion is limited to the current split and candidate set, not all possible alphas.

## 11. Feature importance describes contribution within the current tree

`feature_importances_` measures **feature contribution to impurity reduction**. Each split's impurity reduction is weighted by the node's share of training samples. Contributions are summed for each feature, then divided by the total contribution across features to form relative proportions.[^importance]

If no impurity reduction occurs, every value can be zero. Do not assume the sum must always be one. This is an attribute rather than a method, so read `model.feature_importances_` without parentheses.

For example, if `flipper_length_mm` ranks highest, splits using flipper length contributed most to weighted impurity reduction in the current tree. The suffix `mm` denotes millimeters.

| Interpretation | Why the importance does not establish it |
| --- | --- |
| Importance 0.55 means this feature alone predicts 55% correctly. | It is not validation accuracy using that feature alone. |
| Removing this feature lowers accuracy by 55%. | It does not measure retraining after feature removal. |
| Increasing the feature makes the prediction class 1. | Importance does not encode prediction direction. |
| The feature causes the penguin's species. | Importance does not establish a real-world causal effect. |

Conditions and predictions for individual inputs can show how the model responds. That response is still different from a causal relationship in the world.

An importance of zero means no impurity-reduction contribution is assigned to that feature in this model. Often the feature is unused, but it can also appear in a split with zero reduction. Actual use can be checked through feature indices in `model.tree_.feature`.

Impurity-based importance can favor features with many distinct values. Features carrying similar information may share contribution or concentrate it in one of them. To investigate validation usefulness, retrain without a column or separately measure the score change when its validation values are shuffled. These checks do not automatically establish causality either.[^importance]

### Match feature names in training-column order

Values in `feature_importances_` follow the training-column order. `zip(feature_names, importance)` creates an **iterator** that yields corresponding values as **tuples**. Collecting them with `list(...)` produces `[(feature name, importance), ...]`.

`zip` does not match names semantically. In Python 3.10 or later, `strict=True` raises an error for unequal lengths, but it cannot check whether names are in the correct order.[^python]

`sorted(pairs, key=lambda pair: pair[1], reverse=True)` returns a new list ordered from highest importance to lowest. `pair` is one tuple; `pair[0]` is the name and `pair[1]` the importance. `lambda pair: pair[1]` is a short function that supplies the sort key. The sorted order is not necessarily the original training-column order.

## 12. Run split calculations and model inspection on six samples

The following code starts with the fictional dataset in Section 5 and runs Gini and entropy calculations, threshold comparisons, model training, and prediction. It requires Python 3.10 or later with NumPy, pandas, and scikit-learn.

`proportions`, `gini`, `entropy`, and `compare_settings` are functions defined below. Their inputs and outputs are:

| Function | Input | Return value |
| --- | --- | --- |
| `proportions` | Per-class counts | An array of proportions of the same length |
| `gini`, `entropy` | Per-class counts | A numerical impurity value |
| `compare_settings` | Training inputs and targets, validation inputs and targets | A DataFrame with one row per setting |

`compare_settings` accepts training and validation data that have already been separated. Defining the function does not run training. The six-sample execution below trains the calculation model on all six samples, so **its printed accuracy is training accuracy, not validation performance**.

Inside the comparison function, `settings` contains `(experiment name, parameter dictionary)` pairs. `**parameters` supplies dictionary entries as keyword arguments such as `max_depth=...`. Each iteration trains a fresh model and adds a result dictionary with `rows.append(...)`. One table row therefore represents a model experiment, not an original sample.

<details class="dt-code">
<summary>Dataset and complete Python code</summary>

```python
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text


def proportions(counts):
    # Convert per-class counts to floats while preserving their order.
    counts = np.asarray(counts, dtype=float)
    # ndim is the number of dimensions; isfinite rejects NaN and infinity.
    if counts.ndim != 1 or not np.all(np.isfinite(counts)):
        raise ValueError("counts must be a finite one-dimensional sequence.")
    total = counts.sum()
    if np.any(counts < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError("Counts must be nonnegative with a finite positive total.")
    return counts / total


def gini(counts):
    p = proportions(counts)
    return float(1 - np.sum(p ** 2))  # Square each class proportion once.


def entropy(counts):
    p = proportions(counts)
    positive = p[p > 0]  # Avoid evaluating log2(0).
    return float(-np.sum(positive * np.log2(positive)))


def compare_settings(X_train, y_train, X_val, y_val):
    # All data are arguments; this function does not depend on external datasets.
    rows = []
    settings = [(f"depth={depth}", {"max_depth": depth})
                for depth in [1, 2, 4, None]]
    settings += [(f"alpha={alpha}", {"ccp_alpha": alpha})
                 for alpha in [0, 0.005, 0.01, 0.02, 0.05]]
    for name, parameters in settings:
        # ** supplies the dictionary entries as keyword arguments.
        candidate = DecisionTreeClassifier(random_state=41, **parameters)
        candidate.fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, candidate.predict(X_train))
        validation_accuracy = accuracy_score(y_val, candidate.predict(X_val))
        # append changes the list; do not assign its return value to rows.
        rows.append({"setting": name,
                     "depth": candidate.get_depth(),
                     "leaves": candidate.get_n_leaves(),
                     "train_accuracy": train_accuracy,
                     "validation_accuracy": validation_accuracy,
                     "gap": train_accuracy - validation_accuracy})
    return pd.DataFrame(rows)


# Each row is one fictional sample. sample_id is an identifier, not a feature.
samples = pd.DataFrame({
    "sample_id": [1, 2, 3, 4, 5, 6],
    "x1": [1, 2, 3, 4, 5, 6],
    "x2": [1, 2, 4, 3, 5, 6],
    "label": [0, 0, 0, 1, 1, 1],
})
left_gini, right_gini = gini([2, 0]), gini([1, 3])
weighted_gini = (2 / 6) * left_gini + (4 / 6) * right_gini
print("Gini:", left_gini, right_gini, weighted_gini)
# assert stops on a false condition; isclose compares floats within tolerance.
assert np.isclose(weighted_gini, 0.25)

# Sorting whole rows preserves correspondence between features and targets.
ordered = samples.sort_values("x2")
x_sorted = ordered["x2"].to_numpy()
y_sorted = ordered["label"].to_numpy()
left_counts, right_counts = [0, 0], [3, 3]
count = len(ordered)
parent_entropy = entropy(right_counts)
results = []
for i in range(count - 1):
    label = int(y_sorted[i])
    left_counts[label] += 1
    right_counts[label] -= 1
    if x_sorted[i] == x_sorted[i + 1]:
        continue  # Do not split between equal feature values.
    threshold = (x_sorted[i] + x_sorted[i + 1]) / 2
    left_weight, right_weight = (i + 1) / count, (count - i - 1) / count
    weighted_gini = left_weight * gini(left_counts) + right_weight * gini(right_counts)
    weighted_entropy = left_weight * entropy(left_counts) + right_weight * entropy(right_counts)
    # Copies preserve this boundary's counts as later iterations change them.
    results.append({"threshold": threshold, "left": left_counts.copy(),
                    "right": right_counts.copy(), "weighted_gini": weighted_gini,
                    "weighted_entropy": weighted_entropy,
                    "information_gain": parent_entropy - weighted_entropy})
print(pd.DataFrame(results).to_string(index=False))
assert np.allclose([row["weighted_gini"] for row in results],
                   [0.4, 0.25, 4 / 9, 0.25, 0.4])

# This model learns the six samples, not the illustrative tree in Section 3.
X, y = samples[["x1", "x2"]], samples["label"]
model = DecisionTreeClassifier(max_depth=2, random_state=41)
model.fit(X, y)
pred = model.predict(X)
print("training accuracy:", accuracy_score(y, pred))
print("depth / leaves:", model.get_depth(), model.get_n_leaves())
print(export_text(model, feature_names=list(X.columns)))

# Keep the second sample as a two-dimensional, one-row input.
one_row = X.iloc[1:2]
print("input / prediction shape:", one_row.shape, model.predict(one_row).shape)
print("prediction:", model.predict(one_row))
print("leaf ID:", model.apply(one_row))
print("class order:", model.classes_)
print("class probabilities:", model.predict_proba(one_row))
indicator = model.decision_path(one_row)
print("node-use matrix:", indicator.toarray())  # Expand only this small row.

# Pair names in training-column order, then sort the pairs by importance.
pairs = list(zip(X.columns, model.feature_importances_, strict=True))
print("feature importance:", sorted(pairs, key=lambda pair: pair[1], reverse=True))
probe = DecisionTreeClassifier(random_state=41)
print("candidate alphas:", probe.cost_complexity_pruning_path(X, y).ccp_alphas)
```

</details>

The example was executed with scikit-learn 1.8.0, NumPy 2.3.5, and pandas 2.2.3, producing the following results:

```text
Gini: 0.0 0.375 0.25
training accuracy: 1.0
depth / leaves: 1 2

|--- x1 <= 3.50
|   |--- class: 0
|--- x1 >  3.50
|   |--- class: 1

input / prediction shape: (1, 2) (1,)
prediction: [0]
leaf ID: [1]
class order: [0 1]
class probabilities: [[1. 0.]]
node-use matrix: [[1 1 0]]
candidate alphas: [0.  0.5]
```

Although `max_depth=2` permits depth 2, the actual depth is 1 because `x1 <= 3.5` perfectly separates the two classes. The second sample has `x1=2`, reaches left leaf node 1, and is predicted as class 0. This is a different tree from the Section 4 example that reached node 3.

The output `[0, 0.5]` contains pruning boundary candidates, not an optimal alpha selected through validation. Likewise, training accuracy 1.0 means the model correctly predicts these six training samples. Assessing new-data performance requires comparing predictions with actual targets on samples not used in training.

## References

[^classifier]: scikit-learn, [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). Model creation, fitting, prediction, parameters, and tie handling.
[^tree]: scikit-learn, [Decision Trees](https://scikit-learn.org/stable/modules/tree.html). Training, splitting criteria, Gini, and entropy.
[^accuracy]: scikit-learn, [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html). Accuracy inputs and return values.
[^structure]: scikit-learn, [Understanding the decision tree structure](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html). Nodes, `apply`, and `decision_path`.
[^indexing]: pandas, [Indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html). Position-based and label-based selection.
[^csr]: SciPy, [csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html). Compressed sparse row storage.
[^implementation]: scikit-learn, [split search implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_splitter.pyx) and [class statistics and criterion implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pyx). The example loop illustrates the principle rather than reproducing the internal implementation.
[^entropy]: SciPy, [entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html). Information and entropy.
[^pruning]: scikit-learn, [Cost complexity pruning](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html) and [Minimal Cost-Complexity Pruning](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning).
[^selection]: scikit-learn, [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html). Candidate settings and cross-validation.
[^pitfalls]: scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html). Consistent preprocessing and leakage.
[^importance]: scikit-learn, [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html). Interpretation and limitations of impurity-based importance.
[^python]: Python, [Built-in functions](https://docs.python.org/3/library/functions.html). `zip` and `sorted`.
