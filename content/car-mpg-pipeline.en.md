+++
title = "Preprocessing, Cross-Validation, and Ridge Through Car MPG Prediction"
description = "Predict fuel economy from a car's weight and region of origin, connecting fit, transform, predict, data leakage, cross-validation, Grid Search, and refit. Work through standardization, loss derivatives, and L1/L2 regularization step by step."
date = 2026-10-12T19:22:35+09:00
slug = "car-mpg-pipeline"

[extra]
katex = true
toc = true
styles = ["car-mpg-pipeline/article.css"]
+++

Suppose we want to predict a car's fuel economy from its weight and region of origin. We could train a model on records of past cars and use it to predict a new car, but several questions come up along the way. Which data should determine how weight is transformed? Should validation data be transformed too? After comparing different settings, which model should make the final prediction?

This article connects those steps around one rule: <strong>learn preprocessing statistics and prediction coefficients from the training data, then apply those statistics and coefficients to evaluation data and new inputs.</strong> We begin with the whole task, then examine the table transformations and evaluation methods that recur throughout it. Later, we use equations to see why scaling and regularization affect how coefficients are learned.

The processing steps throughout this article connect to <strong>one runnable example using 16 fictional cars</strong>. We first explain the operations we need, then provide the complete Python code, including the data and function definitions, in Section 13. The example shows the order in which preprocessing, training, and evaluation fit together.

## 1. From a car's information to a predicted value

A <strong>sample</strong> is one data record. Here, each row represents one car. A <strong>feature</strong> is information supplied to the model, and the <strong>target</strong> is the actual value we want the model to predict.

| Name | Meaning in this problem | Purpose |
| --- | --- | --- |
| `weight` | The car's weight | Use its numerical size and differences as an input. |
| `origin` | A code identifying the region where the car was produced | Use the region as an input. |
| `mpg` | Fuel economy measured in miles traveled per gallon of fuel | The target the model predicts. |
| Row `index` or `row_id` | An identifier showing which car a record belongs to | Check that inputs match their targets. It is not a feature in this example. |

In this article's example, weight is measured in <strong>lb (pounds)</strong>, and codes 1, 2, and 3 identify fictional regions of production. When working with real data, check the data description for the weight unit and the region represented by each code.

We write the input table as `X`, actual fuel economy as `y`, and the model's predicted fuel economy as `y_pred`, or $\hat y$ in equations. <strong>The actual value `y` and the prediction `y_pred` are different values.</strong> Comparing them tells us how far the prediction is from the target.

If we include `mpg` in `X` even though it is unknown at prediction time, training uses information the model will not receive in actual use. We therefore separate `weight` and `origin` as inputs and `mpg` as the target.

### First fix the roles of the training and evaluation data

<strong>Training</strong> learns the relationship between inputs and targets. <strong>Validation</strong> measures prediction error on data that the particular model did not use for training. <strong>Cross-validation (CV)</strong> repeats this process while changing the training and validation portions within the training data. After choosing a setting, <strong>refit</strong> means training again on all of the training data.

The names below keep the following meanings throughout this article.

| Name | What it contains | When it is used |
| --- | --- | --- |
| `train_frame`, `y_train` | Inputs and targets used to build the model | Cross-validation, setting selection, and final refitting |
| Training portion of a CV split | The part of `train_frame` used for training in the current split | Calculate preprocessing statistics and coefficients for this split |
| Validation portion of a CV split | The part of `train_frame` held out for evaluation in the current split | Calculate prediction error for this split |
| `valid_frame`, `y_valid` | Inputs and targets held out from the beginning | A separate final evaluation after choosing settings |
| `sample_request` | Inputs for a new car whose actual fuel economy is not yet known | Predict with the fitted model |

Although its name contains `valid`, `valid_frame` is <strong>separate from the validation portion inside cross-validation</strong>. Keeping this distinction prevents us from confusing the scores.

In the workflow below, `transform` applies the saved preprocessing statistics to the inputs, and `predict` calculates predictions using the learned coefficients. <strong>Mean squared error (MSE)</strong> is the average of the squared differences between predicted and actual values. A smaller MSE means smaller prediction error when we compare the same evaluation data and use the same target units.

<figure class="mpg-figure">
  <img src="/car-mpg-pipeline/workflow-en.svg" alt="Past car records are split into training data and final evaluation data. Within the training data, cross-validation compares candidate settings, and the selected setting is refitted on all training data. Final evaluation data and new cars are predicted using the saved preprocessing statistics and coefficients." loading="lazy" />
  <figcaption>Final evaluation data are used neither to select settings nor to refit the model. Prediction error can be calculated only when actual targets are available.</figcaption>
</figure>

## 2. fit learns the statistics; transform applies them

<strong>Preprocessing</strong> prepares the representation of the inputs supplied to a model. Here, it adjusts the numerical scale of weight and converts region codes into several columns that the model can use. Preprocessing that changes the numerical reference or scale is called <strong>scaling</strong>.

`StandardScaler` is a <strong>class</strong> that creates a scaling object. Calling `StandardScaler()` creates the object, but it has not yet calculated a mean from any car data.

A function called through an object is a <strong>method</strong>. The behavior of `fit`, `transform`, and `predict` depends on the object on which we call them.[^scaler][^pipeline]

| Call | What it does | Return value in this example |
| --- | --- | --- |
| `scaler.fit(X_train)` | Calculate and store transformation statistics, such as means and variances, from the training inputs. | The scaler object itself, with the learned statistics stored |
| `scaler.transform(X)` | Transform inputs using the stored statistics. | The transformed array |
| `scaler.fit_transform(X_train)` | Learn the statistics, then use them to transform the training inputs. | The transformed training array |
| `model.fit(X_train, y_train)` | Learn coefficients describing the relationship between inputs and targets. | The fitted model object itself |
| `model.predict(X_new)` | Calculate predictions using the stored coefficients. | An array containing one fuel-economy prediction per car |

Consequently, `train_scaled = scaler.fit(X_train)` does not put transformed inputs in `train_scaled`. It puts the scaler object there. <strong>To obtain transformed values, use the return value of `transform` or `fit_transform`.</strong>

### Validation data must be transformed too

If we train a model on standardized inputs, we must also standardize the validation inputs before passing them to that model. The distinction is <strong>which data determine the transformation statistics</strong>.

1. The scaler learns its statistics from the training data.
2. The scaler uses those statistics to transform the training data.
3. The model learns coefficients from the transformed training inputs and their targets.
4. The same scaler uses its stored statistics to transform the validation data.
5. The model predicts the validation data using its stored coefficients.

Recalculating the mean and standard deviation from the validation data can represent the same original weight as a different number from the one used during training. The model's coefficients, however, were learned for the training representation. <strong>Changing only the input reference while keeping the coefficients fixed breaks the input-output relationship the model learned.</strong>[^pitfalls]

Combining training and validation data to learn preprocessing statistics also lets information from validation data affect the transformation of training inputs. When information that should be reserved for evaluation enters training, the problem is called <strong>data leakage</strong>. Leakage does not mean the score must improve in every case. The problem is that the evaluation no longer respects the conditions we intended to assess.[^pitfalls]

## 3. What does standardized weight mean?

Standardization <strong>expresses the difference from the training mean in units of the training standard deviation</strong>. To understand that statement, we first need to see what the mean, variance, and standard deviation each calculate.

### Calculate how values differ from the mean

In this section, $x_i$ is the weight of the $i$th training car, and $n$ is the number of training cars. $\mu$ denotes the training mean, $\sigma^2$ the training variance, and $\sigma$ the training standard deviation. The symbol $\sum_{i=1}^{n}$ means to add the terms from $i=1$ through $n$.

<div class="mpg-math">
$$
\mu=\frac{1}{n}\sum_{i=1}^{n}x_i
$$
$$
\sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2,
\qquad \sigma=\sqrt{\sigma^2}
$$
</div>

The mean is the sum of the weights divided by the number of cars. To calculate variance, subtract the mean from each weight, square each difference, and average the squared differences. Simply adding the differences would let positive differences above the mean cancel negative differences below it.

The standard deviation is the square root of the variance. Taking a square root finds the number that gives the original value when squared. If weight is measured in lb, variance has units of lb², while standard deviation has units of lb again. Taking the square root therefore expresses the spread in the same units as the original weight. This is not the only possible way to describe spread.

### Dividing by the standard deviation describes relative position

Let $x$ be the weight to transform and $z$ its standardized value. When the training standard deviation is positive, the transformation is:[^scaler]

<div class="mpg-math">
$$
z=\frac{x-\mu}{\sigma}
\quad\Longleftrightarrow\quad
x-\mu=z\sigma
\quad\Longleftrightarrow\quad
x=\mu+z\sigma
$$
</div>

Multiplying both sides of the first equation by $\sigma$ gives the second equation. Adding $\mu$ to both sides of the second gives the third. Thus, $z$ tells us <strong>how many training standard deviations the original weight lies above or below the mean</strong>.

| Result | Meaning |
| --- | --- |
| $z=0$ | The weight equals the training mean. |
| $z=1$ | The weight is one training standard deviation above the training mean. |
| $z=-2$ | The weight is two training standard deviations below the training mean. |

$x-\mu$ is a <strong>signed difference</strong>: it can be negative. Dividing by the positive value $\sigma$ preserves that direction. For the same difference in weight, a larger standard deviation gives a smaller $|z|$, and a smaller standard deviation gives a larger $|z|$. Here, $|z|$ is the magnitude without its sign, or its absolute value.

In this calculation, we divide lb by lb, so no weight unit remains in the result. Division does not always remove units. This interpretation works here because the numerator and denominator have the same kind of unit.

### Standardization does not force values into a fixed range

With the default `StandardScaler` settings, transforming a nonconstant training column using its own statistics gives a mean of 0 and a standard deviation of 1. Actual numerical calculations may include small rounding errors.

This does not place every value between $-1$ and $1$. Validation data merely have the training statistics applied to them, so <strong>the validation data themselves do not need to have a mean of 0 or a standard deviation of 1.</strong> Standardization does not turn the data's distribution into a normal distribution either. A normal distribution is a particular probability distribution with many values near its center; subtracting a mean and dividing by a fixed number cannot make every dataset follow it.[^preprocessing]

When `StandardScaler` calculates variance, it uses $n$ as the denominator, as in the equation above. In NumPy notation, this is `ddof=0`. `ddof` specifies the number subtracted from the denominator. For example, `ddof=1` makes the denominator $n-1$. Do not confuse the $n-1$ convention used when estimating a population's variance from a sample with the convention used here to calculate input transformation statistics.[^scaler][^std]

A constant training column, in which every value is the same, has variance 0. In that case, `StandardScaler` sets `scale_`, the divisor used in the transformation, to 1. With the default mean subtraction enabled, every value in that training column becomes 0.[^scaler]

### MinMax uses the training minimum and maximum

`MinMaxScaler` is also a class that creates a transformation object. Its default settings represent the training minimum as 0 and the training maximum as 1. Let the training minimum be $m_{\mathrm{train}}$ and the maximum be $M_{\mathrm{train}}$. When they differ, the transformation is:[^minmax]

<div class="mpg-math">
$$
x_{\mathrm{scaled}}=
\frac{x-m_{\mathrm{train}}}{M_{\mathrm{train}}-m_{\mathrm{train}}}
$$
</div>

A new weight above the training maximum can produce a value greater than 1. A weight below the training minimum can produce a value less than 0. The default `clip=False` does not force these values back into the interval from 0 to 1. They are the result of applying the original training reference.

Whichever method we use, each row still represents the same car before and after transformation. For example, comparing `row_id=101` in the original table with `row_id=101` in the transformed result shows the same car's weight expressed relative to different reference values. The transformed weight is neither a fuel-economy prediction nor a loss value.

## 4. Encode region codes and keep each car's information together

The values 1, 2, and 3 in `origin` act as names identifying regions. Code 3 does not describe a region that is three times as large as code 1. We therefore create columns that use 0 and 1 to indicate membership in each region. This is <strong>one-hot encoding</strong>.[^onehot]

| Original `origin` | `origin_1` | `origin_2` | `origin_3` |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 0 |
| 2 | 0 | 1 | 0 |
| 3 | 0 | 0 | 1 |

There was one region-code column; there are now three columns for computation. We have not aggregated car rows by summing or averaging them. <strong>We have added region-indicator columns to the same row for each car.</strong>

### Distinguish the shapes of a Series, a DataFrame, and an array

A Pandas <strong>Series</strong> contains one column of values and an `index` that labels those values. A <strong>DataFrame</strong> is a table with row labels in its `index` and column names. A typical numerical NumPy <strong>array</strong> contains values and a shape, but does not retain Pandas row labels and column names along with them.

`shape` is an attribute that reports the number of values along each dimension. With $n$ cars, the following selections produce different results.[^structures]

| Expression | Result type | `shape` |
| --- | --- | --- |
| `train_frame["weight"]` | Series | `(n,)` |
| `train_frame[["weight"]]` | DataFrame | `(n, 1)` |
| `train_frame[["weight"]].to_numpy(dtype=float)` | Floating-point NumPy array | `(n, 1)` |
| `train_frame[["weight", "origin"]]` | DataFrame | `(n, 2)` |

Passing a <strong>list of column names</strong>, such as `["weight"]`, to the selection keeps the result as a table even when it has only one column. That is why the brackets look doubled. `to_numpy` is a method that converts the table to an array, and `dtype=float` requests floating-point elements.

In learning code, `X_train_num` may refer to numerical inputs such as weight, while `train_cat` may refer to categorical inputs such as region. `valid_row_ids` stores the original `index` of the separate evaluation rows. Similar variable names do not make their purposes the same.

### Assigning category dtype is different from one-hot encoding

`pd.CategoricalDtype` is a class that specifies possible categories and whether they have an order. `categories=[1, 2, 3]` lists the allowed categories, and `ordered=False` means that no greater-than or less-than ordering is assigned to them.

Applying this dtype to `origin` with the `astype` method still leaves one column. Calling the <strong>function</strong> `pd.get_dummies` then creates the separate category columns shown above. `columns=["origin"]` identifies the column to encode, and `dtype=int` requests integer indicators of 0 and 1.[^categorical][^dummies]

Specifying the same category list in advance allows an `origin_3` column to be created even if no evaluation car belongs to region 3. This helps keep the training and evaluation columns consistent. Here, we assume that 1, 2, and 3 are <strong>codes defined in advance by the problem</strong>. That is different from examining evaluation data and then choosing preprocessing rules.

Converting a column to categorical dtype is not a required step before one-hot encoding. An integer column can be named directly in the `columns` argument of `get_dummies`. Conversely, a value outside the specified categories can become missing during categorical conversion, and the default `get_dummies` settings can then encode that row with all category indicators set to 0. Check whether codes are allowed before transforming them.[^categorical][^dummies]

### concat aligns by index, rather than simply by row position

Suppose we attach the original `index` to the standardized weight array to create a DataFrame called `train_weight_frame`, then place it beside the region table `train_onehot`.

`pd.concat` is a function that combines tables. With `axis=1`, it <strong>aligns matching row index labels and appends columns side by side.</strong> It does not average or reduce the rows.[^concat]

| Weight table before concatenation | Region table before concatenation | Correctly combined result |
| --- | --- | --- |
| $n$ cars × 1 weight column | The same $n$ cars × 3 region columns | The same $n$ cars × 4 input columns |
| `(n, 1)` | `(n, 3)` | `(n, 4)` |

Creating a DataFrame from an array without specifying `index` gives it default row labels starting at 0. If the region table's original labels are 101, 205, and 309, Pandas will not regard the row labeled 0 and the row labeled 101 as the same car. This can add rows or introduce missing values. Assigning incorrect labels can also connect information from different cars.

Therefore, <strong>first check that the transformed array preserves the original row order, then attach the original index corresponding to that order.</strong> Check that the model inputs and `y_train` are also in the same order. Pandas aligning tables during concatenation does not mean scikit-learn automatically reconnects inputs and targets by car identifiers. In this example, each input row and target element must refer to the same car at the same position.[^validation]

The manual preprocessing code at the end is an exercise in checking these shapes and correspondences. <strong>We do not pass the resulting `X_train_ready`, transformed in advance using all training data, to cross-validation.</strong> The next section arranges for preprocessing to be learned separately inside each split.

## 5. Train preprocessing and the model together with Pipeline

<strong>Linear regression</strong> predicts a quantity by multiplying each input by a coefficient, adding those terms, and adding an intercept. With one input, the prediction equation is $\hat y=wx+b$. $w$ is the <strong>coefficient</strong> multiplying the input, and $b$ is the <strong>intercept</strong> added at the end. Training determines these values from the data.[^linear]

<strong>Ridge</strong>, the model used here, is linear regression with an additional cost for coefficient size. It considers both prediction error and the cost of larger coefficients. Adding this cost is called <strong>regularization</strong>, and `alpha` in `Ridge(alpha=1.0)` controls its strength. `alpha` is not a gradient-descent learning rate. We will connect the calculation behind regularization after working through derivatives.[^ridge]

<strong>Pipeline</strong> is a class that combines preprocessing and a model in a specified execution order. <strong>ColumnTransformer</strong> is a class that applies different transformations to different columns and then combines their results. <strong>OneHotEncoder</strong> is a class that reuses the category structure established during fitting in later transformations.[^pipeline][^column-transformer][^onehot]

Our Pipeline has the following structure.

| Original input | Object responsible | Result |
| --- | --- | --- |
| `weight` | `StandardScaler` | 1 standardized weight column |
| `origin` | `OneHotEncoder` | 3 region-indicator columns |
| Both transformation results | `ColumnTransformer` | 4 input columns for each car |
| Transformed inputs and targets | `Ridge` | Learn one coefficient per input and an intercept |

### make_model wraps repeated preprocessing setup in a function

To compare a model using weight alone with one using both weight and region, we need preprocessing configurations that fit each model. Instead of rewriting shared code such as weight standardization for every model, we can <strong>define the shared operations once inside a function, then call it with the settings we need</strong>.

We name that function `make_model`. `make_model` is <strong>a function that combines preprocessing objects with the supplied regression model and returns a Pipeline object</strong>. It does not run training internally. Its return value is neither a transformed data table nor a prediction. Its definition is included in the complete code in Section 13.

Consider the call `cv_model = make_model(Ridge(alpha=1.0), include_origin=True)`. Python first creates a Ridge object with `Ridge(alpha=1.0)`. `make_model` receives that object through the `regressor` argument and places it at the end of the Pipeline. The setting `include_origin=True` includes the region column among the inputs. Changing it to `False` configures the model to use weight alone.[^functions]

<strong>The internal operations not visible in this short call</strong> are listed below. The code for each operation appears in the body of `make_model`.

| Order | What the function configures | Role during later training and prediction |
| --- | --- | --- |
| 1 | Create a `StandardScaler` object to handle `weight`. | Standardize weight. |
| 2 | If `include_origin=True`, add a `OneHotEncoder` object to handle `origin`. | Convert region codes into separate columns containing 0 or 1 for each region. |
| 3 | Combine the column-specific transformers in a `ColumnTransformer`. | Join each car's transformed values into one input. |
| 4 | Combine preprocessing and the supplied `regressor` in a `Pipeline`, then return it. | Preserve the order of preprocessing followed by the regression model. |

What the call leaves out is <strong>the internal configuration code we would otherwise repeat at each call site</strong>. The preprocessing operations themselves still take place. Keeping the configuration inside a function means we do not need to copy the common preprocessing code when changing the model or deciding whether to include region. For example, `make_model(LinearRegression(), include_origin=False)` configures linear regression with the same weight standardization and without region.

Inside the function, the weight and region transformations are named `weight` and `origin`. Within the Pipeline, the complete preprocessing step is named `preprocess`, and the regression model is named `model`. These names let us locate stored objects later or change a particular step's setting through a name such as `model__alpha`. The names do not create the corresponding behavior automatically; the code associates each name with an object.

When region is included, we pass `categories=[[1, 2, 3]]` to `OneHotEncoder`. The outer list contains one list per input column; the inner list contains the allowed region codes for the single `origin` column. Defining these three categories in advance preserves three region-indicator columns even when a training portion contains no cars from a particular region. `sparse_output=False` requests an ordinary two-dimensional array, and `handle_unknown="error"` raises an error when a value outside the allowed codes appears.[^onehot]

### After configuration, use fit to train and predict to make predictions

`make_model` creates and connects objects internally. We do not pass it any car data or targets at this point, so it does not calculate the mean, standard deviation, or prediction coefficients. The same is true when we construct a `Pipeline` object directly. This configuration code defines the types of preprocessing and their execution order.

Later, calling `cv_model.fit(train_frame, y_train)` learns preprocessing statistics from the training data, transforms those same data, and then fits Ridge coefficients using the transformed inputs and targets.

After training, calling `cv_model.predict(sample_request)` transforms the new inputs using the stored preprocessing statistics and predicts using the stored coefficients. <strong>Prediction does not relearn either the preprocessing statistics or the model coefficients.</strong>[^pipeline]

## 6. Cross-validation refits preprocessing in every split

If we split data only once to judge a model, the result can depend on that particular split. <strong>Cross-validation (CV)</strong> repeats training and evaluation while changing which portions of the training data are used for fitting and validation.[^cv-guide]

The example in Section 13 sets aside 4 of its 16 fictional cars for final evaluation and uses the remaining 12 as training data. Dividing these <strong>12 training cars</strong> into three groups, A, B, and C, gives 4 cars per group. A, B, and C are names used to explain the roles in the three splits below.

| Split | Cars used to fit preprocessing and the model | Cars used to evaluate predictions in this split |
| --- | --- | --- |
| Split 1 | A + B, 8 cars | C, 4 cars |
| Split 2 | A + C, 8 cars | B, 4 cars |
| Split 3 | B + C, 8 cars | A, 4 cars |

The 4 validation cars in each split come from the 12 training cars. The 4 cars set aside for final evaluation at the outset are not part of these three groups.

<strong>The fitted model from the previous split is not carried forward.</strong> Each split trains a new model with the same configuration. That is why C can be validation data in split 1 and training data in split 2. The model in split 1 was not trained on C, and the model in split 2 is trained separately.

The scaler in split 1 must also calculate its mean and standard deviation only from A+B. Calculating a mean from all of A+B+C in advance would introduce information from C into split 1's training inputs. We therefore pass <strong>the entire Pipeline, including preprocessing</strong>, to cross-validation.[^pitfalls][^cv]

`cross_validate` is a <strong>function</strong> that runs these repeated evaluations. It receives a model configuration, the original input table, targets, splitting rules, and an evaluation method. Here, it returns a <strong>dictionary</strong> containing scores and times for each split. A dictionary is a data structure that associates names, called keys, with their corresponding values.[^cv]

| Result key | Stored information | Shape for three splits |
| --- | --- | --- |
| `train_score` | The training score for each split, requested with `return_train_score=True`. | `(3,)` |
| `test_score` | The validation score for each split | `(3,)` |
| `fit_time` | The fitting time for each split | `(3,)` |
| `score_time` | The time spent calculating the validation score for each split | `(3,)` |

The `test` in `test_score` refers to the validation portion inside cross-validation. It does not report an evaluation on the separately held-out `valid_frame`. Also, `cross_validate` runs each split's experiment using cloned models. After the call, we must not assume that the original `cv_model` itself has been fitted on the entire training dataset.[^cv][^validation]

The complete code uses `KFold` to create three splits and stores them as a list in `cv_splits`. `KFold` is a class that divides data so each group is used once for validation. `shuffle=True` shuffles the order before splitting, and the integer `random_state=42` reproduces the split for the same inputs. The number 42 has no special performance advantage. Reusing the list compares all candidates on the same cars.[^kfold]

This example treats each car row as an individual observation. Data with additional structure, such as time-ordered records or repeated measurements of the same subject, need a split that respects that structure. Pipeline does not automatically correct a bad split or leakage that has already occurred outside the Pipeline.[^cv-guide]

## 7. Read MSE and negative scores consistently

<strong>MSE (mean squared error)</strong> is the average of the squared prediction errors. If the $i$th car's target is $y_i$, prediction is $\hat y_i$, and error is $e_i$, then:[^mse]

<div class="mpg-math">
$$
e_i=\hat y_i-y_i,
\qquad
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}e_i^2
$$
</div>

Consider two predictions <strong>invented for illustration</strong>. If the first car's actual fuel economy is 20 mpg and its prediction is 18 mpg, the error is $18-20=-2$. If the second car's actual fuel economy is 25 mpg and its prediction is 26 mpg, the error is $26-25=1$.

| Car | Actual fuel economy | Predicted fuel economy | Error | Squared error |
| --- | ---: | ---: | ---: | ---: |
| A | 20 | 18 | −2 | 4 |
| B | 25 | 26 | 1 | 1 |

The mean of the squared errors is $(4+1)/2=2.5$. That is the MSE. Its unit is mpg², so it does not mean the predictions are off by 2.5 mpg on average. When comparing predictions against the same targets on the same evaluation data, a smaller MSE means smaller squared error.

<strong>Training MSE</strong> is calculated on the data used for fitting, while <strong>validation MSE</strong> is calculated on data that were not used to fit that particular model. Fitting the training data well while performing poorly on new data is called <strong>overfitting</strong>. A very small training MSE together with a large validation MSE can suggest overfitting, but the score gap alone cannot establish the cause.[^cv-guide]

### neg_mean_squared_error changes the score's sign, not the errors

A scikit-learn <strong>scorer</strong>, used to compare models, treats a larger score as better. Since a smaller MSE is better, `scoring="neg_mean_squared_error"` returns the negative of the MSE.[^scoring]

| Candidate | Actual MSE | Comparison score `−MSE` | Interpretation |
| --- | ---: | ---: | --- |
| A | 20 | −20 | Larger prediction error than B. |
| B | 10 | −10 | Smaller MSE and a larger comparison score. |

This table is also an illustration of the sign convention. Because $-10$ is larger than $-20$, scikit-learn selects B as the better candidate too. <strong>The prediction errors have not been reduced; only the score representation has changed to match the comparison rule.</strong>

Negating `cv_result["test_score"]` again restores each element to the actual MSE. Therefore, the values `-cv_result["train_score"]` and `-cv_result["test_score"]` placed in `cv_table` should once again be read as <strong>smaller is better</strong>. We are not negating every possible evaluation score. We are reversing the sign because we explicitly requested negative MSE.

### A baseline establishes the starting point for comparison

A <strong>baseline</strong> is a reference model chosen to assess how much later models improve. This example uses `LinearRegression` with weight as its only input. The variable name `baseline` does not have any special functionality.

After changing input features or model settings, we must still evaluate on the same splits with the same metric to make a comparison. The complete code below also compares a weight-only reference model with Ridge using weight and region. Both the inputs and the model change in this comparison, so <strong>we must not attribute the performance difference to regularization alone</strong>.

A score by itself is not a complete record. Recording the data split, input columns, preprocessing, model settings, and evaluation metric lets us revisit the result. To reuse the same prediction process later, save preprocessing and the model together. That does not mean every performance comparison requires saving a fitted object.

## 8. Select settings with Grid Search and build the final model with refit

We need to distinguish values learned by the model from settings supplied as candidates. <strong>Model parameters</strong> are coefficients $w$ and the intercept $b$, learned from data. <strong>Hyperparameters</strong> are settings chosen before an individual training run starts; Ridge's `alpha` is one of them.

<strong>Grid Search</strong> evaluates a predefined set of candidate settings and selects a good one. `GridSearchCV` is a class that creates a search object evaluating each candidate through cross-validation.[^grid]

Here, the `alpha` candidates are `[0.01, 0.1, 1, 10, 100]`. Each is evaluated on the same three splits, and the candidate with the smallest mean validation MSE is selected. In terms of negative MSE, this means choosing the largest mean score.

### model__alpha identifies the step whose setting is being changed

Because the Pipeline's last step is named `model`, `model__alpha` refers to <strong>the `alpha` setting inside that step</strong>. The two underscores, `__`, separate the step name from the setting name. If the step had a different name, we would use that name instead.[^pipeline]

| Name | Meaning |
| --- | --- |
| `param_grid` | A dictionary of settings and candidate values to compare |
| `cv_results_` | A results dictionary containing scores, settings, and other information for each candidate |
| `param_model__alpha` | Each candidate's `alpha`, recorded in the results table |
| `mean_test_score` | The candidate's mean cross-validation score; negative MSE here |
| `best_params_` | A dictionary containing the selected settings |
| `best_score_` | The selected candidate's mean cross-validation score |
| `best_estimator_` | With `refit=True` in this example, the final Pipeline refitted on all training data |

`search.cv_results_`, `search.best_params_`, `search.best_score_`, and `search.best_estimator_` are <strong>attributes</strong> stored on the object. We read their values rather than calling them with parentheses as we would methods. By contrast, `param_model__alpha` and `mean_test_score` are <strong>keys</strong> in the `cv_results_` dictionary; they become column names when the results are converted to a DataFrame. `param_grid` is the configuration passed when creating the search object. In scikit-learn, attributes established during fitting often end with an underscore, `_`.[^grid]

### refit does more than retrieve one model used during selection

<strong>refit</strong> trains the selected configuration again on all the training data passed to `search.fit`. With `refit=True`, running `search.fit(train_frame, y_train)` performs this refitting step after evaluating the candidates.[^grid]

| Stage | Data used to learn the scaler's statistics | Data used to learn model coefficients |
| --- | --- | --- |
| One cross-validation split | That split's training portion | That split's training portion |
| Refit after selecting settings | All of `train_frame` | All of `train_frame` |
| Final evaluation and new predictions | Statistics are not relearned. | Coefficients are not refitted. |

In this example, each CV split fits preprocessing and the model on 8 training cars. After selecting settings, it refits on all 12 cars in `train_frame`. The <strong>final statistics and coefficients learned from those 12 training cars</strong> are then applied to the 4 cars in `valid_frame` and to new cars.

Evaluating five candidates across three splits fits the Pipeline $5\times3=15$ times. If every candidate runs successfully, adding one final refit gives 16 fits in one `search.fit` call. This does not mean coefficients were updated 16 times. It means <strong>the entire Pipeline fitting procedure ran 16 times</strong>. Fits performed by a separate earlier `cross_validate` call are not included in this count.

`selected_model = search.best_estimator_` assigns a reference to the already refitted object. That line does not run training again. `selected_model.predict(valid_frame)` predicts with that object, and `search.predict(valid_frame)` also uses the refitted final model.[^grid]

### Inspect stored values on the object that learned them

`selected_model.named_steps["preprocess"]` locates the preprocessing step in the fitted Pipeline. Accessing its `named_transformers_["weight"]` locates the weight transformer, whose `mean_` and `scale_` can then be read. The final `model` step exposes `coef_` and `intercept_`.

`mean_` is the stored mean, `scale_` is the divisor used during transformation, `coef_` contains the input coefficients, and `intercept_` is the intercept. Ridge in this example has four coefficients for its four transformed input columns, plus one intercept. <strong>Inspect the actual stored values on the final selected object, rather than guessing which object from the initial construction holds them.</strong>[^scaler][^ridge][^column-transformer]

### Interpret the complete example's execution results

Running the fictional car data and complete code in Section 13 produces the following results. Read each value together with the data and stage used to calculate it.

| Execution result | What it tells us |
| --- | --- |
| Selected setting `{"model__alpha": 0.01}` | This setting was selected from the five specified candidates. |
| Mean validation MSE of about `0.5831` | The mean validation MSE across the three splits used to select the setting. |
| Separate evaluation MSE of about `0.2756` | The MSE calculated from the final model's predictions for the 4 cars set aside at the outset. |
| Prediction of about `27.7189 mpg` for `weight=2800`, `origin=2` | The fuel economy calculated by the final model for a new request. |

`0.5831` and `0.2756` are scores calculated on different data. The first is the cross-validation score used to select a candidate; the second is the score calculated on the separate evaluation data after the final selection. `27.7189 mpg` is a prediction for one car, so we cannot know its prediction error until we compare it with the actual fuel economy.

The selected `alpha=0.01` is a value chosen from the specified candidates. It does not mean the search found the best possible `alpha` among all real numbers. Also, repeatedly checking the final evaluation score and changing settings makes those data influence selection. They can then no longer be said to serve the originally intended role of an independent final evaluation set.[^cv-guide]

## 9. A loss derivative is different from the coefficient of the prediction line

So far, we have delegated training to the library. Now let us examine why scaling inputs affects coefficient learning. To keep the derivation focused, we narrow the problem to <strong>linear regression with weight as the only input</strong>. We are not expanding every internal calculation of the full model that also includes region columns.

A <strong>function</strong> is a rule that maps inputs to outputs; a <strong>function value</strong> is the result of applying that rule to a particular input. The prediction equation $\hat y=wx+b$ maps weight $x$ to predicted fuel economy. A loss function assigns a numerical value to how poor the predictions produced by the current coefficients are.

Differentiation determines how quickly one value changes when another changes by a very small amount. Here, the quantity being varied is not input weight, but <strong>a coefficient adjusted during training, and we want the instantaneous rate of change of the loss with respect to that coefficient</strong>. Differentiating with respect to one variable while holding the others fixed is called <strong>partial differentiation</strong>, denoted by $\partial$.

In this section, $\ell$ denotes the loss for one car, and $J_{\mathrm{data}}$ denotes the average loss across cars. The symbol $\ell$ is a lowercase L used to represent loss.

<div class="mpg-math">
$$
\hat y=wx+b,\qquad e=wx+b-y,\qquad \ell(e)=\frac{e^2}{2}
$$
$$
e_i=wx_i+b-y_i,\qquad
J_{\mathrm{data}}=\frac{1}{n}\sum_{i=1}^{n}\ell(e_i)
=\frac{1}{2n}\sum_{i=1}^{n}e_i^2
$$
</div>

<strong>$J_{\mathrm{data}}$ is half the MSE.</strong> We use this convention so the factor of 2 produced by differentiating a square matches the 2 in the denominator. Keep the MSE displayed in results tables separate from $J_{\mathrm{data}}$, which we use to explain training.

| Symbol | Meaning here |
| --- | --- |
| $w$ | The prediction coefficient multiplying input $x$ |
| $dw=\partial J_{\mathrm{data}}/\partial w$ | A code variable name storing the derivative of average loss with respect to $w$, evaluated at the current $w$ |
| $db=\partial J_{\mathrm{data}}/\partial b$ | A code variable name storing the derivative of average loss with respect to $b$, evaluated at the current $b$ |
| $\eta$ | The positive learning rate that multiplies the derivative when updating coefficients |

The code name `dw` does not mean a very small change in $w$ here. Likewise, $d\ell/de$ is not an instruction to divide $\ell$ by $e$. <strong>The whole notation represents the derivative of loss with respect to $e$.</strong>

### Why does error e change at a rate of x when w changes?

Our goal is to differentiate the loss $\ell=(wx+b-y)^2/2$ with respect to $w$. Since $w$ first changes the error $e$, we begin with that relationship. During this calculation, $x$, $b$, and $y$ stay fixed. That does not mean $b$ remains fixed throughout training; the effect of $b$ is calculated separately.

Define $h$ as <strong>a change added to the coefficient $w$</strong>. While dividing by that change, we require $h\ne0$.

<div class="mpg-math">
$$
e(w)=wx+b-y,
\qquad e(w+h)=(w+h)x+b-y
$$
$$
\begin{aligned}
e(w+h)-e(w)
&=(w+h)x+b-y-(wx+b-y)\\
&=wx+hx+b-y-wx-b+y\\
&=hx
\end{aligned}
$$
</div>

The second line expands the multiplication and applies the minus sign before the parentheses to every term inside them. In the third line, $wx$ cancels $-wx$, $b$ cancels $-b$, and $-y$ cancels $y$, leaving $hx$.

Dividing the change in error by the change in $w$, denoted by $h$, gives:

<div class="mpg-math">
$$
\frac{e(w+h)-e(w)}{h}=\frac{hx}{h}=x
$$
$$
\frac{\partial e}{\partial w}
=\lim_{h\to0}\frac{e(w+h)-e(w)}{h}=x
$$
</div>

$\lim_{h\to0}$ asks what value the expression approaches as $h$ approaches 0. Here, shrinking $h$ keeps the ratio equal to $x$, so the derivative is also $x$. We have not directly substituted 0 into the denominator and divided by it.

We can use the same approach by calling the change added to $b$ by the name $k$. This time, hold $w$, $x$, and $y$ fixed, with $k\ne0$.

<div class="mpg-math">
$$
e(b+k)-e(b)
=wx+(b+k)-y-(wx+b-y)=k
$$
$$
\frac{\partial e}{\partial b}
=\lim_{k\to0}\frac{k}{k}=1
$$
</div>

Whatever amount is added to the intercept is added directly to the error, so the rate of change is 1.

### Why does loss change at a rate of e when the error changes?

Now view loss as a function of error alone: $\ell(e)=e^2/2$. Define $t$ as <strong>a change added to the error $e$</strong>, with $t\ne0$ while dividing by it.

<div class="mpg-math">
$$
\begin{aligned}
\frac{\ell(e+t)-\ell(e)}{t}
&=\frac{(e+t)^2/2-e^2/2}{t}\\
&=\frac{e^2/2+et+t^2/2-e^2/2}{t}\\
&=\frac{et+t^2/2}{t}\\
&=e+\frac{t}{2}
\end{aligned}
$$
</div>

The first line substitutes the errors before and after the change into the loss equation. The second expands $(e+t)^2=e^2+2et+t^2$. In the third, the $e^2/2$ terms cancel. The last line divides each of the two remaining terms by $t$.

<div class="mpg-math">
$$
\frac{d\ell}{de}
=\lim_{t\to0}\left(e+\frac{t}{2}\right)=e
$$
</div>

Finally, $t/2$ approaches 0, leaving $e$. The 2 in the denominator has not disappeared without a reason.

### Connecting the two rates gives the loss derivative

A change in coefficient $w$ affects loss through <strong>$w\rightarrow e\rightarrow\ell$</strong>. Using $\Delta$ to denote a change, the following equality holds when both $\Delta w$ and $\Delta e$ are nonzero.

<div class="mpg-math">
$$
\frac{\Delta\ell}{\Delta w}
=\frac{\Delta\ell}{\Delta e}\times\frac{\Delta e}{\Delta w}
$$
</div>

The right side equals the left because the same change, $\Delta e$, appears in both the numerator and denominator. The rule for multiplying instantaneous rates along such connected relationships is the <strong>chain rule</strong>. Connecting the two derivatives we just calculated gives:

<div class="mpg-math">
$$
\frac{\partial\ell}{\partial w}
=\frac{d\ell}{de}\frac{\partial e}{\partial w}
=e\times x
$$
$$
\frac{\partial\ell}{\partial b}
=\frac{d\ell}{de}\frac{\partial e}{\partial b}
=e\times1=e
$$
</div>

This does not mean the differentiation symbols $d$ or $\partial$ can be canceled like ordinary numbers. When $x=0$, $e$ does not change with $w$, so the loss does not change with $w$ either. The derivative is then 0, which also agrees with the final expression $ex$.

### For several cars, average the individual loss derivatives

$J_{\mathrm{data}}$ adds the losses for all cars and divides by $n$. Changing $w$ changes each car's loss, so <strong>adding those individual loss changes and dividing by the same $n$ gives the change in average loss</strong>. Divide that result by the same coefficient change and let the change approach 0. The result is the average of the individual derivatives shown below. The car count $n$ stays fixed during this differentiation.

<div class="mpg-math">
$$
\begin{aligned}
dw=\frac{\partial J_{\mathrm{data}}}{\partial w}
&=\frac{1}{n}\sum_{i=1}^{n}\frac{\partial\ell(e_i)}{\partial w}\\
&=\frac{1}{n}\sum_{i=1}^{n}e_ix_i
\end{aligned}
$$
$$
\begin{aligned}
db=\frac{\partial J_{\mathrm{data}}}{\partial b}
&=\frac{1}{n}\sum_{i=1}^{n}\frac{\partial\ell(e_i)}{\partial b}\\
&=\frac{1}{n}\sum_{i=1}^{n}e_i
\end{aligned}
$$
</div>

This average is <strong>the derivative, at the current point, of the function that averages the losses across cars</strong>. It is not an average rate of change between two widely separated coefficient values. Since MSE itself is $2J_{\mathrm{data}}$, differentiating MSE itself gives twice these results.

### Gradient descent updates coefficients in the direction opposite the derivative

<strong>Gradient descent</strong> calculates the loss derivative at the current coefficients and updates the coefficients in the opposite direction. We denote the learning rate by $\eta$ throughout this article.

<div class="mpg-math">
$$
w_{\mathrm{new}}=w-\eta dw,
\qquad b_{\mathrm{new}}=b-\eta db
$$
</div>

If $dw>0$, increasing $w$ near the current point increases loss, so the update decreases $w$. If $dw<0$, increasing $w$ points toward decreasing loss, so the update increases $w$. The size of the step depends on the learning rate and the magnitude of the derivative.

A derivative describes the neighborhood of the current point. If the learning rate is too large, moving in that direction can still increase the loss at the new point. Updating with the opposite sign therefore does not guarantee decreasing loss for every learning rate.

In one iteration, we predict using the current coefficients, calculate errors and loss, calculate derivatives, and update the coefficients. Calculating a prediction or a loss is not itself differentiation. Actual code substitutes current values into derivative formulas derived in advance, then repeats the process using the new coefficients.

### Check scaling and stopping conditions together

$dw$ contains $e_ix_i$. Changing the numerical representation of inputs changes this calculation and the current errors, and can change the update size and stability at the same learning rate. However, terms from different samples can cancel, so larger inputs do not always imply a larger overall $dw$.

When comparing gradient descent with different input representations, we should therefore check both the final MSE and <strong>how the loss changed and why iteration stopped</strong>. If we write our own iterative training code, we can use different stopping conditions such as those below. This table explains how to interpret stopping conditions; it does not report measurements.

| Example stopping condition | What stopping under this condition means | What else to check |
| --- | --- | --- |
| Stop when MSE exceeds a predetermined limit. | The loss has become too large, so the run stops instead of continuing the calculation. | Check the learning rate, input scale, and derivative and update calculations. |
| Stop after using the specified maximum number of updates. | The run has used all the allowed iterations. | Check whether the loss fell sufficiently and whether it was still changing at the end. |

Exceeding an MSE limit does not by itself prove mathematical divergence. Completing the specified number of iterations also does not establish that the exact minimum has been reached. Even if two methods finish with similar training MSE, they can require different numbers of updates to reach the same value or perform differently on separate evaluation data.

The derivation above differentiates MSE/2 to obtain the quantities used to update coefficients. If a manually written loop reports its progress as MSE, it displays twice the loss used for the updates. <strong>Check the definition of the loss being differentiated and the definition of the reported evaluation value separately</strong> to avoid confusing constants in the update equations with numbers in a results table.

## 10. L1 and L2 do not overwrite coefficients with their squares

Now let us examine <strong>what regularization adds to coefficient learning</strong>. To impose a cost on coefficient size while reducing prediction error, we add a measure of coefficient size to the training objective. The model chooses coefficients that reduce the sum.[^linear]

With one input, the prediction equation is still $\hat y=wx+b$. The average error-based loss is the $J_{\mathrm{data}}$ defined earlier, and $\lambda$ is a nonnegative value controlling the strength of the additional cost.

<div class="mpg-math">
$$
J_{\mathrm{L1}}=J_{\mathrm{data}}+\lambda|w|
$$
$$
J_{\mathrm{L2}}=J_{\mathrm{data}}+\lambda w^2
$$
</div>

<strong>L1</strong> imposes a cost on the coefficient's absolute value, while <strong>L2</strong> imposes a cost on its square. We have neither changed the prediction equation to $\hat y=w^2x+b$ nor overwritten the stored $w$ with $w^2$. <strong>We have added a new term to the objective being minimized.</strong>

For $p$ coefficients, add the cost for each coefficient. Here, $p$ is the number of model input columns after transformation, and $j$ identifies a column.

<div class="mpg-math">
$$
P_{\mathrm{L1}}=\sum_{j=1}^{p}|w_j|,
\qquad
P_{\mathrm{L2}}=\sum_{j=1}^{p}w_j^2
$$
</div>

L2 <strong>squares each coefficient and then adds the squares</strong>. That differs from adding all coefficients first and then squaring the sum. These regularization expressions exclude the intercept $b$ from the penalty.

### How the L2 term enters a coefficient update

Let us differentiate the L2 cost $\lambda w^2$ for one coefficient with respect to $w$. As before, let $h$ be a change in $w$, and hold $\lambda$ fixed.

<div class="mpg-math">
$$
\begin{aligned}
\frac{d(\lambda w^2)}{dw}
&=\lim_{h\to0}\frac{\lambda(w+h)^2-\lambda w^2}{h}\\
&=\lim_{h\to0}\frac{2\lambda wh+\lambda h^2}{h}\\
&=\lim_{h\to0}(2\lambda w+\lambda h)\\
&=2\lambda w
\end{aligned}
$$
</div>

Expanding the square cancels the $\lambda w^2$ terms. Dividing the remaining terms by $h$ gives $2\lambda w+\lambda h$. As $h$ approaches 0, the final term approaches 0. Thus, <strong>$2\lambda w$ is added</strong> to the derivative of the error term.

<div class="mpg-math">
$$
\frac{\partial J_{\mathrm{L2}}}{\partial w}
=\frac{\partial J_{\mathrm{data}}}{\partial w}+2\lambda w
$$
$$
\begin{aligned}
w_{\mathrm{new}}
&=w-\eta\left(\frac{\partial J_{\mathrm{data}}}{\partial w}+2\lambda w\right)\\
&=w-2\eta\lambda w-\eta\frac{\partial J_{\mathrm{data}}}{\partial w}\\
&=(1-2\eta\lambda)w-\eta\frac{\partial J_{\mathrm{data}}}{\partial w}
\end{aligned}
$$
</div>

We multiply the parentheses in the first line by $-\eta$, then combine the two terms containing $w$. Looking only at the direction contributed by L2, it subtracts a positive value when $w>0$ and subtracts a negative value when $w<0$, directing $w$ toward 0.

For one update considering only the L2 term, if $0<2\eta\lambda<1$, the multiplier on $w$ is between 0 and 1. Its sign stays the same while its absolute value shrinks. The actual update also includes the derivative of the error term, however. <strong>Every coefficient's absolute value does not necessarily decrease in every iteration.</strong> An excessively large learning rate can make training unstable even with regularization.

These equations explain L2's effect using gradient descent. They do not imply that `Ridge` necessarily trains through this recurrence. Depending on its configured solver, Ridge can find coefficients through other numerical methods.[^ridge]

### Why L1 and L2 can select different coefficients

| Method | Additional cost | Possible pattern in the learned result |
| --- | --- | --- |
| L1 | Sum of absolute coefficient values. | Can make some coefficients exactly 0. |
| L2 | Sum of squared coefficients. | Tends to restrain coefficient size while retaining multiple coefficients. |

If L1 makes a coefficient 0, its input term contributes nothing to that linear prediction equation. This does not mean the selected features are causes in the real world. L2 does not rule out the possibility that a coefficient is 0 either.[^linear]

$|w|$ has different rates of change from the left and the right at $w=0$, so its ordinary derivative does not exist there. We must therefore not explain exact zeros under L1 by claiming that ordinary gradient-descent updates always reach zero. Lasso finds coefficients with a computational method suited to its L1 objective.[^lasso]

### Prediction equations and training objectives of LinearRegression, Ridge, and Lasso

With the same inputs, all three models share the form of a linear prediction equation: multiply inputs by coefficients and add the terms. Their difference is the objective used to choose the coefficients during training.[^linear][^ridge][^lasso]

| Class | Training objective |
| --- | --- |
| `LinearRegression` | Reduce the sum of squared errors without an additional L1 or L2 penalty. |
| `Ridge` | Reduce the sum of squared errors plus an L2 penalty. |
| `Lasso` | Reduce an error term normalized by sample count plus an L1 penalty. |

`Ridge` is not merely an object that calculates an L2 cost. It is a model object responsible for fitting coefficients, predicting, and managing settings and numerical methods. Regularization is not exclusive to Ridge either. It can be added to other models or to a training objective we write ourselves.

Also, <strong>the training objective and the evaluation score must be kept separate</strong>. The objective Ridge reduces during fitting includes an L2 cost, but `mean_squared_error` and the cross-validation MSE in this article measure only prediction error. The displayed MSE does not include the L2 cost.[^ridge][^mse]

### Check the scaling conventions of λ in the equations and alpha in Ridge

Our definition of $J_{\mathrm{data}}$ divides the sum of squared errors by $2n$. The documented scikit-learn Ridge objective uses the following scaling convention instead.[^ridge]

<div class="mpg-math">
$$
\sum_{i=1}^{n}e_i^2
+\texttt{alpha}\sum_{j=1}^{p}w_j^2
$$
</div>

Dividing this entire expression by the positive value $2n$ does not change which coefficients minimize it, because every candidate objective value is divided by the same constant. The resulting expression is:

<div class="mpg-math">
$$
J_{\mathrm{data}}
+\frac{\texttt{alpha}}{2n}\sum_{j=1}^{p}w_j^2
$$
</div>

Therefore, <strong>matching this article's L2 equation to Ridge requires $\lambda=\texttt{alpha}/(2n)$</strong>. We must not assume that $\lambda$ in the general equation defined above and `alpha` in code are numerically identical.

Lasso's documented objective is $J_{\mathrm{data}}+\texttt{alpha}\sum_j|w_j|$, so under this convention, $\lambda=\texttt{alpha}$. The penalties have different forms and the error terms use different scaling conventions. Giving Ridge and Lasso the same numerical `alpha` therefore does not produce the same regularization effect.[^lasso]

### Scaling changes inputs; regularization changes the training objective

Scaling changes the numerical representation of input $x$. Regularization changes the training objective to account for the size of coefficient $w$. Changing input units changes the coefficient size needed to represent the same prediction, so a large coefficient is not inherently bad.

When numerical features have very different units and scales, their representations also affect regularization. That is why input scale matters when using regularized linear models. However, scaling is not required for every model, and stronger regularization does not always improve validation performance.[^preprocessing][^linear]

## 11. Input count, coefficient count, and graph axes are different

<strong>Expressiveness</strong> is the range of input-output relationships a model can represent. <strong>Training</strong> chooses a relationship within that range to fit the data, and <strong>prediction</strong> evaluates the chosen relationship at a supplied input.

With one input, $\hat y=wx+b$ is a straight line regardless of its coefficient values. Allowing an additional input term $x^2$ gives $\hat y=b+w_1x+w_2x^2$, which can represent a quadratic curve when $w_2\ne0$. Training a model restricted to straight lines for longer does not automatically add an $x^2$ term that its code does not provide.[^polynomial]

Decision-tree regression can return different predictions depending on input conditions. A neural network represents relationships by connecting operations within its specified architecture. Not every model produces a quadratic curve, and a function does not have to be expressible as a short polynomial.[^tree][^mlp]

### Four coefficients do not imply four inputs

| Prediction equation | Original input variables | Learned coefficients and intercept | Plotting the prediction relationship |
| --- | --- | --- | --- |
| $\hat y=b+w_1x+w_2x^2+w_3x^3$ | One variable, $x$ | Four values: $w_1,w_2,w_3,b$ | Can be plotted with $x$ horizontally and $\hat y$ vertically. |
| $\hat y=b+w_1x_1+w_2x_2+w_3x_3$ | Three variables: $x_1,x_2,x_3$ | Four values: $w_1,w_2,w_3,b$ | Representing all inputs and the output simultaneously requires four coordinates. |

In the first equation, $x^2$ and $x^3$ are calculated from the same $x$. The second equation takes $x_1$, $x_2$, and $x_3$ as separate input coordinates. Their values can still be related in real data. Preparing a model input table for the first equation gives three transformed input columns, but only one original variable generates their values.

The car model likewise has two original features in terms of meaning: weight and region. Preprocessing creates one standardized weight column and three region-indicator columns, for four input columns in total. There are four corresponding coefficients, plus a separate intercept. <strong>Count the original features, transformed columns, and coefficients separately.</strong>

### Check the axes to identify what a graph shows

| Graph | Value being varied or compared | Vertical axis | Question it answers |
| --- | --- | --- | --- |
| Prediction relationship | Input $x$ | Prediction $\hat y$ | What fuel economy is predicted as weight changes? |
| Loss function | Coefficient $w$, with other coefficients fixed | Loss $J$ | How does changing a coefficient change the loss? |
| Training progress | Number of updates | Loss or MSE | How does error change as training repeats? |

Allowing both $w$ and $b$ to vary gives the loss $J(w,b)$, a function of two variables. It can be represented as a surface using a $w$ axis, a $b$ axis, and a loss axis. A straight prediction relationship does not require a straight loss function. A quadratic loss function does not mean the graph of training progress against update count is a parabola.

## 12. Distinguish functions we define from library features

`make_model` is a function defined in this article's code, while `Pipeline` and `Ridge` are classes provided by scikit-learn. A <strong>function definition</strong> specifies the values the function will receive and the operations it will perform. A <strong>function call</strong> passes values to that definition and runs the operations. In Section 13, the statements indented below `def make_model(regressor, include_origin=True):` form the function body, and the final `return` sends the constructed Pipeline back to the point of the call.[^functions]

If we move the function definition to another Python file, we can bring it in with `import`. The function body is then absent from the file making the call, but the code that runs is in the imported file. When reading a short function call, check <strong>what we pass in, what it configures or calculates internally, and what it returns</strong>.

| Kind | Examples in this article | How to read it |
| --- | --- | --- |
| Class | `StandardScaler`, `Ridge`, `Pipeline`, `GridSearchCV`, `pd.DataFrame` | Calling it creates an object with that role. |
| Function | `make_model`, `cross_validate`, `mean_squared_error`, `pd.get_dummies`, `pd.concat` | Pass the required arguments, run the operation, and receive a return value. |
| Method | `fit`, `transform`, `predict`, `astype` | Functionality called through a particular object. |
| Attribute | `shape`, `mean_`, `scale_`, `coef_`, `best_score_` | Read information stored on the object. |

This article includes the definition and calls of `make_model` in the same code listing. To examine the location or contents of a function imported from another file, Python provides `inspect.getfile` and `inspect.getsource`. They return, respectively, the file location of the definition and its accessible Python source text as strings. For example, in the downloaded Python file, `inspect.getsource(make_model)` lets us inspect this function's definition. Inspection can fail when the source cannot be located, as can happen in some notebook environments.[^inspect]

A project that downloads data files may need file preparation code before model training. `Path` handles file paths, and ZIP handling extracts compressed materials. SHA-256 is used to compare values calculated from file contents to check whether a file has changed. Adding a directory to `sys.path` allows Python to find modules there. These operations prepare files and modules and serve a different role from learning model coefficients.[^python-files]

Plotting and display code should also be distinguished from training. `np.linspace` is a function that creates evenly spaced numbers over a specified interval. Creating 200 numbers supplies 200 input positions for drawing a line, rather than adding measurements of 200 cars. `scatter` displays values as points, while `plot` can connect values with a line. `head` shows the beginning of a table, and `round` produces rounded values. Rounding a results table does not refit the model's coefficients.[^plotting]

## 13. Run the full Python example with fictional car data

The following code is <strong>one complete example covering data creation, function definitions, preprocessing, training, and evaluation</strong>. It uses 16 fictional cars created to demonstrate the processing sequence. Twelve cars are used for training and four for separate evaluation, so each of the three CV splits described earlier trains on eight cars and validates on four.

The required libraries are NumPy, Pandas, and scikit-learn. Because the code uses `sparse_output`, it targets scikit-learn 1.2 or later. While preparing this article, the complete code was run and checked with NumPy 2.3.5, Pandas 2.2.3, and scikit-learn 1.8.0. The library behavior described in the article was checked against official documentation on October 12, 2026.

`assert` is a statement that checks a condition and stops execution if it is false. Here, before examining prediction scores, we use it to check that <strong>data shapes and the correspondence between cars are preserved</strong>. `train_test_split` is a function that splits inputs and targets together. `test_size=4` means holding out four records, rather than an evaluation proportion.[^split]

The complete code follows the sequence below. Manual preprocessing is used only to inspect the data structure; from cross-validation onward, we use the original input table and a Pipeline.

1. Create fictional data and split inputs and targets together.
2. Use manual preprocessing to check shapes, category columns, and row correspondence.
3. Define `make_model`, then cross-validate the baseline and Ridge created with this function on the same splits.
4. Use `GridSearchCV` to select `alpha` and refit on all training data.
5. Predict the separate evaluation data and a new car, and inspect the stored statistics.
6. Finally, run a separate, simple calculation for comparison with Rust.

[Download the complete Python code](/car-mpg-pipeline/examples-en.py)

```python
"""Review preprocessing, cross-validation, and Ridge with fictional cars.

The required make_model function is defined in this file, so no separate helper file is needed.
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

# 3. Group the repeated preprocessing and regression configuration in the make_model function.
# Pass a Ridge or LinearRegression object as regressor.
# include_origin determines whether to include region as an input; its default is True.
def make_model(regressor, include_origin=True):
    """Return an unfitted Pipeline combining preprocessing and the supplied regression model.

    Always include weight standardization; add region encoding when include_origin=True.
    This function only creates the configuration. A later fit learns statistics and coefficients.
    """
    # Create a new list and scaler on each call. Each entry is (name, transformer, target columns).
    transformers = [("weight", StandardScaler(), ["weight"])]
    if include_origin:
        # The append method adds the region transformation to the end of the current list.
        transformers.append((
            "origin",
            OneHotEncoder(
                categories=[[1, 2, 3]],  # Previously allowed codes for the single origin column.
                handle_unknown="error",  # Report an error for codes outside the allowed list.
                sparse_output=False,  # Return an ordinary two-dimensional NumPy array.
            ),
            ["origin"],
        ))
    # ColumnTransformer combines the results. remainder="drop" excludes columns not specified.
    preprocess = ColumnTransformer(transformers, remainder="drop")
    # Pipeline runs preprocessing and the model in order. This function defines the step names.
    return Pipeline([
        ("preprocess", preprocess),
        ("model", regressor),
    ])


# Calling the function only creates the configuration. Cross-validation below uses the original train_frame.
# alpha is regularization strength, not a learning rate. include_origin=True uses standardized weight and region indicators together.
cv_model = make_model(Ridge(alpha=1.0), include_origin=True)
# The baseline is the reference model. include_origin=False excludes region and uses only weight.
baseline = make_model(LinearRegression(), include_origin=False)

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

# Uncomment the next two lines to inspect the location and implementation of make_model defined in this file.
# print(inspect.getfile(make_model))  # Inspect the file containing the function definition.
# print(inspect.getsource(make_model))  # Inspect the function body when Python source is accessible.

# 6. A separate calculation for comparison with Rust. These are arbitrary saved values, not fitted values above.
# This checks weight-only standardization and prediction; origin encoding and Ridge fitting are excluded.
saved_mean, saved_scale, saved_w, saved_b = 3000.0, 500.0, -4.0, 25.0
demo_weight = 2800.0
demo_weight_scaled = (demo_weight - saved_mean) / saved_scale
demo_prediction = saved_w * demo_weight_scaled + saved_b
assert np.isclose(demo_prediction, 26.6)
print("Prediction checked with arbitrary saved values (mpg):", demo_prediction)
```

Running the fictional example selected an `alpha` of `0.01`, with a mean cross-validation MSE of about `0.5831` and a separate evaluation MSE of about `0.2756`. The new request's prediction was about `27.7189 mpg`. These values were calculated from the fictional data above, so performance on real cars must be evaluated separately using actual measurements.

The separate evaluation weights in this example also include values below the training minimum. The smallest value after the default MinMax transformation is therefore about `-0.1765`. As explained earlier, this follows from applying the training reference unchanged to new data.

## 14. Rust also predicts new inputs using stored statistics

The essential behavior of Python's Pipeline was to store the statistics and coefficients established during training, then read and use those values during prediction. Rust can likewise store the required values in a <strong>`struct`</strong>, a structure that groups several values under field names.

The code below demonstrates only weight standardization and prediction. `f64` is a 64-bit floating-point type for representing fractional values, and an `impl` block defines methods for a structure. `&self` is a borrowed reference used to read the current model. This method changes neither the stored mean nor the coefficients.[^rust]

To demonstrate the execution flow, we choose an arbitrary mean of 3000, divisor of 500, coefficient of −4, and intercept of 25. These are not values copied from the fitted Ridge model above. For weight 2800, the standardized value is $(2800-3000)/500=-0.4$, and the prediction is $(-4)\times(-0.4)+25=26.6$. This matches the final calculation in the complete Python code.

| Shared behavior | Difference in this example |
| --- | --- |
| Transform a new weight using the stored mean and divisor. | The complete Python example uses the library's Pipeline; the Rust example calculates directly inside a structure. |
| Predict using the stored coefficient and intercept. | The Rust example explicitly declares field types and a reference for reading the model. |
| Do not recalculate statistics for every prediction. | The Rust example does not implement coefficient fitting or `origin` encoding. |

[Download the Rust review code](/car-mpg-pipeline/saved-weight-model-en.rs)

```rust
// A standalone example that reads stored model state to make a prediction.
// The saved values below are arbitrary numbers, not actual training results.
// origin processing and Ridge fitting are excluded. Weight is measured in lb.
struct WeightModel {
    mean: f64,  // The mean weight stored during training.
    scale: f64, // The transformation divisor stored during training.
    w: f64,     // The coefficient multiplying transformed weight.
    b: f64,     // The intercept added at the end of the prediction equation.
}

impl WeightModel {
    // &self is a borrowed reference for reading the model. Stored values are unchanged.
    // weight is the new car's weight; the return value is predicted fuel economy.
    fn predict(&self, weight: f64) -> f64 {
        assert!(self.scale.is_finite() && self.scale > 0.0);
        let weight_scaled = (weight - self.mean) / self.scale;
        self.w * weight_scaled + self.b
    }
}

fn main() {
    // Create the object with the same arbitrary saved values as the final Python example.
    let model = WeightModel {
        mean: 3000.0,
        scale: 500.0,
        w: -4.0,
        b: 25.0,
    };
    let prediction = model.predict(2800.0);
    // Allow a small floating-point representation error when comparing with the expected 26.6.
    assert!((prediction - 26.6).abs() < 1e-12);
    println!("Prediction checked with arbitrary saved values (mpg): {prediction:.1}");
}
```

## 15. Key distinctions to revisit

| Common source of confusion | How to distinguish it |
| --- | --- |
| `fit` versus `transform` | `fit` learns transformation statistics; `transform` applies them. |
| Preprocessing validation data | Transform validation data too, but do not use them to learn the statistics. |
| Repeated cross-validation | Refit preprocessing and the model using only the training portion of each split. |
| The final scaler | Use the statistics relearned from all training data during refit. |
| Pipeline configuration | Code specifies preprocessing and the model. Ridge does not select a scaler automatically. |
| Category dtype versus one-hot encoding | One defines categories; the other creates input columns for those categories. |
| Combining tables | `concat` aligns row index labels. The information must belong to the same cars. |
| MSE versus negative MSE | Smaller MSE is better; larger negative-MSE scores are better. |
| `best_score_` | The mean CV score used for candidate selection, not a separate final evaluation score. |
| Ridge's `alpha` | Regularization strength, not the gradient-descent learning rate. |
| Squaring in L2 | Add a cost to the training objective; do not overwrite coefficients with their squares. |
| When derivatives are used | Calculate loss, then update using derivatives at the current coefficients. |
| $w$ versus `dw` | One is a prediction coefficient; the other is a name storing a loss derivative. |
| What standardization means | Express differences from the training mean in training-standard-deviation units, without imposing a fixed range. |
| Expressiveness versus training | Expressiveness is the range of possible relationships; training chooses a relationship within that range. |
| Dimensions and counts | Check original input count, transformed column count, coefficient count, and graph axes separately. |

## References

The official documents below support the explanations of function definitions and library behavior. The values from cross-validation, setting selection, and final evaluation in this article were calculated using the fictional data and code provided in Section 13.

[^pitfalls]: scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html). Consistent preprocessing, separation of training and evaluation, and data leakage.
[^scaler]: scikit-learn, [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). `fit`, `transform`, `mean_`, `scale_`, `ddof=0`, and constant-column handling.
[^pipeline]: scikit-learn, [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). Step execution, transformation during prediction, and nested setting names.
[^functions]: Python, [Defining Functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions). Function definitions, passing arguments, and return values.
[^preprocessing]: scikit-learn, [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html). Purposes and scope of standardization and scaling.
[^std]: NumPy, [numpy.std](https://numpy.org/doc/stable/reference/generated/numpy.std.html). Standard deviation and the meaning of `ddof`.
[^minmax]: scikit-learn, [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html). Training ranges and the `clip` setting.
[^onehot]: scikit-learn, [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html). Category encoding, `categories`, `sparse_output`, and `handle_unknown`.
[^structures]: Pandas, [Intro to data structures](https://pandas.pydata.org/docs/user_guide/dsintro.html), [DataFrame.to_numpy](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html). Row and column labels and conversion to arrays.
[^categorical]: Pandas, [Categorical data](https://pandas.pydata.org/docs/user_guide/categorical.html), [Categorical](https://pandas.pydata.org/docs/reference/api/pandas.Categorical.html), [DataFrame.astype](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html). Category definitions and dtype conversion.
[^dummies]: Pandas, [get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html). Category-indicator columns and default handling of missing values.
[^concat]: Pandas, [concat](https://pandas.pydata.org/docs/reference/api/pandas.concat.html). Concatenation direction and index alignment on the other axis.
[^linear]: scikit-learn, [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html). Least-squares linear regression, Ridge, and Lasso.
[^column-transformer]: scikit-learn, [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html). Column-specific preprocessing and access to fitted transformers.
[^ridge]: scikit-learn, [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html). The L2 objective, regularization strength, learned coefficients, and solvers.
[^cv-guide]: scikit-learn, [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html). Cross-validation, final evaluation data, and splitting strategies.
[^cv]: scikit-learn, [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html). Inputs, the returned dictionary, and scores and times for each split.
[^validation]: scikit-learn, [Official model-selection and validation implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_validation.py). Cloning models for each split and splitting inputs and targets by position.
[^kfold]: scikit-learn, [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html). Split count, shuffling, and reproducibility settings.
[^mse]: scikit-learn, [mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html). Calculating mean squared error.
[^scoring]: scikit-learn, [Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html). The scorer convention that larger values are better and the use of negative loss scores.
[^grid]: scikit-learn, [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). Candidate search, results attributes, and `refit`.
[^lasso]: scikit-learn, [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html). The L1 objective and coefficient fitting.
[^polynomial]: scikit-learn, [Polynomial regression](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions). Creating polynomial features from original inputs.
[^tree]: scikit-learn, [Decision Trees](https://scikit-learn.org/stable/modules/tree.html). Branching on input conditions and regression predictions.
[^mlp]: scikit-learn, [Neural network models](https://scikit-learn.org/stable/modules/neural_networks_supervised.html). Multilayer model architectures and representations.
[^inspect]: Python, [inspect](https://docs.python.org/3/library/inspect.html). Behavior and limitations of `getfile` and `getsource`.
[^python-files]: Python, [pathlib](https://docs.python.org/3/library/pathlib.html), [zipfile](https://docs.python.org/3/library/zipfile.html), [hashlib](https://docs.python.org/3/library/hashlib.html), [The module search path](https://docs.python.org/3/tutorial/modules.html#the-module-search-path). Preparing files and importing modules.
[^plotting]: NumPy, [linspace](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html); Matplotlib, [scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html), [plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html); Pandas, [head](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html), [round](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.round.html). Visualization inputs and inspecting output.
[^split]: scikit-learn, [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Splitting while preserving input-target correspondence.
[^rust]: Official Rust documentation, [Method Syntax](https://doc.rust-lang.org/book/ch05-03-method-syntax.html). Structure methods and `&self`.
