+++
title = "Machine Learning with the Titanic: From Data Preparation to Validation and Ensembles"
description = "Follow a passenger's information through a survival prediction to understand EDA, missing values, new features, data leakage, Pipeline, cross-validation, learning curves, tuning, and soft voting. Learn when preprocessing must be fitted and how the steps lead to a final submission, with concepts and essential code."
date = 2026-10-19T21:33:00+09:00
slug = "titanic-machine-learning-workflow"

[extra]
toc = true
styles = ["titanic-machine-learning-workflow/article.css"]
+++

Suppose we want to predict whether a Titanic passenger survived from their age, sex, and ticket class. We learn rules from records whose survival outcomes are known, then apply those rules to passengers whose outcomes are not provided.

Training a model is only part of the task. We also need to <strong>check whether the model predicts well for people it has never seen</strong>. Correct predictions on the training records do not tell us whether predictions on new records will also be correct. If information from the evaluation data enters preprocessing, we may also judge performance too favorably.[^pitfalls]

Understanding the records, creating useful features, and comparing models all serve this purpose. First, let us look at the whole process. Below, <strong>development data</strong> means the records used to build and choose a model. <strong>Final evaluation data</strong> means records kept separately to check performance after those choices are complete.

<figure class="titanic-figure">
<img src="/titanic-machine-learning-workflow/workflow-en.svg" alt="Split passenger records with known outcomes into development and final evaluation data. Within the development data, repeat EDA, cross-validation of preprocessing and models, and changes to features and settings. After making the choices, evaluate on the final evaluation data, then retrain on all labeled records and predict Kaggle's test.csv.">
<figcaption>Preprocessing rules are learned results, too. Rows used for evaluation must not help calculate those rules.</figcaption>
</figure>

## 1. Separate each passenger's inputs from the actual outcome

In machine learning, a <strong>sample</strong> is one item to learn from or make a prediction about. In the Titanic table, one passenger is one sample, and one row holds that passenger's record. Input information used for prediction is called a <strong>feature</strong>. The value the model is trying to predict is the <strong>target, or actual outcome</strong>.

These are the main columns in Kaggle's Titanic data.[^titanic]

| Column | What the record means |
| --- | --- |
| `Survived` | Survival outcome. `0` means died, and `1` means survived. |
| `PassengerId` | An identification number that distinguishes passenger records. |
| `Pclass` | The passenger's ticket class. `1`, `2`, and `3` mean first, second, and third class. |
| `Sex`, `Age` | Recorded sex and age. Age is measured in years. |
| `SibSp` | The number of siblings and spouses aboard with the passenger. |
| `Parch` | The number of parents and children aboard with the passenger. |
| `Fare` | The recorded passenger fare. |
| `Ticket` | The ticket number. |
| `Cabin` | The recorded cabin number. |
| `Embarked` | The port of embarkation. `C`, `Q`, and `S` mean Cherbourg, Queenstown, and Southampton. |
| `Name` | The recorded name. |

By convention, the input table is called `X`, and the targets are called `y`. We leave `Survived` out of `X` and put only `Survived` in `y`. If we supply the survival outcome as an input, even though it would be unknown when making a prediction, we cannot evaluate the model's predictive ability.

We also exclude `PassengerId` from the model inputs in this article. We keep this number to match each submitted prediction to the original passenger. These roles are separate so that removing the number from the inputs does not lose the information needed for submission.

<strong>Predicting one of a fixed set of categories, such as survived or died, is called classification.</strong> A model may first calculate a survival probability. For example, if it predicts a survival probability of `0.7` for a passenger, this is a different kind of value from the target `1`. The value `0.7` is the model's calculated probability, while `1` records that the passenger actually survived.

If we are predicting outcomes for three people, a single model call can return results for all three. <strong>The number of calls and the number of predictions are different.</strong>

| Call | What it returns in binary classification |
| --- | --- |
| `model.predict(X)` | One `0` or `1` per passenger. Three passengers produce three values. |
| `model.predict_proba(X)` | Probabilities for both categories for each passenger. Three passengers produce 3 rows × 2 columns. |

Here, `model` is a trained model object. A function belonging to an object is called a <strong>method</strong>; `predict` and `predict_proba` are methods. The probability columns follow the order stored in the model's `classes_`. If `classes_` is `[0, 1]`, the first column is the probability of death and the second is the probability of survival.[^classifier]

## 2. Decide what each part of the data will be used for

Kaggle's `train.csv` includes the targets, while `test.csv` does not give participants the targets. We therefore cannot calculate accuracy locally using only Kaggle's test file. <strong>We must split the records with known outcomes again to create data for training and evaluation.</strong>[^titanic]

In this article, we allocate 80% of the labeled records to development and 20% to final evaluation. An 80:20 split is one choice, not a fixed rule. What matters is which records are used for which purpose.

| Data | When and why it is used |
| --- | --- |
| Development data | Explore the records, create features, and compare models and settings. |
| Training portion within the development data | Learn preprocessing rules and the model for one comparison. |
| Validation portion within the development data | Calculate performance on rows not used for training in that comparison. |
| Final evaluation data | Check final performance after choosing the model and settings. |
| Kaggle's `test.csv` | Generate predictions for the submission. The targets cannot be compared locally. |

The training and validation portions within the development data change several times during cross-validation, which we will cover later. The final evaluation data stays separate throughout development. Evaluation data set aside in this way is called a <strong>holdout</strong>.

`train_test_split` is a <strong>function</strong> that splits several inputs using matching row positions. When we pass both `X` and `y`, it returns development inputs, evaluation inputs, development targets, and evaluation targets. Shuffling the inputs and targets independently could pair one passenger's information with another passenger's outcome.[^split]

The setting `test_size=0.2` reserves 20%. The setting `stratify=y` keeps the survival and death proportions as similar as possible in the two parts. The setting `random_state=42` controls randomness so that the split can be reproduced with the same data and execution conditions. The number 42 itself does not guarantee a better split or better performance.

## 3. In EDA, check what columns mean and what they actually contain

<strong>EDA stands for Exploratory Data Analysis.</strong> It means examining the table's structure and value distributions to find problems in the records or relationships that may help prediction.

Reading a file and checking column names, data types, and whether a target column exists are basic steps in understanding the data. After that, analyses that guide feature selection or transformation choices should use the development data. If we keep inspecting the final evaluation data's survival rates and prediction errors to change the model, we are using that data for development as well.[^selection]

### A missing value does not mean zero

A <strong>missing value</strong> means that a value is absent from the record. A missing age does not mean that the passenger was zero years old. Nor does a missing cabin number prove that the passenger had no cabin.

When we find missing values, we first check which columns contain them and how many there are. We then decide whether to keep a column, fill its missing values, or preserve the fact that a value was missing as a separate feature. We do not immediately discard every affected row or column just because many values are missing.

### Matching values do not necessarily identify the same passenger

Several passengers can have the same age, sex, and class. Repeated feature values are not automatically record errors. If the same passenger's record was accidentally entered twice, however, we need to handle it so that it does not appear in both training and evaluation.

Before checking duplicates, decide <strong>which matching values would make two rows count as the same record</strong>. If the comparison includes an identifier that differs in every row, rows whose other information matches may not be detected as duplicates. For candidates found by excluding the identifier, we need to return to the meaning of the original records.

### A histogram shows how many records fall in each interval

In a histogram of `Fare`, the horizontal axis shows fare intervals, and the vertical axis shows the number of passenger records in each interval. The setting `bins=30` means counting the values in 30 intervals. One bar does not represent one passenger.

If records cluster around low fares while a few large fares extend much farther, the distribution has a long right tail. This is a <strong>right-skewed distribution</strong>. A large value is not necessarily an error. First check whether the fare is plausible and whether the records use the same units.

When comparing survival rates, examine the number of people as well as the proportion. In fictional groups used for illustration, `1 survivor out of 1 person` and `100 survivors out of 100 people` both give 100%, but the amounts of evidence are very different. A proportion observed in a small group should not immediately become a rule for all passengers.

## 4. Create useful features from existing records

Calculating or combining existing columns to create new inputs is called <strong>feature engineering</strong>. More important than making many new columns is being able to explain what each column's values mean.

### Add up family members aboard

`SibSp` counts siblings and spouses, while `Parch` counts parents and children. Adding the passenger to these two counts gives the following feature.

<p class="titanic-formula"><code>FamilySize = SibSp + Parch + 1</code></p>

For a passenger with `SibSp=1` and `Parch=2`, the calculation is `1+2+1=4`. The final `1` is the passenger. Thus, `FamilySize=1` means that these two columns record no family members traveling with the passenger. It does not mean that we know the total number of companions, including friends.

We can use validation to compare keeping `SibSp` and `Parch` alone with also adding `FamilySize`. A new feature may summarize the two original values, but providing the same information again does not necessarily improve performance.

### Preserve whether a cabin number was recorded

We can define `Has_Cabin` as `1 if a cabin number is recorded, 0 if it is missing`. The model can then use the presence of the record without receiving the original text directly.

Here, `0` means only that <strong>cabin information was not recorded</strong>. Treating it as proof that someone was a third-class passenger or belonged to a particular social group would add information that the record does not contain. We must examine any relationship with ticket class in the development data and validate whether the feature helps prediction.

### Name length depends on the cleaning rules

Another possible feature is the number of characters left after removing a surname and title. Its value depends on whether we count spaces, include names in parentheses, and how we handle `Mr.` and other titles.

For example, the fictional string `" Lee"` has four characters if we count its leading space. After removing surrounding spaces, `"Lee"` has three characters. A name such as `name_cnt` therefore does not explain the calculation by itself. We first need to define <strong>which text remains and what is counted</strong> so that the feature has a consistent meaning.

These calculations use only one passenger's row. They differ from calculations that need averages or frequencies from other passengers. This distinction matters in the next preprocessing step.

## 5. Learn how to fill missing values from the training data

Filling missing values with replacements is called <strong>imputation</strong>. One possible replacement for a missing age is the median. <strong>The median is the middle value after sorting values from smallest to largest.</strong> If there is an even number of values, we use the average of the middle two.

For illustration, suppose the ages in the training portion are `20, 30, 40` years. The median is 30 years. If we mix in ages `80, 90` from the validation portion, the median of all five values becomes 40 years. The validation portion has now influenced the value used to fill missing ages.

<strong>Data leakage occurs when information intended for evaluation enters model training or the model-building process.</strong> It is not limited to supplying the target directly. Using validation input values to determine preprocessing rules also lets validation information participate in training.[^pitfalls]

If we calculate a median of 30 from the training portion, we should therefore fill missing ages in both the training and validation portions of that comparison with the same value, 30. We do not recalculate the median on the validation data. This keeps the evaluation conditions consistent with predicting a new passenger's outcome.

### We can calculate a separate age median for each ticket class

Instead of calculating one median across all training rows, we can calculate a median for each `Pclass`. The following <strong>fictional results</strong> illustrate the structure of this calculation.

| Ticket class in the training data | Median age among passengers of that class whose ages are recorded |
| --- | ---: |
| First class | 40 years |
| Second class | 30 years |
| Third class | 20 years |

If a validation passenger is in second class and their age is missing, we look up the second-class value, 30, in the stored table and use it. The validation passengers' ages do not change this rule.

pandas is a collection of tools for working with tables in Python. A table with rows and columns is called a <strong>DataFrame</strong>. Data that pairs each value with an identifying label called an index is called a <strong>Series</strong>.

Running `groupby("Pclass")["Age"].median()` on the training table returns <strong>a Series whose index contains ticket classes and whose values contain each class's median age</strong>. Applying the `map(medians_table)` method to the validation table's `Pclass` column looks up the median for each passenger's class. The result is a Series with the same length as the number of passengers, retaining the original row index. Passing it to the `Age` column's `fillna` method fills only missing cells while preserving ages already recorded.[^pandas-map]

If every training age for a class is missing, that class's median cannot be calculated. We need to decide on a fallback in advance, such as the age median across the entire training portion. If all ages in that portion are also missing, there is no median to use. We then need a policy such as excluding the feature or using a separate fixed value.

### Missing ports and fares also need rules

For categorical values such as a port, we can use the <strong>mode</strong>. The mode is the value that appears most often. If `S` occurs most often in the training portion, we fill missing ports with `S`. This is a replacement for the model's input, not a discovery of the passenger's actual port.

A logarithm alone does not fill a missing `Fare`. We can choose an order such as filling missing fares with the training median first, then applying the log transformation. When comparing models, we also need to check whether each can accept missing values. For example, logistic regression needs numeric inputs with missing values handled, while current random forests and XGBoost can handle missing values themselves.[^imputer][^forest][^xgb-api]

### Ticket frequency is also learned from the training data

We can create a feature that counts records with the same `Ticket`. The expression `groupby("Ticket").size()` returns <strong>a Series whose index contains ticket numbers and whose values contain the number of rows for each ticket</strong>.[^pandas-size]

Suppose an illustrative training table contains ticket A twice and ticket B once. The stored mapping is `A → 2` and `B → 1`. A validation passenger with ticket A also receives 2. We do not add that validation passenger and use 3.

The precise meaning of this feature is therefore <strong>the number of occurrences of the same ticket number in the reference training table</strong>. We cannot assume it equals the actual size of a traveling group among all passengers. Training rows include themselves in the count, while validation rows receive only the count in the reference training table.

If we choose to assign 1 to an unseen ticket, that 1 is a default value, not a verified passenger count. Another possible policy is `0=not seen in training`. Whichever policy we use, we need to keep the feature's meaning consistent and compare its effect using the same validation procedure.

## 6. Log transformations and scaling do different jobs

### A log transformation reduces the gaps between large values

When fares cover a wide range, a log transformation is one candidate to consider. <strong>A logarithm tells us what power of a base number produces a given number.</strong> For example, using base 10, the logarithms of 10, 100, and 1000 are 1, 2, and 3. Values that increase tenfold become equally spaced.

NumPy's `np.log1p` is <strong>a function that adds 1 to its input and takes the natural logarithm</strong>. The natural logarithm uses the number `e`, approximately 2.718, as its base. Given an array, the function transforms each element and returns an array of the same shape. Adding 1 first lets a fare of 0 be calculated as `log(1)=0`.[^log1p]

| Illustrative fare value | Value after `log1p`, to three decimal places |
| ---: | ---: |
| 0 | 0.000 |
| 10 | 2.398 |
| 100 | 4.615 |
| 1000 | 6.909 |

The original gap between 10 and 100 is 90, and the gap between 100 and 1000 is 900. After transformation, the gaps are approximately 2.217 and 2.294. The gaps between large values have shrunk considerably.

This does not guarantee a symmetric, bell-shaped <strong>normal distribution</strong>. Logistic regression does not require its input columns to follow a normal distribution, either. A log transformation changes how the fare is represented to the model, so we use validation results to decide whether to keep it.[^logistic-model]

Trees in random forests and XGBoost mainly split rows according to whether values are larger or smaller than a threshold. Reducing the numeric range alone therefore does not improve predictions. Log transformations and scaling need not be required steps for tree models, although implementations and numerical handling can produce differences in results.[^scaling]

### Scaling adjusts the numeric ranges of columns

Ages may be in the tens, family sizes may be a few people, and fares may cover a wider range. <strong>Scaling</strong> is preprocessing that adjusts these numeric ranges. Models such as logistic regression make predictions by learning <strong>coefficients</strong>, which are numbers multiplied by each input value. Adjusting the numeric ranges of columns can help their training calculations proceed more effectively.[^scaling]

A common method called standardization subtracts a column's training mean from each value, then divides by its training standard deviation. The <strong>mean</strong> is the sum of the values divided by their count. The <strong>standard deviation</strong> measures how spread out the values are around that mean.

scikit-learn is a collection of Python tools for preprocessing, model training, and evaluation. These tools use a <strong>class</strong> to define the capabilities and settings of new objects. `StandardScaler` is a class for creating a standardization tool; the resulting object stores means and standard deviations and performs transformations. The mean and standard deviation are calculated only from training rows, and validation rows are transformed using the stored values. Standardization does not automatically turn a column into a normal distribution, either.[^scaler]

## 7. Turn text into numeric columns with consistent meanings

The essential code in this article passes a numeric table to the model. Values that distinguish different kinds, such as the `C`, `Q`, and `S` in `Embarked`, are called <strong>categorical values</strong>.

If we simply replace the text with `C=0`, `Q=1`, and `S=2`, a model that uses numeric distances may treat Q as halfway between C and S. No such numeric distance has been defined between these ports.

<strong>One-hot encoding</strong> creates a column for each category. It puts 1 in the matching column and 0 in the others.[^encoder]

| Port of embarkation | Port_C | Port_Q | Port_S |
| --- | ---: | ---: | ---: |
| C | 1 | 0 | 0 |
| Q | 0 | 1 | 0 |
| S | 0 | 0 | 1 |

The number of rows remains the number of passengers, while the number of columns representing the port increases. `Pclass` is also recorded as a number, but it represents ticket class. The classes have an order, yet we need not assume that the gaps between them are equal-sized differences like differences in age or fare. The essential code below also encodes `Pclass` as a category.

### Training and validation columns must match in name and order

Suppose the training data contains C, Q, and S, while the validation data contains only C and S. Applying `pd.get_dummies` separately to each table may produce three columns for training and two for validation. Even if we force the column counts to match, a different order makes the model read the values with different meanings.[^dummies]

`OneHotEncoder` is a class that stores a category list during training and applies the same structure to other data. With `handle_unknown="ignore"`, an unseen port causes all port-related columns to be 0. This does not mean the model has learned what the new port means. It is a rule for maintaining the defined input structure.[^encoder]

Category lists learned in different cross-validation fits may differ. The requirement is not to force different models to have the same column count. It is that <strong>each model receives its validation data in the structure used for its own training</strong>.

## 8. Pipeline fits preprocessing and the model together

The calculations so far fall into two groups.

| Calculations using only the current passenger's row | Calculations that learn rules from other training rows |
| --- | --- |
| Adding family counts; checking whether a cabin was recorded | Calculating age and fare medians |
| Applying predefined name-cleaning rules | Calculating the most frequent port |
| Applying a predefined log transformation to the fare | Storing ticket frequencies, means and standard deviations, and encoding category lists |

The calculations on the left do not themselves consult information from other validation rows. Those on the right depend on which rows provide the reference. However, if we use data to <strong>choose a method</strong>, such as whether to apply a log transformation, that choice is also part of development.

Preprocessing tools distinguish learning these rules from applying them.[^imputer][^pipeline]

| Operation | What it actually does |
| --- | --- |
| Preprocessing `fit` | Calculates and stores rules, such as medians and category lists, from training data. |
| Preprocessing `transform` | Transforms inputs using rules already stored. |
| Preprocessing `fit_transform` | Learns the rules, then transforms the inputs using them. |
| Model `fit` | Learns prediction rules from transformed inputs and targets. |
| Model `predict` | Uses the learned rules to predict a category for each row. |

In scikit-learn, <strong>`Pipeline` is a class that connects processing steps in order</strong>. Combining preprocessing and a model in one object means that the necessary steps run together when we ask that object to train or predict.

<strong>Creating an object with `Pipeline(...)` does not yet learn medians or model rules.</strong> At this point, we are choosing the tools and their order. Learning starts when we call `pipeline.fit(X_train, y_train)`.

That call first learns preprocessing rules from the training inputs, transforms those inputs, and then trains the model using the transformed result and the targets. In contrast, `pipeline.predict(X_valid)` transforms validation inputs using the stored rules and makes predictions. It does not relearn preprocessing rules from the validation inputs.[^pipeline]

To relearn preprocessing rules in every cross-validation fit, <strong>we must include the steps that calculate medians and category lists inside the Pipeline.</strong>

## 9. In cross-validation, learn preprocessing from scratch each time

A score from a single split can change depending on which passengers enter training and which enter validation. <strong>Cross-validation, or CV,</strong> trains and evaluates models across multiple splits.[^cv]

In five-fold cross-validation, we divide the development data into five groups. Each group is called a <strong>fold</strong>. We can label them A through E to show how they are used.

| Comparison | Groups used for training | Group used for validation |
| --- | --- | --- |
| 1 | B, C, D, E | A |
| 2 | A, C, D, E | B |
| 3 | A, B, D, E | C |
| 4 | A, B, C, E | D |
| 5 | A, B, C, D | E |

Each row is <strong>a separate comparison that fits preprocessing and the model from scratch</strong>. We do not continue training the first model in the second comparison. If we did, we could not treat group B as unseen validation data in the second comparison, because that model already trained on B in the first.

`StratifiedKFold` is a class that forms folds while considering target category proportions. The setting `n_splits=5` requests five folds, and `shuffle=True` shuffles before splitting. It keeps survival and death proportions as similar as possible across folds, but the proportions may not be identical because passenger counts are whole numbers.[^stratified]

Its `split(X, y)` method yields pairs of <strong>arrays of row positions</strong> for training and validation. It determines the split positions; it does not train a model. To select rows at these positions in a pandas table, we use `iloc`, which selects by position. This differs from `loc`, which selects by index label.

### Preprocessing before creating the folds reverses the required order

Suppose we calculate the median age across all development data, then divide it into A through E. In the first comparison, A is the validation group, but A's ages already contributed to the median. Even if the separate 20% holdout is correctly kept aside, <strong>validation information within the development data has entered preprocessing</strong>.

In the correct first comparison, we calculate medians and category lists from B, C, D, and E. We use those rules to transform B, C, D, and E and train the model, then apply the same rules to A to make predictions. In the second comparison, we learn preprocessing and the model again using A, C, D, and E.

We therefore give the cross-validation tool <strong>the original inputs and a Pipeline containing preprocessing and the model, rather than only a preprocessed table and a model</strong>. This ensures that every step requiring learning runs within that fold's training portion.[^compose]

### Matching category proportions does not solve every validation problem

Passengers from the same family or sharing a ticket may appear on both sides of a split. Whether this is appropriate depends on the intended prediction task. Predicting other passengers whose records were collected under the same conditions as our current records is different from predicting entirely unseen families or groups. To evaluate the latter, we should consider splitting by group so that the same group does not appear on both sides.[^cv]

Cross-validation does not automatically remove preprocessing leakage, incorrect records, or differences in the conditions under which data was collected. <strong>The split method needs to match the data we intend to predict in the future.</strong>

## 10. Accuracy and loss evaluate different aspects of predictions

To choose a model, we need to decide what counts as doing well. That criterion is an <strong>evaluation metric</strong>.

### Accuracy is the proportion of passengers predicted correctly

If a model correctly predicts survival outcomes for 80 of 100 illustrative evaluation records, its accuracy is `80÷100=0.8`, or 80%. The function `accuracy_score` takes the actual targets and predicted categories and returns this proportion.[^accuracy]

Suppose the cross-validation scores are the illustrative values `0.78, 0.82, 0.80, 0.79, 0.81`. Their mean is `4.00÷5=0.80`. We examine each fold's score as well as the mean. Two models can have similar means while one performs much worse on a particular fold.

Before comparing performance, we also need a simple reference, called a <strong>baseline</strong>. Suppose more passengers died than survived in an illustrative training portion. From then on, this baseline predicts death for every new passenger. If 70% of the evaluation passengers also actually died, this simple model has 70% accuracy. We should therefore compare a trained model's accuracy with the baseline's score on the same evaluation data.

scikit-learn's `DummyClassifier` is a class for creating these simple reference models. A baseline needs to be evaluated on the same training and validation splits for the comparison to be meaningful.[^dummy]

### Log loss evaluates how much probability the model gives the actual outcome

For a passenger who survived, models predicting survival probabilities of 0.6 and 0.9 may both choose survival. Accuracy counts both as correct, but the probabilities they calculated differ.

<strong>Loss</strong> measures how far a prediction is from the target, using a number defined so that smaller is better. Classification <strong>log loss</strong> uses the probability assigned to the actual outcome.[^logloss]

First, let `p` be the model's survival probability and `q` be the probability it assigns to the category that actually occurred.

- If the passenger survived, `q=p`.
- If the passenger died, `q=1-p`.

One passenger's log loss is `−log(q)`. Here, `log` is the natural logarithm introduced earlier. For a probability greater than 0 and no greater than 1, its natural logarithm is at most 0. With the minus sign, loss approaches 0 as the probability assigned to the actual outcome approaches 1, and loss grows as that probability becomes smaller.

| Actual record | Survival probability `p` | Probability of the actual category `q` | Approximate log loss for one passenger |
| --- | ---: | ---: | ---: |
| Survived | 0.60 | 0.60 | 0.511 |
| Survived | 0.90 | 0.90 | 0.105 |
| Died | 0.60 | 0.40 | 0.916 |
| Died | 0.99 | 0.01 | 4.605 |

These are all fictional predictions used to explain the calculation. The last row has a large loss because the model treated a passenger who died as almost certain to survive. Overall log loss is the mean: add each passenger's loss and divide by the number of passengers. Actual implementations use numerical handling to prevent zero from entering the logarithm.

We pass the targets and <strong>predicted probabilities</strong> to the `log_loss` function. Supplying only final decisions converted to `0` or `1` loses the original probability information. Accuracy and log loss assess different properties, so a model with higher accuracy does not necessarily have lower log loss.

## 11. Distinguish the rules learned by the three models

Logistic regression, random forests, and XGBoost can all be used for this survival classification problem. However, they learn different kinds of rules from the inputs and targets.

### Logistic regression learns a coefficient for each feature

A <strong>coefficient</strong> is a number multiplied by an input value. Logistic regression calculates a score by multiplying inputs by their coefficients, adding the results, and adding an <strong>intercept</strong>, a value that shifts the starting point. It turns that score into a survival probability between 0 and 1, then adjusts the coefficients and intercept to give high probabilities to the training targets.[^logistic-model]

This model contains no decision trees. The setting `max_iter` limits how many times the calculation algorithm may iterate to adjust coefficients. It does not count trees or passengers.

### A random forest averages probabilities from several trees

A <strong>decision tree</strong> is a model that splits rows using input conditions and predicts at the final points. A component that checks a condition or returns an output within a tree is called a <strong>node</strong>. A final node that does not split further is called a <strong>leaf</strong>.

A random forest trains several trees by sampling training rows while allowing the same row to be selected more than once, and adding randomness to the features considered at splits. Each tree is a complete model that predicts survival. In scikit-learn, a classification random forest uses <strong>the average of the category probabilities calculated by its trees</strong> for its final prediction.[^forest]

Adding trees can help reduce prediction fluctuations caused by the random choices. The additional benefit may shrink, however, while computation and model size increase. Increasing the tree count alone does not make validation scores keep rising or eliminate all overfitting.

### XGBoost adds trees that correct the current prediction

XGBoost is a library that provides <strong>boosting</strong> models. In the tree boosting discussed here, each new tree is trained to reduce the loss of the current overall prediction, and its output is added to the overall score. This process repeats.[^xgb-model]

For binary classification, the model uses current survival probabilities and actual targets to calculate which direction of score change would reduce loss. The new tree uses this information to learn score changes to add to the overall prediction. The scores and probabilities used here have not yet been converted into final categories.

One round of adding a new tree is called a <strong>boosting stage</strong>. Even within one stage, multiple nodes inside that new tree are learned. The <strong>learning rate</strong> controls how much of the new tree's output is applied. A smaller weight makes each correction smaller, so it needs to be considered together with the number of stages required.

| Unit | What it refers to |
| --- | --- |
| One node | One branching or output point inside a tree |
| One tree | One individual model with its own complete set of branches and leaves |
| One random forest | Several classification trees plus a rule for averaging their probabilities |
| One XGBoost classification model | Trees added across multiple stages plus a rule for combining their outputs |
| One model `fit` call | One call that trains that entire model |

For example, a single `fit` call for a random forest creates the specified number of trees internally. It does not mean that the user must call `fit` separately for every tree. See [Decision Tree Basics](/en/decision-tree-basics/) for how a tree works internally, and [Random Forests and Gradient Boosting](/en/random-forest-gradient-boosting/) for how several trees are combined.

## 12. Read the horizontal axis before interpreting a learning curve

A graph may show loss or accuracy, but its interpretation depends on the horizontal axis. Figures commonly called learning curves can represent different comparisons such as these.[^curves]

| Horizontal axis | What changes | What to examine |
| --- | --- | --- |
| Number of training records | How many passengers are used for training | How training and validation performance change as data increases |
| A setting such as tree depth | The range of rules the model can represent | Which settings improve validation performance |
| Optimization iteration or boosting stage | How far one model has progressed in training | How loss changes as training proceeds |

scikit-learn's `learning_curve` function calculates training and validation scores while varying <strong>the number of training records</strong>, as in the first row. The `validation_curve` function calculates scores while varying <strong>a specified setting</strong>, as in the second row. Check both the function name and the graph's actual horizontal axis.

### Look at training and validation curves together

<strong>Underfitting</strong> means that a model has not learned enough of the rules needed for the task. <strong>Overfitting</strong> means that a model follows details particular to the training records and consequently performs worse on new records.

| What we observe | Possibilities to examine first |
| --- | --- |
| Both training and validation accuracy are low, or both losses are high. | The model may be too simple, useful inputs may be missing, or training may not have progressed far enough. |
| Training accuracy is high, but validation accuracy is low. | The model may have learned rules that work well only for training. Also examine the split and differences in the data. |
| Further training reduces training loss, but validation loss keeps increasing. | Check whether the additional training is failing to help predictions on new data. |

Lower loss is better, while higher accuracy is better. Reversing these directions reverses the interpretation of the same curve. We should also avoid concluding that a model is overfitting from one small fluctuation; check whether similar patterns appear across folds.

### “Training more” means different things for different models

For logistic regression, `max_iter=10` allows at most 10 optimization iterations in one `fit` call. <strong>It does not guarantee that all 10 iterations occur.</strong> The setting `warm_start=True` reuses previously learned coefficients as the starting values for the next fit.[^logistic-api]

Calling `fit` 30 times with these settings does not mean that one optimization process ran for exactly 300 consecutive iterations. It may stop early when the calculation is judged stable enough, and the algorithm's internal progress state may restart between calls. Labeling the horizontal axis `10, 20, …, 300 iterations` merely by counting calls may therefore misrepresent what was actually measured.

The process of coefficients becoming sufficiently stable is called <strong>convergence</strong>. If a convergence warning appears, check input scaling, the iteration limit, and the training settings. Hiding all warnings also hides clues about whether training finished properly.

Increasing the number of trees in a random forest does not continue training the same tree. It <strong>creates more separate, complete trees and includes them in the average</strong>. Increasing boosting stages adds trees that correct the current prediction. A tree-count graph therefore cannot be interpreted in the same way as a graph of logistic regression optimization iterations.

Boosting can also use <strong>early stopping</strong>, which stops adding stages when validation performance no longer improves. The data used to choose when to stop also participates in model selection, so it must be distinguished from data reserved for the final independent evaluation.[^early-stopping]

## 13. Hyperparameter tuning compares candidate settings with the same procedure

Values a model learns from data differ from settings a person chooses before training. Logistic regression coefficients and tree split conditions are learned from data. Settings such as how many trees to build or how deep they may grow are <strong>hyperparameters</strong>. Comparing candidate settings and choosing among them is called <strong>tuning</strong>.

| Setting | What it actually controls |
| --- | --- |
| `n_estimators` | In a random forest, the number of trees. In the basic binary XGBoost configuration discussed here, the number of boosting stages corresponds to the number of added trees. |
| `max_depth` | The maximum depth of one tree. Its first node, the root, has depth 0. The setting `max_depth=3` allows at most four layers, at depths 0, 1, 2, and 3. The actual tree may be shallower. |
| `min_samples_split` | The minimum number of samples required to attempt a split at a node in a random forest. |
| `min_samples_leaf` | The minimum number of samples that must remain in each leaf of a random forest. |
| `learning_rate` | How much of each new tree's correction is applied in boosting. |
| `subsample` | The fraction of rows used for training in each XGBoost boosting stage. |
| `colsample_bytree` | The fraction of features selected when building one XGBoost tree. |

Limiting depth or increasing the required number of samples in a leaf makes very detailed rules harder to create. Methods that restrict model complexity in this way are called <strong>regularization</strong>. Stronger restrictions are not always better. If they also prevent the model from learning necessary rules, underfitting may result.[^forest][^xgb-params]

### GridSearchCV compares every specified combination

`GridSearchCV` is a class that receives a model and candidate settings and compares them using cross-validation. It examines all combinations of the candidate values, so adding candidates increases computation.[^grid]

For example, suppose we specify three tree counts, three depths, two learning rates, two row fractions, and two feature fractions for XGBoost. The counts are calculated as follows.

<p class="titanic-formula"><code>Number of setting combinations = 3 × 3 × 2 × 2 × 2 = 72</code><br><code>Model fits in five-fold cross-validation = 72 × 5 = 360</code></p>

The 360 fits mean <strong>training the entire XGBoost model 360 times</strong>. They do not mean a total of 360 trees or nodes.

For a random forest, comparing three tree counts, four depths, and three minimum split sample counts gives `3×4×3=36` combinations. Five-fold cross-validation then requires `36×5=180` model fits. We can begin with a small candidate range, check which settings affect the results, and then expand it.

`RandomizedSearchCV` is a class that samples setting combinations from specified ranges or distributions and compares them a fixed number of times. It can limit the search effort when there are many candidates, but it does not mean every combination has been checked.[^random-search]

### Distinguish the best score from the final fitted model

The results of `GridSearchCV` include attributes with different meanings. An <strong>attribute</strong> is a value stored in an object.

| Attribute or setting | Meaning |
| --- | --- |
| `best_params_` | The settings selected from the candidates compared. |
| `best_score_` | The selected settings' mean cross-validation score. |
| `refit=True` | After selecting settings, fit again using all supplied development data. |
| `best_estimator_` | The model after that refit. If a Pipeline was supplied, it includes preprocessing. |

The preceding XGBoost example therefore adds one refit on all development data to its 360 cross-validation fits. The result is not created by joining the five fold models or automatically making them vote.[^grid]

The value `best_score_` has already been used to choose among many candidates. The chosen settings may include some that happened to work particularly well on those validation records. We should not interpret the value as a guaranteed accuracy on new data; we check it with the separate final evaluation.[^selection]

## 14. Use feature importance to guide further analysis

<strong>Feature importance</strong> provides numbers for examining which input features a trained model used. The meaning depends on how the importance was calculated.

A common random forest importance measure adds up how much splits improve the separation of categories. XGBoost's `gain` measure is based on how much splits using a feature improve the training objective. Different ways of calculating importance can produce different rankings.[^forest-importance][^xgb-api]

The attribute `feature_importances_` returns an importance for each feature actually supplied to the model. After one-hot encoding, the single `Embarked` column becomes several columns, so the names paired with importances must also be <strong>the column names after transformation</strong>. Attaching the original column names may mismatch names and values.

High importance does not prove that a feature directly causes survival. For example, an association between recorded cabin information and survival predictions may also reflect differences in ticket class or other records. A relationship useful for prediction must be distinguished from real-world cause and effect.[^causal]

### We can also examine how performance changes when values are shuffled

<strong>Permutation importance</strong> keeps the model fixed, shuffles the values of one column among rows, and compares how the evaluation score changes. If predictions worsen when we disrupt the pairing between passengers and that column's values, it is a clue that the model was using that relationship.[^permutation]

However, if another column contains similar information, the model may still predict using it after the first column is shuffled. A low importance in this case does not justify concluding that all of that information is useless.

This analysis can lead to new features. For example, if family information seems useful, we can examine record counts, survival rates, and prediction errors again for passengers with large and small families. After checking the relationship and proposing a feature, we compare performance before and after adding it using the same cross-validation procedure.

The cycle of <strong>data exploration → candidate features → model validation → importance and error analysis → feature changes</strong> takes place within the development data. If we keep changing the model after inspecting errors on the final evaluation data, that score can no longer be called a final evaluation independent of development.

## 15. Soft voting combines probabilities for the same passenger

An <strong>ensemble</strong> combines predictions from several models. A random forest is already an ensemble of trees, but we can also combine a complete random forest model with a complete XGBoost model.

<strong>Hard voting</strong> collects the final categories chosen by the models and takes a majority vote. <strong>Soft voting</strong> averages the models' probabilities for each category and selects the category with the highest average probability.[^voting]

These are <strong>fictional predictions for illustration</strong>, all for the same passenger.

| Model | Probability of death | Probability of survival |
| --- | ---: | ---: |
| Random forest | 0.70 | 0.30 |
| XGBoost | 0.10 | 0.90 |
| Equally weighted average | `(0.70+0.10)÷2=0.40` | `(0.30+0.90)÷2=0.60` |

The individual models' final decisions differ: one chooses death and the other survival. After averaging their probabilities, however, the survival probability of 0.60 is higher. Soft voting therefore predicts survival. In binary classification, this connects to checking whether the survival probability exceeds 0.5; exactly equal probabilities follow the implementation's tie-breaking rule.

We are averaging <strong>probabilities from different models for the same passenger</strong>. We are not averaging probabilities across different passengers and giving everyone the same prediction.

Models can also receive different weights. For example, with a weight of 1 for the random forest and 2 for XGBoost, the survival probability becomes `(1×0.30+2×0.90)÷(1+2)=0.70`. These weights are another setting to compare and choose using the development data.

### Check separately whether combining probabilities improves performance

Combining models may help if they make mistakes in different places. Conversely, the average may hurt if both make similar mistakes for the same passengers, or if one gives excessively high probabilities to incorrect predictions. We therefore compare individual models and ensembles using the same development validation procedure. Soft voting depends on the quality of predicted probabilities as well as accuracy.[^voting-guide]

`VotingClassifier` is a class for creating such voting models. With `voting="soft"`, calling `fit` copies the settings of the supplied models and <strong>trains new models again</strong>. Even if the supplied models have already been trained, it does not simply preserve and combine their learned results.[^voting]

To cross-validate models that need preprocessing, we can include preprocessing in each model's Pipeline and validate the entire voting model. Keep the structure clear: one final voting model contains one random forest and one XGBoost model, and each of those contains several trees.

## 16. After final evaluation, retrain the model for submission

Once we have chosen features, preprocessing, a model, and settings using the development data, we prepare a model trained on all development records. Running `GridSearchCV` with `refit=True` already gives us a model that has completed this step.

We then predict the final evaluation inputs set aside at the start and compare those predictions with the saved targets. We must not relearn preprocessing here. We apply the medians, category lists, and model stored during development.

<strong>This score checks the chosen configuration on new records.</strong> Because the evaluation table is finite, the score is not guaranteed to match actual future performance exactly. If we keep changing settings in response to this result, the evaluation table is also being used for selection, so later scores no longer have the same meaning as an independent evaluation.[^cv]

After settling on the model configuration and preparing to submit, we can fit preprocessing and the model from scratch using <strong>all records with known outcomes</strong> in `train.csv`. The 20% previously kept for final evaluation can now join the submission training data. This step builds the final prediction model from more labeled records.

This retrained model uses different training data from the model evaluated earlier. We should not rename the earlier holdout score as an independent evaluation of the model retrained on all data.

We pass Kaggle's `test.csv` to the submission model. Preprocessing uses the rules newly learned from all labeled data; it does not relearn medians or category lists from the test file. Finally, we pair each passenger's predicted `0` or `1` with the original `PassengerId`.[^titanic]

The file `submission.csv` stores two columns: `PassengerId` and `Survived`. Its row count must match the number of test passengers predicted, and each identifier must stay paired with its prediction. In pandas, `to_csv(..., index=False)` prevents the DataFrame's row index from being written as an extra column.[^csv]

## 17. Essential code: keep the same order from preprocessing to submission

The following code runs <strong>preprocessing → setting selection through cross-validation → final evaluation → retraining on all labeled data → saving the submission file</strong>. It uses one random forest model and includes the derived features `FamilySize` and `Has_Cabin`.

Missing ages and fares are each filled with the corresponding training portion's median. Class-specific age imputation, ticket frequency, name length, and log transformation, discussed earlier, remain additional candidates to compare. They are not included in this basic model. An ensemble is also a separate candidate to choose after comparing it with an individual model through validation.

Three tools help connect the processing steps explained earlier to the code.

| Tool | Role and return value |
| --- | --- |
| `FunctionTransformer` class | Wraps a custom transformation function as a Pipeline step. The function below copies the input DataFrame and returns a DataFrame with two added features. |
| `ColumnTransformer` class | Applies different preprocessing to numeric and categorical columns, then joins their results side by side to make a numeric table. |
| `clone` function | Creates an unfitted object with a copy of the model's settings. It prepares the submission Pipeline for fresh training on all data. |

We explicitly specify which columns `ColumnTransformer` will use. Its default behavior excludes the remaining original columns, so `PassengerId`, `Name`, `Ticket`, and the original `Cabin` strings do not enter the model.[^column-transformer][^function-transformer][^clone]

In `model__max_depth`, `model` is the name assigned to the Pipeline's last step, and the text after the double underscore `__` is that model's setting name. Below, three depths and two minimum leaf sizes produce six combinations. Five-fold cross-validation trains 30 times, followed by one more fit with the selected settings.

Running the code requires pandas, scikit-learn, and Kaggle's `data/train.csv` and `data/test.csv`. It uses the original column names from these files. The function `make_features` does not calculate statistics across other rows, and `validate=False` allows it to receive the DataFrame with its column names intact.

<details class="titanic-code">
<summary>Expand the essential Python code</summary>

```python
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def make_features(frame):
    # Use only the current row's information. Copy to preserve the original table.
    result = frame.copy()
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    result["Has_Cabin"] = result["Cabin"].notna().astype(int)
    return result  # Preserve the order and number of passenger rows.


train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
X = train_df.drop(columns="Survived")  # Exclude the target from the inputs.
y = train_df["Survived"]             # Targets follow the same order as the input rows.

# Keep the final evaluation 20% out of the setting search below.
X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

numeric_columns = ["Age", "Fare", "FamilySize", "Has_Cabin"]
category_columns = ["Pclass", "Sex", "Embarked"]

# These objects configure preprocessing. They have not learned statistics yet.
category_steps = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocess = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_columns),
    ("category", category_steps, category_columns),
])
pipeline = Pipeline([
    ("features", FunctionTransformer(make_features, validate=False)),
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=1
    )),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    pipeline,
    param_grid={"model__max_depth": [3, 5, 7],
                "model__min_samples_leaf": [1, 3]},
    cv=cv, scoring="accuracy", n_jobs=-1, refit=True, error_score="raise",
)

# Fit preprocessing and the model from scratch on each fold's training rows.
# After selection, refit the Pipeline on all development data.
search.fit(X_dev, y_dev)
print("Selected settings:", search.best_params_)
print("Mean CV accuracy used to select settings:", search.best_score_)

# Evaluate the chosen configuration. Only make predictions here.
holdout_pred = search.predict(X_holdout)
holdout_proba = search.predict_proba(X_holdout)
print("Final evaluation accuracy:", accuracy_score(y_holdout, holdout_pred))
print("Final evaluation log loss:", log_loss(y_holdout, holdout_proba))

# Create a submission object and fit from preprocessing onward on all labeled data.
submission_model = clone(search.best_estimator_)
submission_model.fit(X, y)
test_pred = submission_model.predict(test_df)

# Save identifiers and predictions in the order of the test passengers.
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"].to_numpy(),
    "Survived": test_pred.astype(int),
})
submission.to_csv("submission.csv", index=False)
```

</details>

The setting `n_jobs=-1` runs the search's training jobs in parallel. The model itself uses `n_jobs=1` so that the two layers do not both multiply the number of concurrent jobs. The setting `error_score="raise"` exposes training errors immediately instead of hiding them. The setting `sparse_output=False` returns the one-hot result as an ordinary two-dimensional numeric array.

This code does not announce an actual Kaggle score or optimal settings in advance. The numerical examples illustrate the concepts, and the preprocessing and prediction flow in the code was checked on a separately created fictional table. Actual accuracy and which features are useful must be established from the CSV files used and the validation results.

[^titanic]: Kaggle, [Titanic — Data description and submission task](https://www.kaggle.com/competitions/titanic/data).
[^pitfalls]: scikit-learn, [Common pitfalls and recommended practices — Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
[^classifier]: scikit-learn, [RandomForestClassifier — predict, predict_proba, classes_](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
[^split]: scikit-learn, [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
[^selection]: scikit-learn, [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html).
[^pandas-map]: pandas, [Series.map](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html), [SeriesGroupBy.median](https://pandas.pydata.org/docs/reference/api/pandas.api.typing.SeriesGroupBy.median.html).
[^imputer]: scikit-learn, [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).
[^pandas-size]: pandas, [DataFrameGroupBy.size](https://pandas.pydata.org/docs/reference/api/pandas.api.typing.DataFrameGroupBy.size.html).
[^log1p]: NumPy, [numpy.log1p](https://numpy.org/doc/stable/reference/generated/numpy.log1p.html).
[^logistic-model]: scikit-learn, [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
[^scaling]: scikit-learn, [Importance of Feature Scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html).
[^scaler]: scikit-learn, [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
[^encoder]: scikit-learn, [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
[^dummies]: pandas, [get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html).
[^pipeline]: scikit-learn, [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
[^compose]: scikit-learn, [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html).
[^cv]: scikit-learn, [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html).
[^stratified]: scikit-learn, [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).
[^accuracy]: scikit-learn, [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).
[^dummy]: scikit-learn, [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html).
[^logloss]: scikit-learn, [log_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).
[^forest]: scikit-learn, [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
[^xgb-model]: XGBoost, [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).
[^curves]: scikit-learn, [Validation curves and learning curves](https://scikit-learn.org/stable/modules/learning_curve.html).
[^logistic-api]: scikit-learn, [LogisticRegression — max_iter, warm_start](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
[^early-stopping]: XGBoost, [Using the Scikit-Learn Estimator Interface — Early Stopping](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html#early-stopping).
[^xgb-params]: XGBoost, [Parameters for Tree Booster](https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster).
[^grid]: scikit-learn, [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
[^random-search]: scikit-learn, [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).
[^forest-importance]: scikit-learn, [Feature importances with a forest of trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html).
[^xgb-api]: XGBoost, [Python API — feature_importances_, n_jobs](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
[^causal]: scikit-learn, [Failure of Machine Learning to infer causal effects](https://scikit-learn.org/stable/auto_examples/inspection/plot_causal_interpretation.html).
[^permutation]: scikit-learn, [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
[^voting]: scikit-learn, [VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html).
[^voting-guide]: scikit-learn, [Voting Classifier — Soft voting](https://scikit-learn.org/stable/modules/ensemble.html#weighted-average-probabilities-soft-voting).
[^csv]: pandas, [DataFrame.to_csv](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html).
[^column-transformer]: scikit-learn, [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html).
[^function-transformer]: scikit-learn, [FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html).
[^clone]: scikit-learn, [clone](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html).
