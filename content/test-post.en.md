+++
title = "Pandas agg(), pivot_table(), and melt(): Reshaping Rows and Columns for Your Analysis"
description = "What stays the same, and what changes, when column values become column names and then return to values in rows? Animated tables connect the unit of analysis, aggregation, reshaping, and the meaning of an average."
date = 2026-09-29
updated = 2026-09-29
slug = "pandas-agg-pivot-table-melt"
aliases = ["/en/test-post/"]
+++

You could memorize `agg()` as aggregation, `pivot_table()` as spreading values across columns, and `melt()` as making a table longer. But those descriptions alone do not tell you **why you need to reshape this particular table, or which calculations make sense afterward**.

The purpose of reshaping a table is not to make it look nicer. It is **to compare the subjects of your analysis on a consistent basis and apply each calculation at the right level**.

Here, we follow one set of spending records to distinguish three things: **summarizing multiple records, changing where information is stored, and answering new questions using the reshaped table**.

**Define the analytical question → Choose the subjects and time period → Decide which distinctions to preserve → Aggregate as needed → Arrange the information in rows and columns → Check what the results mean**

## 1. Rows define what you read as one unit; columns define what you distinguish within it

A **DataFrame** is a table made up of rows and columns. In an analysis, though, their direction matters less than **what each one represents**.

The **unit of analysis** is the unit you want to compare or make a judgment about. It might be one customer when comparing spending patterns, one customer in one month when examining monthly changes, or one record when inspecting individual spending entries.

The following are **fictional records from the same analysis period**. All amounts are in South Korean won (KRW). The categories `food`, `electronics`, and `clothing` mean food, electronic goods, and clothing, respectively. We assume that each record belongs to exactly one category.

| record_id | customer_id | category | amount |
| --- | --- | --- | ---: |
| r1 | 101 | food | 10000 |
| r2 | 101 | food | 5000 |
| r3 | 101 | electronics | 30000 |
| r4 | 101 | clothing | 5000 |
| r5 | 102 | food | 7000 |
| r6 | 102 | clothing | 20000 |

At this point, one row represents **one spending record for a particular customer and category**, not one customer. There are two customers but six records. `record_id` identifies each record. With real order data, you would also need to check whether one record and one order represent the same unit.

The columns become clearer when we distinguish their roles.

| Column | Role in this table | Question it answers |
| --- | --- | --- |
| record_id | Identifies an individual record | Which record is this? |
| customer_id | Identifies customers and provides a grouping key | Whose record is this? |
| category | Classifies the type of spending | What kind of spending is this? |
| amount | Stores the measured value in the record | How much was spent? |

`customer_id` alone cannot identify a single row in the original table, because `101` appears four times. **Having a customer ID is different from having one row per customer.**

Here, **distinctions to preserve** means the fields we retain so that different pieces of information do not get mixed together. To calculate totals by customer and category, we must keep both customer and category. If we only need an overall total for each customer, we can combine the categories.

In Pandas, an **index** contains the row labels used to select or sort rows and is managed separately from ordinary columns. Placing a value in the index does not automatically aggregate the data or remove duplicates. Likewise, `reset_index()` moves the index into ordinary columns; it is not an aggregation that changes the unit of analysis.[^reset]

> **First decide, “What do I want to compare in each row?” Then ask, “Which pieces of information about that subject do I need to keep separate?”**

## 2. Watch the information move in an animated table

In the diagram below, select **② pivot** to move `food` from a value inside `category` to a column name in the result. Select **③ melt** to move that column name back into a value in `category`.

Follow **the connections between customer ID, category, and amount**, not just the numbers. You can play or pause the animation, adjust its speed, and use the progress slider to stop at any point.

<iframe src="/pandas-table-lab-en.html" title="Interactive diagram showing how customer IDs, categories, and amounts move through agg, pivot, and melt" loading="lazy" style="display:block;width:100%;height:1280px;border:0;max-width:100%;"></iframe>

[Open the diagram in a separate window](/pandas-table-lab-en.html)

The path **① → ② → ③** aggregates individual records, spreads the results across columns, and then unpacks them into rows. **Customer metrics** is a separate analysis that starts from the original records. You do not have to run every function in a single sequence.

## 3. agg(): Which distinctions do you keep, and which details do you combine?

### Keeping both customer and category

Start with the question, “How much did each customer spend in each category?”

Grouping records with the same `customer_id` and `category` and summing `amount` produces the following table. Calculating a summary value from multiple records is called **aggregation**.[^groupby]

| customer_id | category | total_amount |
| --- | --- | ---: |
| 101 | food | 15000 |
| 101 | electronics | 30000 |
| 101 | clothing | 5000 |
| 102 | food | 7000 |
| 102 | clothing | 20000 |

**KRW 10,000 in r1 + KRW 5,000 in r2 → Customer 101's food total of KRW 15,000**

One row now represents **one customer's total for a particular category**. The result no longer includes `record_id`, which distinguished individual records. The meaning of the amount has also changed from an individual record's amount to a total, so we name it `total_amount`.

The important change is not simply that the table has gone from six rows to five. It is that **two original records have been summarized into one total**. From 15,000 alone, you cannot tell whether the original amounts were 10,000 and 5,000, or 8,000 and 7,000.

### Keeping only customer

If the question is, “What are each customer's total, mean amount per record, and largest recorded amount?”, we do not need to distinguish categories. We group only by customer and apply three calculations to the same `amount` column.

| customer_id — index | total_amount | mean_record_amount | max_record_amount |
| --- | ---: | ---: | ---: |
| 101 | 50000 | 12500.0 | 30000 |
| 102 | 27000 | 13500.0 | 20000 |

One row represents one customer. The columns are **metrics: total amount, mean amount per record, and maximum recorded amount**. A metric here means a summary value calculated to answer an analytical question.

Customer 101's mean is `50000 / 4 = 12500`. This is the average of four records, not a monthly average or an average across categories.

The result no longer preserves the distinction between categories. Knowing a customer's total, mean, and maximum does not let you recover how much they spent on food.

> **Leaving a field out of the grouping keys is a decision to combine the distinctions that field represents. Do not casually combine distinctions you may need to compare later.**

## 4. pivot: What changes when values in a column become column names?

Now take the customer–category totals and arrange them into one row per customer.

| customer_id — index | food | electronics | clothing |
| --- | ---: | ---: | ---: |
| 101 | 15000 | 30000 | 5000 |
| 102 | 7000 | 0 | 20000 |

A **pivot** here rearranges values that distinguish records, such as categories, into column names. This is called **wide format**, because several categories for the same customer are collected in one row. By contrast, the earlier table that kept category in one column and used multiple rows is in **long format**.[^reshape]

### What changes is the role of values and names, not the position of an entire column

Before the pivot, `food` was a **value** inside the `category` column. After the pivot, `food` is the **name of a column** that stores amounts.

| Information | Location before the pivot | Location after the pivot |
| --- | --- | --- |
| Customer 101 | A value in customer_id | Row index 101 |
| Category food | A value in category | Column name food |
| Total 15000 | A value in total_amount | The cell where row 101 meets column food |

The ordinary `category` column has disappeared, but the category information itself has not. **It is now expressed through column names instead of cell values.**

The `food` column also means more than just “food.” In this table, it means **“the total amount this customer spent on food during the same analysis period.”** It has not become a 0/1 indicator of whether the customer bought something in that category.

**Before the pivot:** `(customer_id=101, category=food, total_amount=15000)`  
**After the pivot:** `The value at (row=101, column=food) is 15000`

Both representations express the same fact.

### The unit represented by a row differs from the level of detail in each amount

This is an easy point to miss.

After the pivot, one row represents one customer, but **each amount is still a customer–category total**. The three category amounts have not been added into a single number. They have been placed in separate cells within the same row.

A **two-row table aggregated into one overall total per customer** therefore differs from a **two-row table with category totals spread across columns**. Their row counts are the same, but the second table preserves the category distinctions.

> **Fewer rows do not necessarily mean that information has been summarized. Several amounts may simply have been spread across columns. Alongside “What does one row represent?”, ask, “What exactly does the amount in one cell measure?”**

### This is where pivot() and pivot_table() differ

If a table already contains exactly one amount for each customer–category combination, `pivot()` can rearrange it. In the original records, however, `101 × food` has two amounts. Calling `pivot()` directly on those records raises an error, because there is no single value to put in that cell.[^pivot]

Applying `pivot_table(index="customer_id", columns="category", values="amount", aggfunc="sum")` to the original records **calculates the customer–category totals and arranges them in one operation**.[^pivot-table]

**Individual records → [Aggregate] Customer–category totals → [Pivot] A category amount column for each customer**

Even when `pivot_table()` performs both operations together, it helps to keep them conceptually separate. **Summing the records summarizes information; moving category information into column names rearranges it.**

The electronics cell for customer 102 needs a separate explanation. There is no record for that combination in the original data. We fill it with 0 here on the assumption that collection is complete and the missing combination means no purchase. **Filling a missing cell with 0 is a separate judgment from reshaping the table.** An unknown value should not be treated as no purchase. Nor does this operation automatically add customers who never appear in the table.

## 5. melt(): Bring distinctions stored in column names back into data values

Suppose you now want to **treat category as one common field**, for operations such as selecting only electronics records or grouping customers by category.

In the wide table, the category names are spread across several column names. `melt()` collects those names into one column called `category` and the amounts from the cells into one column called `total_amount`.[^melt]

| customer_id | category | total_amount |
| --- | --- | ---: |
| 101 | food | 15000 |
| 102 | food | 7000 |
| 101 | electronics | 30000 |
| 102 | electronics | 0 |
| 101 | clothing | 5000 |
| 102 | clothing | 20000 |

**Column name food → Value food in category**  
**15000 in the food column → Value 15000 in total_amount**  
**Row index 101 → Value 101 in customer_id, identifying whose amount this is**

You can now filter on category or use `groupby("category")`. The wide table was not wrong. **The change makes it easier to work with category as a data value rather than as a column name.**[^layout]

Customer IDs repeat so that every amount stays connected to its customer. Here, `id_vars` specifies **the columns whose connection to each value must be preserved, even if their values repeat**. `value_vars` specifies **the columns whose names and values will be unpacked into multiple rows**.

This transformation calculates no new spending amounts. It represents each of three categories for each of two customers in its own row, producing `2 × 3 = 6 rows`. There are still two customers.

The 15,000 is also still a total. It does not split back into the original 10,000 and 5,000. **melt() can undo the layout change, but it cannot recover individual records that were lost during aggregation.**

In this example, both the original table and the melted result happen to have six rows. Yet one original row is **one spending record**, while one result row is **one customer–category total**. Matching row counts or similar column names do not mean that two tables contain the same data.

## 6. How should you arrange columns for your analytical goal?

### To compare customers, collect each customer's characteristics in one row

A wide table is convenient when you want to compare how customers distribute their spending.

Comparing `food`, `electronics`, and `clothing` within the same row shows **one customer's spending mix**. Comparing customers 101 and 102 down the `food` column shows **the difference between customers for the same category**.

It is also easy to calculate each customer's food spending share as `food / customer's total spending`. For customer 101, this is `15000 / 50000 = 0.3`, or 30%.

This structure also connects to machine learning inputs that produce one prediction per customer. Each row is a customer, and each amount column is a **feature** describing that customer. A feature is an input item the model can use to make its prediction. `customer_id` primarily identifies the customer and connects them to a result. Being numeric does not make it a quantity to calculate with in the same way as an amount.[^ml]

### To apply common filtering, calculation, or plotting rules across categories, keep category as a value

To select only electronics, you can use `category == "electronics"` in the long table. To calculate total spending by category, group by `category` and sum `total_amount`.

When a new category appears, it becomes another value in the same `category` column, so you can process categories using the same rules. This format can also be useful when passing category to a plotting function as a color or grouping field.

Of course, you can sum each category column in the wide table too. **The choice is not about which format permits calculation. It is about what you want to treat as the common field in the next operation.**

### If the comparison changes, the basis for each row can change too

If you want to compare the two customers within each category, you could arrange the table like this.

| category — index | 101 | 102 |
| --- | ---: | ---: |
| food | 15000 | 7000 |
| electronics | 30000 | 0 |
| clothing | 5000 | 20000 |

Now category defines the rows, and customer IDs are the column names. **There is no rule that customer_id must always be placed in rows.** Customer 101's food total is still 15,000; the direction of the comparison has changed.

For these two amount tables, a **transpose**, which exchanges the row and column axes, can also perform this change. But a general pivot that groups and spreads values from original records is not simply a transpose.[^reshape]

### If time is part of the distinction, preserve it throughout the transformation

For monthly spending analysis, you could define one row as **one customer in one month**, rather than one customer. The row keys for aggregation and pivoting would then be `customer_id + month`. Both columns would also need to remain in `id_vars` when melting.

If several months for the same customer have already been combined, you cannot recover the monthly changes from that result. **The analysis period is part of what a number means, not a description you can attach afterward.**

## 7. After reshaping, pay close attention to the unit of each calculation

### Even for the same customer, different denominators answer different questions

Compare two calculations using customer 101's records.

| Calculation | Actual calculation | Meaning |
| --- | --- | --- |
| Mean of the four original records | (10000 + 5000 + 30000 + 5000) / 4 = 12500 | Mean amount per record |
| Mean of the three amount columns after pivoting | (15000 + 30000 + 5000) / 3 ≈ 16666.67 | Mean of the category spending totals |

The pivot did not arbitrarily change the mean. **The calculation changed from averaging four records to averaging three category totals.**

The mean food total across the two customers is `(15000 + 7000) / 2 = 11000`. But the mean of the three original food records is `(10000 + 5000 + 7000) / 3 ≈ 7333.33`. The first gives each customer equal weight; the second gives each record equal weight.

Before calling `.mean()`, put the question into words: **“An average per what?”**

### Not every numeric column can meaningfully be added to the others

If the categories do not overlap and together cover all spending in the period, `food + electronics + clothing` gives the customer's total spending.

By contrast, `total_amount + mean_record_amount + max_record_amount` is not total spending. It adds metrics that summarize the same records in different ways. **Category amount columns and calculated metric columns may look similar, but the reasons for adding them are different.**

### Distinguish repeated identifiers from repeated measurements

It is normal for customer 101 to appear three times in the melted result. But if you also attach the customer's overall spending of KRW 50,000 to every category row and then sum that column, you get KRW 150,000. **Treating a repeated customer-level value as a category-level value causes double counting.**

For the same reason, you should not count the rows after melting and call that the number of customers. Nor should you immediately treat several rows from one customer as independent new samples. When evaluating a model's performance on previously unseen customers, consider splitting by customer so that rows from the same customer do not appear in both the training and evaluation sets.[^groups]

### Zeros and missing values give an average different meanings

In this table, the electronics average across both customers is `(30000 + 0) / 2 = 15000`. If you exclude the customer with no electronics record and consider only customers with such records, it is KRW 30,000. **Interpreting an average also requires deciding which customers belong in the analysis.**

Likewise, `melt()` can place columns with different units into one value column, but that does not make it meaningful to average heights in centimeters together with weights in kilograms. Check that the units and measurements are comparable. Keep different measurements distinguishable by their type and unit, and calculate them separately.

## 8. Read the difference between agg() and pivot_table() through what their columns mean

| Aspect | agg() producing several metrics per customer | pivot_table() producing spending by category |
| --- | --- | --- |
| Subject grouped into a row | One customer | One customer |
| What the columns distinguish | Total, mean, and maximum | Food, electronics, and clothing |
| Where column names come from | The chosen calculations or metric names you specify | Values in the original category column |
| Meaning of one cell | The result of that calculation for the customer | The customer's total for that category |
| Category distinctions | Not retained in this example | Preserved in the column names |

The core purpose of `agg()` is **to specify what to summarize and how**. `pivot_table()` **calculates and arranges summary values according to the chosen row and column categories**. Completely separating them into “a calculation function” and “a reshaping function” misses the aggregation that `pivot_table()` also performs.

This table compares the specific uses shown above. `agg()` can also group by both customer and category, and you can choose the names of the resulting metrics. With ordinary `DataFrame.agg()`, calculation names may appear in rows instead. Conversely, `pivot_table(aggfunc=["sum", "mean"])` creates multiple levels of column labels that distinguish both calculation type and category.[^groupby][^df-agg][^pivot-table]

**The function name does not determine what the columns mean. The grouping keys, the values you calculate from, and the calculation you apply determine that.**

## Complete example code

The code below uses the same original data for every table in the article. It follows two separate paths: **original records → customer metrics**, and **original records → customer–category totals → wide table → long table**.

```python
import pandas as pd  # Use Pandas, which works with tabular data, under the name pd.

# Every record belongs to the same analysis period. One row = one spending record.
orders = pd.DataFrame({
    "record_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
    "customer_id": [101, 101, 101, 101, 102, 102],
    "category": ["food", "food", "electronics", "clothing", "food", "clothing"],
    "amount": [10000, 5000, 30000, 5000, 7000, 20000],
})
category_order = ["food", "electronics", "clothing"]  # Display order of the result columns.

# 1. Customer metrics: summarize amounts by customer without retaining category distinctions.
# new_column_name=(source_column, calculation) specifies the result name and calculation.
stats = orders.groupby("customer_id").agg(
    total_amount=("amount", "sum"),         # Sum of all records for the customer.
    mean_record_amount=("amount", "mean"),  # Mean amount per record for the customer.
    max_record_amount=("amount", "max"),    # Largest recorded amount for the customer.
)

# 2. Aggregate while keeping both customer and category. One row = customer × category.
# as_index=False: keep the grouping keys as ordinary columns.
# sort=False: keep groups in the order they first appear in the original records.
pair_totals = orders.groupby(
    ["customer_id", "category"], as_index=False, sort=False
).agg(total_amount=("amount", "sum"))

# 3. Move category values in the aggregated table into column names. No further aggregation.
# pivot() works because each customer–category combination already has exactly one value.
wide_from_pairs = pair_totals.pivot(
    index="customer_id",    # One row in the final table = one customer.
    columns="category",     # Use the values in category as column names.
    values="total_amount",  # Place the total already calculated for each combination.
).reindex(columns=category_order)  # Rearrange the columns into the specified order.
# In this example only, fill missing combinations with 0 on the assumption of no purchase.
wide_from_pairs = wide_from_pairs.fillna(0)

# 4. Aggregate and pivot the original records in one operation to create the same amount table.
wide = orders.pivot_table(
    index="customer_id",
    columns="category",
    values="amount",
    aggfunc="sum",  # Sum the records for each customer–category pair. The default is mean.
    fill_value=0,   # Assumes no purchase, rather than a gap in data collection.
).reindex(columns=category_order)

# 5. Column names → values in category; amount cells → values in total_amount.
# reset_index(): move customer IDs from the index into an ordinary column.
long = wide.reset_index().melt(
    id_vars="customer_id",       # Repeat the customer identifier to keep it with every amount.
    value_vars=category_order,   # Category amount columns to unpack into rows.
    var_name="category",         # New column that holds the original column names.
    value_name="total_amount",   # New column that holds the cell values. They are still totals.
)

# 6. Use the same information for the analytical question at hand.
# axis=1: sum the category amounts within each customer's row.
# Assumes the categories do not overlap and cover all spending in the analysis period.
food_share = wide["food"] / wide.sum(axis=1)  # Food's share of each customer's spending.
category_totals = long.groupby("category")["total_amount"].sum()  # Total by category.
by_category = wide.T  # T exchanges the two axes of this table. Rows now represent categories.

# The same mean operation produces different metrics when the values being averaged differ.
record_mean = stats.loc[101, "mean_record_amount"]  # loc selects by row and column labels.
category_mean = wide.loc[101].mean()  # Mean of the three category totals for customer 101.

print("Customer metrics\n", stats)  # \n starts a new line in the output.
print("Customer-category totals\n", pair_totals)
print("Table pivoted after aggregation\n", wide_from_pairs)
print("Original records processed with pivot_table\n", wide)
print("Columns unpacked back into rows\n", long)
print("Table with categories as rows\n", by_category)
print("Food spending share by customer\n", food_share)  # 101: 0.3; 102: about 0.2593.
print("Totals by category\n", category_totals)  # food 22000, electronics 30000, clothing 25000.
print("Mean per record for customer 101:", record_mean)  # 12500.0.
print("Mean of category totals for customer 101:", category_mean)  # About 16666.67.
```

The wide tables produced by the two methods contain the same amounts. Their stored data types, such as integers or floating-point numbers, and their displayed formatting may differ depending on how missing cells are handled and which Pandas version you use.

## A final check

When reshaping a table, being able to complete the following sentence matters more than **whether the number of rows has increased or decreased**.

**“One row in this table represents ______, and the values in this column are ______ calculated per ______ over the period ______.”**

Then check where the distinctions remain. Is category stored as a value in an ordinary column, as a column name, or has it been combined with other categories during aggregation?

**Aggregation determines which details you summarize. Pivoting and melting change where you read the distinctions and measurements that remain. Any calculation that follows must respect what the reshaped table now means.**

[^groupby]: Official Pandas documentation — [Grouping, aggregation, and named aggregation for choosing result names](https://pandas.pydata.org/docs/user_guide/groupby.html).
[^reshape]: Official Pandas documentation — [Long and wide tables, pivoting, and reshaping](https://pandas.pydata.org/docs/user_guide/reshaping.html).
[^pivot]: Official Pandas documentation — [pivot(): reshaping without aggregation and the restriction on duplicate combinations](https://pandas.pydata.org/docs/reference/api/pandas.pivot.html).
[^pivot-table]: Official Pandas documentation — [pivot_table(): row and column keys, aggfunc, and filling missing values](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html).
[^melt]: Official Pandas documentation — [melt(): unpacking column names and values into rows while preserving identifiers](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.melt.html).
[^layout]: Official Pandas introductory guide — [Why and how to change a table's layout](https://pandas.pydata.org/docs/getting_started/intro_tutorials/07_reshape_table_layout.html).
[^df-agg]: Official Pandas documentation — [DataFrame.agg(): how the result structure depends on the call](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html).
[^reset]: Official Pandas documentation — [reset_index(): moving between the index and ordinary columns](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html).
[^ml]: Official scikit-learn documentation — [The number-of-samples × number-of-features structure of model input X](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.cross_validate.html).
[^groups]: Official scikit-learn documentation — [GroupKFold: preventing a group from appearing in both the training and evaluation sets](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GroupKFold.html).
