+++
title = "Weekly Paper 02: Averages and Table Merges"
description = "Notes on interpreting average purchase amounts and the steps for combining customer data from two teams."
date = 2026-10-03
updated = 2026-10-03
slug = "weekly-paper-02"
aliases = ["/en/test-gradient-descent/"]
+++

<style>
/* Apply only to this post's research topics and subtopics. Do not change headings in other posts. */
#main-content article h4.research-topic {
  margin: 2.75rem 0 .65rem;
  font-size: 1.4rem;
  font-weight: 750;
  line-height: 1.5;
  text-wrap: pretty;
  word-break: keep-all;
  overflow-wrap: anywhere;
}
/* Avoid an unnecessarily large gap directly below a Research Notes heading. */
#main-content article h3 + h4.research-topic {
  margin-top: 1.25rem;
}
#main-content article h5.research-detail {
  margin: 1.5rem 0 .6rem;
  font-size: 1.125rem;
  font-weight: 650;
  line-height: 1.55;
}
/* Separate a research topic from the preceding paragraph, while keeping its first subtopic close. */
#main-content article h4.research-topic + h5.research-detail {
  margin-top: 0;
}
</style>

## Question 1

Customer purchase amounts have a strongly right-skewed distribution. Your team lead says, "The average purchase amount is KRW 50,000, so we can assume everyone spends about KRW 50,000." How would you respond?

### 1) Research Notes

#### Google search: "strongly right-skewed distribution" {#research-skew .research-topic}

##### 1. Why Is It Called a Right-Skewed Distribution? {.research-detail}

- Direction of the tail: Most of the data is clustered on the left, forming a peak there, while some large values stretch the tail out to the right.
- Shift in the mean: This long right tail, made up of large values, pulls the mean to the right.

In other words, it is called a **right-skewed distribution**, or a distribution with **positive skewness**, because its tail extends a long way to the right.

#### Google search: "what to watch out for when using averages to describe data" {#research-mean .research-topic}

##### 1. How Outliers Can Give a Misleading Impression {.research-detail}

The mean is calculated by adding all the data values and dividing by the number of values, so it is strongly affected by extremely large or small values, called outliers.

- Example: At a company where nine people earn KRW 30 million a year and one person earns KRW 1 billion a year, the average annual salary is KRW 127 million. This does not reflect the situation of most employees, who earn KRW 30 million.
- What to check as well: When the data is strongly skewed, also look at the median, the value in the middle.

##### 2. Ignoring the Shape of the Distribution {.research-detail}

The mean does not show how the data is spread out, or the shape of its distribution. Even when the mean is the same, the actual data can look completely different.

- Example: Even if a stream's average depth is 1.2 m, some spots may be only 10 cm deep while others are 3 m deep. Entering the water based on the average alone could be dangerous.
- What to check as well: Also look at the standard deviation or variance, which describe how spread out the data is.

### 2) My Answer

I would probably say to my team lead, "The average is KRW 50,000, but some customers' large purchases may have pushed it up. I'll also check the median and the number of customers in each spending range."

So I would explain that the average and the amount most customers actually spend can be different, rather than saying the average is wrong. I would use the median to see the middle level and the customer counts in each spending range to see where most customers are concentrated, then show those figures together.

I would not simply remove customers with large purchases, either. I think I would first need to check whether those are data entry errors or customers really made purchases that large. If the purchases are real, I would keep the average that includes them and look at it alongside the other figures.

## Question 2

You need to combine customer data that two different teams have been managing separately. Both datasets contain duplicate customer IDs or rows with inconsistent values, as well as missing values. In what order would you approach this task?

### 1) Research Notes

#### Google search: "what to watch out for when merging two important tables" {#research-merge .research-topic}

##### Before the Merge: Analyze Data Structure and Quality {.research-detail}

- Back up the original data: Do not work directly on a production server or the main database. Always do the work in a temporary table or staging area first.
- Check that data types match: Make sure the key columns used for the join have matching data types. If one is INT and the other is VARCHAR, type conversion errors or failed matches may occur.
- Clean the key columns: Check whether the join columns contain whitespace, differences in capitalization, or special characters. For example, "Apple " and "Apple" may be treated as different values.
- Identify duplicate keys and the relationships between records: Determine beforehand whether the relationship is 1:1, 1:N, or N:N. In particular, joining an N:N relationship without checking it can greatly increase the number of rows, because rows sharing a key are combined with one another in a Cartesian product.
- Check indexes: When merging large tables, check the indexes on the join key columns. Indexes can help performance, but their effect depends on the data volume and how the data is processed.

##### After the Merge: Validate the Data Carefully {.research-detail}

- Compare row counts: Always compare the number of rows before and after the merge. If a LEFT JOIN produces more rows than the original table, the matching keys in the other table contain duplicates.
- Verify aggregates: When comparing the same set of transactions or customers, cross-check that key figures from the tables before the merge, such as total sales or total member count, match the figures in the merged result. This can catch errors where duplicated data inflates amounts.
- Check samples: Pick a few records from the top, bottom, and middle of the merged result and compare them visually with the original data.

### 2) My Answer

First, I would save copies of the original data and check what one row in each table represents. I think the meaning of a duplicate would depend on whether the table has one row per customer or multiple rows for each customer's purchase records. I would also check whether the two teams use the same customer ID for the same person.

Next, I would make the customer ID data types and formatting consistent, including any differences in whitespace. I would not immediately delete a row just because its ID appears more than once. I would first check whether the same record has been entered twice or whether the records are different.

If the values differ for the same customer, I would ask the person responsible which one to keep. Newer data is not necessarily correct, so I think we would first need to decide which source to use as the reference. For missing values, I would also check whether they could be filled using the same customer's values from the other team's data. If I could not verify them, I would leave them blank for the time being and set aside rows without IDs for a separate check.

After cleaning up the data this way, I would combine the tables and check whether the row count and customer count are what I expected. I would check whether the combined table contains unnecessary repeated entries for the same customer or is missing any information, and compare a few records directly with the originals. I think the total amounts would also need to be compared using the same transactions and time period.

I think the main thing to watch out for in this task is accidentally deleting useful information while rushing to remove duplicates or blanks. I would keep separate notes on what I changed and what still needs to be checked.
