# Exploratory Data Evaluation Package

## Overview
This library was primarily created to perform exploratory data analysis (EDA) on columns, based on segments created by a dimension-describing column. In the code these dimension-describing columns are referred to by "Comparison Groups".

For example, if we have mileage data for various cars under different automobile manufacturers, we can use this package to evaluate if there are any statistically significant differences between the mileage of each automobile manufacturer or compare whether the number of NULLs in one manufacturer is significantly greater than the other.

## Features
- **Segmentation Analysis:** Analyze data based on segments defined by a column or a combination of columns.
- **Statistical Comparisons:** Perform statistical tests to compare means, variances, and other metrics between segments.
- **Null Analysis:** Evaluate and compare the presence of null values across segments.
- **Report Choice:** Choice in evaluating based on a detailed report, a summary report and a flag report which flags inconsistencies based on pre-determined thresholds.
- **Additional Features:** It also offers standard EDA (Exploratory Data Analysis) features such as unique value count, null count, proportion of null values, column type evaluation, and outlier analysis.
