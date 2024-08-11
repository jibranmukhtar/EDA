# Exploratory Data Evaluation Package

## Overview
This library was primarily created to perform exploratory data analysis (EDA) on columns based on segments created by a dimension-describing column. 

For example, if we have mileage data for various cars under different automobile manufacturers, we can use this package to evaluate if there are any statistically significant differences between the average mileage of each automobile manufacturer. Additionally, it allows us to check if the nulls in the mileage column for a given car manufacturer are statistically different from those of another manufacturer.

## Features
- **Segmentation Analysis:** Analyze data based on segments defined by a categorical column.
- **Statistical Comparisons:** Perform statistical tests to compare means, variances, and other metrics between segments.
- **Null Analysis:** Evaluate and compare the presence of null values across segments.
- **Visualization Tools:** Generate visual representations of the analysis for better insights.
