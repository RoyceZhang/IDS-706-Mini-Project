[![Python Tests](https://github.com/RoyceZhang/week2-project-and-rust-assignment/actions/workflows/test.yml/badge.svg)](https://github.com/RoyceZhang/week2-project-and-rust-assignment/actions/workflows/test.yml)

# Week 2 Project: Medical Insurance Cost Analysis

## Dataset

This project uses the Medical Cost Personal Dataset from Kaggle:

https://www.kaggle.com/datasets/mirichoi0218/insurance

The dataset contains information such as age, sex, BMI, number of children, smoking status, region, and medical insurance charges.

The main target variable for the machine learning section is `charges`.

## Setup and Running Instructions

You can see all required Python packages in requirements.txt.

## Importing and Inspecting the Data

I used Pandas to import `insurance.csv`.

I inspected the dataset using:

- `head()` to view the first few rows
- `info()` to check column names and data types
- `describe()` to view summary statistics
- `isnull().sum()` to check missing values
- `duplicated().sum()` to check duplicate rows

## Filtering

I created two filtered subsets:

1. People older than 50
2. People who are smokers

These filters help examine specific groups in the dataset.

## Grouping

I grouped the data by smoking status and calculated:

- Average insurance charges
- Number of observations in each smoking group

I also grouped by region and calculated the average insurance charge for each region.

I compared Pandas and Polars by running the same grouping operation 1000 times.

In this small dataset Pandas is faster than Polars. However, Polars is often faster on larger datasets because it is designed for efficient column-based data processing.

## Visualization

I created a scatter plot of BMI versus insurance charges.

The plot is useful for visually exploring whether BMI and insurance cost appear to be related.

## Machine Learning

I chose **Linear Regression** because the target variable, `charges`, is continuous.

For the first model, I used three numerical input variables:

- `age`
- `bmi`
- `children`

The model predicts medical insurance charges.

I used a train-test split so that the model is trained on one part of the data and evaluated on another part.

For basic evaluation, I calculated:

- Mean Squared Error (MSE)
- R-squared

### Compare Linear Regression Models by Smoking Status

Since smokers and non-smokers have very different average insurance charges,
I build separate Linear Regression models for the two groups and compare
their performance.

## Testing

This project uses `pytest` to test the main parts of the data analysis workflow.

The tests include:

- Loading the insurance dataset
- Removing duplicate rows during preprocessing
- Filtering smokers and computing grouped summary statistics
- Running the complete Linear Regression workflow

To run the tests locally:

```bash
python -m pytest

## Conclusion

The dataset is appropriate for both exploratory data analysis and regression. Filtering and grouping make it easy to compare different groups, especially smokers and non-smokers. The scatter plot provides a simple way to inspect the relationship between BMI and charges. The Linear Regression model serves as a baseline model. In future parts of the project, I could improve it by adding categorical variables such as smoking status, sex, and region.
