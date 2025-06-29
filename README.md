# AI/ML Engineering Internship Tasks

This repository contains solutions for three tasks from the DevelopersHub Corporation AI/ML Engineering Internship, submitted by [Nida Malik].

## Overview
This project includes three tasks focused on data exploration, classification, and regression using Python, pandas, seaborn, matplotlib, and scikit-learn. Each task is organized in a separate folder with scripts, datasets (where applicable), and visualization outputs.

## Task 1: Exploring and Visualizing a Simple Dataset
- **Objective**: Explore and visualize the Iris Dataset to understand trends and distributions.
- **Dataset**: Iris Dataset (loaded via `seaborn.load_dataset('iris')` or `iris.csv` from UCI).
- **Files**:
  - `Task1/iris_exploration.py`: Python script for data loading, inspection, and visualization.
  - Plots: `sepal_scatter_plot.png`, `iris_histograms.png`, `petal_width_boxplot.png`.
  - Optional: `iris.csv` (if downloaded from UCI).
- **Key Results**: Visualizations show Setosa has smaller petal measurements; few outliers in Virginica’s petal width.
- **Findings**: Clear species separation in petal features, with Setosa being most distinct.

## Task 3: Heart Disease Prediction
- **Objective**: Predict heart disease risk using the Heart Disease UCI Dataset.
- **Dataset**: `heart.csv` from Kaggle (https://www.kaggle.com/datasets/ronitf/heart-disease-uci).
- **Files**:
  - `Task3/heart_disease_prediction.py`: Python script for preprocessing, EDA, modeling (Logistic Regression), and evaluation.
  - `Task3/heart.csv`: Dataset.
  - Plots: `correlation_heatmap.png`, `age_distribution.png`, `confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`.
- **Key Results**: Accuracy ~0.80–0.90, AUC ~0.85–0.95, key features: chest pain type (`cp`), maximum heart rate (`thalach`).
- **Findings**: Strong model performance with high AUC; key predictors identified.

## Task 6: House Price Prediction
- **Objective**: Predict house prices using property features.
- **Dataset**: `housing.csv` from Kaggle (e.g., Boston Housing: https://www.kaggle.com/datasets/cdeotte/housing) or via `sklearn.datasets`.
- **Files**:
  - `Task6/house_price_prediction.py`: Python script for preprocessing, EDA, modeling (Linear Regression), and evaluation.
  - `Task6/housing.csv`: Dataset.
  - Plots: `correlation_heatmap.png`, `rooms_vs_price.png`, `predicted_vs_actual.png`, `feature_importance.png`.
- **Key Results**: MAE ~3–5, RMSE ~4–7 (Boston Housing), key features: number of rooms (`rm`), socioeconomic status (`lstat`).
- **Findings**: Reasonable model performance; key predictors identified.

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install pandas seaborn matplotlib scikit-learn