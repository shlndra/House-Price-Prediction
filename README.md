# 🏠 House Price Prediction | Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-yellowgreen)

A machine learning model to predict house prices based on features like square footage, bedrooms, and location. Built with Python and scikit-learn.

📂 **Dataset Source**: [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)


---

## 🏆 Project Overview
This project predicts house prices using regression techniques:
- **Data Preprocessing**: Handles missing values, scaling, and EDA.
- **Models Tested**: Linear Regression, Decision Tree.
- **Evaluation Metrics**: MSE, R² Score.
- **Goal**: Assist buyers/sellers in estimating property values.

---

## ✨ Features
✅ **Exploratory Data Analysis (EDA)**  
✅ **Feature Scaling & Preprocessing**  
✅ **Multiple ML Models Comparison**  
✅ **Model Persistence (Save/Load)**  
✅ **Feature Importance Analysis**  

---

## 🛠 Tech Stack
- **Language**: Python
- **Libraries**: 
  - `pandas`, `numpy` (Data Handling)
  - `matplotlib`, `seaborn` (Visualization)
  - `scikit-learn` (ML Models)
- **Environment**: Jupyter Notebook

---



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Import dataset using CSV file
df = pd.read_csv('/mnt/data/file-UHAt3FHm48GqvSGdeLBRyC')

# 2. Display first 5 rows of the dataset
print("Dataset Preview:")
print(df.head())

# 3. Calculating sum, mean, and mode of a particular column
column = df.columns[1]  # change index if needed
print(f"\nColumn Selected: {column}")
print(f"Sum: {df[column].sum()}")
print(f"Mean: {df[column].mean()}")
print(f"Mode: {df[column].mode().values}")

# 4. Basic Exploratory Data Analysis (EDA)
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

# 5. Data Visualization
plt.figure(figsize=(10, 5))
plt.hist(df[column], bins=20, color='skyblue', edgecolor='black')
plt.title(f'Distribution of {column}')
plt.xlabel(column)
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
