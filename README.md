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



import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
df = sns.load_dataset("iris")

# -----------------------------
# 1. Dataset Overview
# -----------------------------
print("Dataset Info:")
print(df.info(), "\n")

print("Statistical Summary:")
print(df.describe(), "\n")

print("Missing Values:")
print(df.isnull().sum(), "\n")

# -----------------------------
# 2. Univariate Analysis
# -----------------------------
print("Basic Statistics for Sepal Length:")
print("Sum:", df["sepal_length"].sum())
print("Mean:", df["sepal_length"].mean())
print("Mode:", df["sepal_length"].mode().values)

# Histograms
df.hist(figsize=(10, 6), color='skyblue', edgecolor='black')
plt.suptitle("Histogram of Iris Features")
plt.tight_layout()
plt.show()

# Boxplots for outlier detection
sns.boxplot(data=df, orient="h", palette="Set2")
plt.title("Boxplot of All Numerical Features")
plt.show()

# -----------------------------
# 3. Bivariate Analysis
# -----------------------------
# Pairplot with hue
sns.pairplot(df, hue="species", palette="bright")
plt.suptitle("Pairplot of Iris Features by Species", y=1.02)
plt.show()

# Correlation heatmap (numeric features only)
numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# -----------------------------
# 4. Group Statistics by Species
# -----------------------------
print("Mean Values Grouped by Species:")
print(df.groupby("species").mean(), "\n")

print("Detailed Stats by Species:")
print(df.groupby("species").describe(), "\n")

plt.show()
