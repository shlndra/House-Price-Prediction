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
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("sample_employee_data.csv")

# Display the first few rows
print("🔹 First 5 rows:")
print(df.head())

# Dataset info
print("\n🔹 Info:")
print(df.info())

# Check for missing values
print("\n🔹 Missing Values:")
print(df.isnull().sum())

# Summary statistics
print("\n🔹 Summary Statistics:")
print(df.describe())

# Value counts for categorical column
print("\n🔹 Department Counts:")
print(df["Department"].value_counts())

# Mean Salary by Department
print("\n🔹 Mean Salary by Department:")
print(df.groupby("Department")["Salary"].mean())

# Data Visualizations
plt.figure(figsize=(12, 5))

# 1. Histogram of Age
plt.subplot(1, 2, 1)
sns.histplot(df["Age"], kde=True, color="skyblue")
plt.title("Age Distribution")

# 2. Boxplot of Salary by Department
plt.subplot(1, 2, 2)
sns.boxplot(x="Department", y="Salary", data=df, palette="pastel")
plt.title("Salary by Department")

plt.tight_layout()
plt.show()
