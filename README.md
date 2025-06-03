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

# Load sample dataset
df = sns.load_dataset("iris")

# Display dataset
print(df.head())

# Perform similar analysis as before
print("Sum:", df["sepal_length"].sum())
print("Mean:", df["sepal_length"].mean())
print("Mode:", df["sepal_length"].mode().values)

# Visualization
plt.hist(df["sepal_length"], bins=10, color='lightgreen')
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
