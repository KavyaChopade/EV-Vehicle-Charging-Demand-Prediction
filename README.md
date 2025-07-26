
# EV Vehicle Charge Demand Prediction - Week 2 Project

## 📌 Project Overview

This project is part of the AICTE Internship Program and continues from Week 1, focusing on **Electric Vehicle (EV) Charge Demand Prediction**. The goal is to build a predictive model using machine learning techniques to forecast the future charging demand of electric vehicles based on historical data.

## 🧠 Objective

- Perform **data cleaning**, **feature engineering**, and **model development**.
- Train and evaluate multiple **regression models**.
- Select the best-performing model for predicting EV charge demand.
- Interpret results and visualize key findings.

## 📂 Dataset Description

The dataset used in this project contains details related to electric vehicle populations and is assumed to have the following features (example):
- `Make`
- `Model`
- `Electric Vehicle Type`
- `Clean Alternative Fuel Vehicle (CAFV) Eligibility`
- `Electric Range`
- `Base MSRP`
- `Legislative District`
- `County`
- `City`
- `Model Year`

> 🔹 Actual column names and data may vary depending on Week 2 dataset structure.

## 🛠️ Technologies and Tools

- **Python**
- **Jupyter Notebook**
- **Pandas** – data manipulation
- **NumPy** – numerical operations
- **Matplotlib & Seaborn** – data visualization
- **Scikit-learn** – machine learning algorithms

## 🔍 Tasks Completed

### ✅ Data Preprocessing
- Handled missing values
- Removed irrelevant features
- Performed data type conversions
- Created new meaningful features

### 📊 Exploratory Data Analysis (EDA)
- Visualized distributions of numeric and categorical variables
- Examined correlations between features and target
- Analyzed trends and seasonal patterns

### 🤖 Model Building
- Trained and evaluated various regression models including:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Used **train-test split** and **cross-validation** for model validation

### 📈 Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

## 🚀 Results and Insights

- The model with the **lowest RMSE and highest R² score** was selected for prediction.
- Key features affecting EV charge demand were identified.
- Future deployment considerations discussed.

## 📌 Future Work

- Hyperparameter tuning using **GridSearchCV** or **RandomizedSearchCV**
- Incorporate time-series forecasting if temporal data is available
- Build an interactive dashboard using **Streamlit** or **Power BI**

## 📁 Repository Structure

```
📦EV_Vehicle_Charge_Demand
 ┣ 📜EV_Vehicle_Demand_Prediction_Week_2.ipynb
 ┣ 📜README.md
 ┗ 📂data/
     ┗ 📜EV_Vehicle_Population.csv
```

## 🧑‍💻 Author

**Kavya Chopade**  
AICTE Internship Program  
Week 2 Submission

---
