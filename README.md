# 🔋 EV Adoption Forecasting Web App

This project aims to forecast Electric Vehicle (EV) adoption across counties in Washington State using machine learning and present the forecasts through an interactive Streamlit web application.

---

## 📌 Problem Statement

With the increasing shift towards sustainable transportation, forecasting EV growth is essential for effective infrastructure planning, policy-making, and public awareness. This project provides a data-driven solution to predict monthly EV adoption in each county using historical EV registration data.

---

## 🚀 Project Overview

- **Goal**: Predict the number of electric vehicles over the next 3–5 years for each county.
- **Data Source**: Historical EV population by county in Washington (from `.xls` files and preprocessed CSV).
- **Model**: A Random Forest Regressor trained on engineered time-series features.
- **Interface**: A fully functional, dark-themed Streamlit web app allowing real-time forecasting and interactive visualization.

---

## 🛠️ Tools & Technologies Used

| Category              | Technologies                              |
|-----------------------|-------------------------------------------|
| Language              | Python                                     |
| Environment           | Jupyter Notebook, VSCode, Streamlit         |
| Data Processing       | Pandas, NumPy                              |
| Visualization         | Matplotlib, Streamlit                      |
| Machine Learning      | Scikit-learn (Random Forest)               |
| Model Persistence     | Joblib                                     |
| UI/UX Styling         | Custom HTML/CSS in Streamlit               |

---

## 🔄 Methodology Workflow

1. **Data Collection** from open government EV registration datasets  
2. **EDA and Cleaning** to explore patterns and prepare consistent data  
3. **Feature Engineering** including lags, rolling averages, percent change, and trend slope  
4. **Model Training** using Random Forest Regressor  
5. **Evaluation** using MAE and RMSE along with visual comparison  
6. **Model Serialization** using Joblib  
7. **Web App Development** for user interaction and visualization  
8. **Deployment Ready** for public usage and insights

---

## 🌟 Key Features of the Web App

- 📍 **County Selection**: Forecast EV growth in a specific county  
- 🔄 **Forecast Horizon**: Choose prediction window (12–60 months)  
- 📊 **Cumulative Trend Graphs**: Visualize EV growth over time  
- 📈 **Multi-County Comparison**: Compare up to 3 counties side by side  
- 📥 **Download Forecasts**: Export predictions as CSV  
- 🌙 **Dark Mode**: Modern UI using custom CSS and theming  
- 🧠 **Real-Time Forecasting**: Model predictions integrated into user interface  

---

## 📉 Evaluation

- Used **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** to evaluate prediction accuracy.
- Visualization of actual vs. predicted values helped validate model behavior.

---

## 🔮 Future Scope

- Replace Random Forest with advanced models like **XGBoost**, **LightGBM**, or **LSTM** for improved accuracy.  
- Integrate external variables such as **charging station density**, **fuel prices**, and **policy incentives** to enrich feature set and capture real-world influence on EV adoption.  
- Deploy the app on **Streamlit Cloud or Heroku** for public access.

---

## 👩‍💻 Developed By

**Kavya Chopade**   
---
