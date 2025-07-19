
# 🔋 EV Vehicle Charge Demand Prediction - Week 1 Project

## 📌 Project Title
**EV Vehicle Charge Demand Prediction - Week 1 Project**

## 👩‍💻 Author
**Kavya Chopade**

## 📖 Project Overview
This project is the Week 1 submission for the **AICTE Internship**, focusing on the initial steps of an Electric Vehicle (EV) vehicle charge demand prediction. The primary goal of this week's project is to perform **Exploratory Data Analysis (EDA)** and **data preprocessing** on the provided `Electric_Vehicle_Population_By_County.csv` dataset. This involves understanding the dataset's structure, identifying and handling missing values, and detecting and mitigating outliers to prepare the data for future predictive modeling.

## 📂 Dataset
The dataset used in this project is `Electric_Vehicle_Population_By_County.csv`.  
It contains information related to electric vehicle population across various counties and states, including:

- **Date**: The date of the data record.
- **County**: The county where the vehicles are registered.
- **State**: The state where the county is located.
- **Vehicle Primary Use**: The primary use of the vehicle (e.g., Passenger, Truck).
- **Battery Electric Vehicles (BEVs)**: Number of Battery Electric Vehicles.
- **Plug-In Hybrid Electric Vehicles (PHEVs)**: Number of Plug-In Hybrid Electric Vehicles.
- **Electric Vehicle (EV) Total**: Total number of Electric Vehicles (BEVs + PHEVs).
- **Non-Electric Vehicle Total**: Total number of Non-Electric Vehicles.
- **Total Vehicles**: Total number of all vehicles (EV Total + Non-Electric Vehicle Total).
- **Percent Electric Vehicles**: The percentage of electric vehicles out of total vehicles.

## 🧾 Project Structure
The project is structured as a Jupyter Notebook, `EV_Vehicle_Demand_Prediction_Week_1.ipynb`, which includes the following key steps:

### 1. Importing Libraries
Essential Python libraries such as `pandas`, `numpy`, `matplotlib.pyplot`, and `seaborn` are imported for data manipulation, numerical operations, and visualization.

### 2. Data Loading
The dataset is loaded into a pandas DataFrame.

### 3. Initial Data Inspection
- Displaying the first few rows (`df.head()`)
- Checking the dimensions of the dataset (`df.shape`)
- Summary of data types and null values (`df.info()`)
- Checking for null values in each column (`df.isnull().sum()`)

### 4. Data Preprocessing
- **Date Column Conversion**: Converted to datetime using `pd.to_datetime()` and invalid rows removed.
- **Handling Missing Values**: Missing values in `County` and `State` filled with 'NA'.

### 5. Outlier Detection and Handling
- **Statistical Summary**: `df.describe()` used to detect high variance.
- **Boxplot & Distribution Plot**: Visual inspection of outliers in `Percent Electric Vehicles`.
- **IQR Method**:
  - Calculate Q1 and Q3
  - Compute IQR
  - Cap values beyond `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`
- **Post-treatment Visualization**: Updated boxplot shows normalized range.

## 🧰 Technologies Used
- Python 3
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn

## 🛠️ How to Run the Project

### Clone the Repository
```bash
git clone https://github.com/KavyaChopade/EV-Vehicle-Charging-Demand-Prediction.git
cd EV-Vehicle-Demand-Prediction
```

### Install Required Libraries
```bash
pip install jupyter pandas numpy matplotlib seaborn
```

### Add Dataset
Make sure the `Electric_Vehicle_Population_By_County.csv` file is in the same directory as the notebook.

### Launch Jupyter Notebook
```bash
jupyter notebook
```

### Open and Run Notebook
Open `EV_Vehicle_Demand_Prediction_Week_1.ipynb` and run each cell in order.

## 🔮 Future Work (Week 2 and Beyond)
This project lays the groundwork for further analysis. In subsequent weeks, the focus will likely shift toward:

- Feature Engineering (e.g., extracting year/month from dates)
- Further Data Cleaning
- Time Series Analysis
- Predictive Modeling for EV charge demand
- Model Evaluation

---
