### Overview
Customer churn is a critical metric for telecom companies. This project explores a dataset of 7,043 customers to identify patterns that lead to churn and uses statistical modeling to predict future behavior.

### Key Features
Data Cleaning & Preprocessing: Handled missing values in TotalCharges by calculating them based on MonthlyCharges and tenure.

Exploratory Data Analysis (EDA): * Visualized the distribution of internet services and customer tenure.

Analyzed churn levels across different demographics and payment methods.

Predictive Modeling: Built a Linear Regression model using tenure as the independent variable to predict MonthlyCharges, achieving an RMSE of approximately 26.96.

### Tech Stack
Language: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("/content/customer_churn (1) (1) (2).csv")

# Check dimensions and column names
print(df.shape)
print(df.columns)

# Preview random samples
df.sample(10)

# Check data types and missing values
df.info()

# Convert TotalCharges to numeric, handling errors as NaNs
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')

# Check for null values
df.isnull().sum()

# Print rows with null values (usually those with tenure = 0)
print(df[df.isnull().any(axis = 1)])

# Fill missing TotalCharges by multiplying MonthlyCharges and tenure
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])

# Verify cleaning
print(df.isnull().sum())
print("Duplicates count:", df.duplicated().sum())

# Extract specific columns
column_5 = df.iloc[:, 4]
column_15 = df.iloc[:, 14]

# Senior male customers using Electronic check
senior_male_electronic = df[(df['gender']=='Male') & (df['SeniorCitizen']==1) & (df['PaymentMethod']=='Electronic check')]

# Customers with tenure > 70 months OR monthly charges > $100
customer_total_tenure = df[(df['tenure'] > 70) | (df['MonthlyCharges'] > 100)]

# Two-year contract, mailed check, and Churn = 'Yes'
two_mail_yes = df[(df['Contract'] == 'Two year') & (df['PaymentMethod'] == 'Mailed check') & (df['Churn'] == 'Yes')]

# Extract 333 random records
customer_333 = df.sample(n=333)

# Count churn levels
df['Churn'].value_counts()

# Bar plot for Internet Service Distribution
internet_counts = df["InternetService"].value_counts()
plt.figure(figsize=(6,4))
plt.bar(internet_counts.index, internet_counts.values, color="orange")
plt.xlabel("Categories of Internet Service")
plt.ylabel("Count of Categories")
plt.title("Distribution of Internet Service")
plt.show()

# Histogram for Tenure
plt.hist(df['tenure'], color='green', bins=30)
plt.title('Distribution of tenure')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define variables
x = df[['tenure']]
y = df['MonthlyCharges']

# Split data (70:30 ratio as requested)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Build and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)
error = y_pred - y_test
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"RMSE: {rmse}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")

