# HR Analytics and Attrition Prediction Project
# This project shows how data, AI, and automation can be used for HR digitization

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the employee dataset
print("Loading employee data...")
df = pd.read_csv("employee_data.csv")
print(df.head())

# Prepare the data for analysis and machine learning
print("\nPreprocessing data...")

le = LabelEncoder()
df['Department'] = le.fit_transform(df['Department'])
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a machine learning model to predict attrition
print("\nTraining AI model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict attrition risk for a new employee
print("\nPredicting attrition risk for a new employee...")

new_employee = pd.DataFrame([{
    "Age": 27,
    "Salary": 40000,
    "YearsAtCompany": 2,
    "WorkLifeBalance": 2,
    "Department": le.transform(["IT"])[0]
}])

prediction = model.predict(new_employee)

if prediction[0] == 1:
    print("High attrition risk detected")
else:
    print("Low attrition risk detected")

# Generate a simple department-wise attrition insight
print("\nDepartment-wise Attrition:")
dept_attrition = df.groupby("Department")["Attrition"].mean()
print(dept_attrition)
