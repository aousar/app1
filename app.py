#========================================
# INSURANCE PREDICTION USING MACHINE LEARNING
# LINEAR REGRESSION AND POLYNOMIAL REGRESSION
# ==========================================

# Import libraries
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("insurance.csv")

print("First 5 rows:")
print(data.head())


# ==========================================
# 2. EXPLORATORY DATA ANALYSIS
# ==========================================

print("\nSex Value Counts:")
print(data["sex"].value_counts())

print("\nChildren Value Counts:")
print(data["children"].value_counts())


# Age vs Charges
sns.regplot(x="age", y="charges", data=data)
plt.title("Age vs Charges")
plt.show()


# Sex vs Charges
sns.barplot(x="sex", y="charges", data=data)
plt.title("Sex vs Charges")
plt.show()


# BMI vs Charges
sns.regplot(x="bmi", y="charges", data=data)
plt.title("BMI vs Charges")
plt.show()


# Smoker vs Charges
sns.barplot(x="smoker", y="charges", data=data)
plt.xlabel("Smoker")
plt.ylabel("Charges")
plt.title("Smoker vs Charges")
plt.show()


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

# Drop unnecessary columns
data = data.drop(["children", "region"], axis=1)


# Convert sex into numerical values
gender_map = {
    "female": 1,
    "male": 0
}

data["sex"] = data["sex"].map(gender_map)


# Convert smoker into numerical values
smoker_map = {
    "yes": 1,
    "no": 0
}

data["smoker"] = data["smoker"].map(smoker_map)


print("\nStatistical Description:")
print(data.describe())


# ==========================================
# 4. CORRELATION
# ==========================================

print("\nCorrelation Matrix:")
print(data.corr())


sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# Drop sex column
data = data.drop(["sex"], axis=1)


# ==========================================
# 5. DEFINE FEATURES AND TARGET
# ==========================================

x = data.drop(["charges"], axis=1)
y = data["charges"]


# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 7. LINEAR REGRESSION
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Create model
lr = LinearRegression()


# Train model
lr.fit(x_train, y_train)


# Prediction
y_pred_linear = lr.predict(x_test)


print("\nLinear Regression Predicted Values:")
print(y_pred_linear)


# R-squared score
r2_linear = r2_score(y_test, y_pred_linear)

print("\nLinear Regression R-squared:", r2_linear)


# ==========================================
# 8. POLYNOMIAL REGRESSION
# ==========================================

from sklearn.preprocessing import PolynomialFeatures


# Create polynomial features
pf = PolynomialFeatures(degree=3)


# Convert training and testing data
x_train_poly = pf.fit_transform(x_train)
x_test_poly = pf.transform(x_test)


# Create new Linear Regression model
poly_model = LinearRegression()


# Train polynomial model
poly_model.fit(x_train_poly, y_train)


# Prediction
y_pred_poly = poly_model.predict(x_test_poly)


print("\nPolynomial Regression Predicted Values:")
print(y_pred_poly)


# R-squared score
r2_poly = r2_score(y_test, y_pred_poly)

print("\nPolynomial Regression R-squared:", r2_poly)


# ==========================================
# 9. STREAMLIT WEB APPLICATION
# ==========================================

import streamlit as st

st.set_page_config(
    page_title="Insurance Prediction",
    page_icon="💰"
)

st.title("💰 Insurance Cost Prediction")

st.write(
    "Predict medical insurance charges using "
    "Linear Regression and Polynomial Regression."
)

st.divider()


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Linear Regression R²",
        f"{r2_linear:.4f}"
    )

with col2:
    st.metric(
        "Polynomial Regression R²",
        f"{r2_poly:.4f}"
    )


st.divider()


# ==========================================
# USER INPUT
# ==========================================

st.subheader("🔮 Enter Patient Details")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)

smoker = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)


# Convert smoker to same format
# used during training

smoker_value = 1 if smoker == "Yes" else 0


# ==========================================
# ALGORITHM SELECTION
# ==========================================

algorithm = st.selectbox(
    "Select Algorithm",
    [
        "Linear Regression",
        "Polynomial Regression"
    ]
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("🚀 Predict Insurance Charges"):

    # Create input DataFrame
    new_patient = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "smoker": [smoker_value]
    })

    # -----------------------------
    # Linear Regression
    # -----------------------------

    if algorithm == "Linear Regression":

        prediction = lr.predict(
            new_patient
        )[0]


    # -----------------------------
    # Polynomial Regression
    # -----------------------------

    else:

        new_patient_poly = pf.transform(
            new_patient
        )

        prediction = poly_model.predict(
            new_patient_poly
        )[0]


    # ==================================
    # DISPLAY PREDICTION
    # ==================================

    st.success(
        f"Predicted Insurance Charges: "
        f"${prediction:,.2f}"
    )

    st.info(
        f"Algorithm Used: {algorithm}"
    )

    st.write("### Patient Information")

    st.write(
        f"**Age:** {age}"
    )

    st.write(
        f"**BMI:** {bmi}"
    )

    st.write(
        f"**Smoker:** {smoker}"
    )

    