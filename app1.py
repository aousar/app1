
# INSURANCE COST PREDICTION
# Linear Regression + Polynomial Regression
# Streamlit + Seaborn EDA

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================
# 1. PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Insurance Cost Prediction")
st.write(
    "Predict medical insurance charges using "
    "Linear Regression and Polynomial Regression."
)

# ============================================
# 2. LOAD DATASET
# ============================================

@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    return df

df = load_data()

# ============================================
# 3. SEABORN EDA FEATURE
# ============================================

st.divider()
st.subheader("📊 Exploratory Data Analysis (EDA)")

st.write(
    "Explore the insurance dataset using "
    "interactive Seaborn visualizations."
)

eda_option = st.selectbox(
    "Select a Seaborn Graph",
    [
        "View Dataset",
        "Age Distribution",
        "BMI Distribution",
        "Insurance Charges Distribution",
        "Gender Count",
        "Smoker Count",
        "Region Count",
        "Age vs Charges",
        "BMI vs Charges",
        "Average Charges by Smoker",
        "Average Charges by Gender",
        "Average Charges by Region",
        "Correlation Heatmap"
    ]
)

if eda_option == "View Dataset":
    st.write("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    st.write("Dataset Shape:", df.shape)

    st.write("Missing Values")
    st.dataframe(df.isnull().sum().rename("Missing Values"))

    st.write("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

elif eda_option == "Age Distribution":
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=df,
        x="age",
        bins=15,
        kde=True,
        ax=ax
    )

    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
    plt.close(fig)

elif eda_option == "BMI Distribution":
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=df,
        x="bmi",
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_title("BMI Distribution")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
    plt.close(fig)

elif eda_option == "Insurance Charges Distribution":
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=df,
        x="charges",
        bins=30,
        kde=True,
        ax=ax
    )

    ax.set_title("Insurance Charges Distribution")
    ax.set_xlabel("Insurance Charges ($)")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
    plt.close(fig)

elif eda_option == "Gender Count":
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        data=df,
        x="sex",
        ax=ax
    )

    ax.set_title("Number of Customers by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")

    st.pyplot(fig)
    plt.close(fig)

elif eda_option == "Smoker Count":
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        data=df,
        x="smoker",
        ax=ax
    )

    ax.set_title("Smoker vs Non-Smoker Count")
    ax.set_xlabel("Smoker Status")
    ax.set_ylabel("Count")

    st.pyplot(fig)
    plt.close(fig)

elif eda_option == "Region Count":
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        data=df,
        x="region",
        ax=ax
    )

    ax.set_title("Customers by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Count")

    st.pyplot(fig)
    plt.close(fig)

elif eda_option == "Age vs Charges":
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.scatterplot(
        data=df,
        x="age",
        y="charges",
        hue="smoker",
        ax=ax
    )

    ax.set_title("Age vs Insurance Charges")
    ax.set_xlabel("Age")
    ax.set_ylabel("Charges ($)")

    st.pyplot(fig)
    plt.close(fig)

elif eda_option == "BMI vs Charges":
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.scatterplot(
        data=df,
        x="bmi",
        y="charges",
        hue="smoker",
        ax=ax
    )

    ax.set_title("BMI vs Insurance Charges")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Charges ($)")

    st.pyplot(fig)
    plt.close(fig)

elif eda_option == "Average Charges by Smoker":
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=df,
        x="smoker",
        y="charges",
        errorbar=None,
        ax=ax
    )

    ax.set_title("Average Insurance Charges by Smoker")
    ax.set_xlabel("Smoker Status")
    ax.set_ylabel("Average Charges ($)")

    st.pyplot(fig)
    plt.close(fig)

elif eda_option == "Average Charges by Gender":
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=df,
        x="sex",
        y="charges",
        errorbar=None,
        ax=ax
    )

    ax.set_title("Average Insurance Charges by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Average Charges ($)")

    st.pyplot(fig)
    plt.close(fig)

elif eda_option == "Average Charges by Region":
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=df,
        x="region",
        y="charges",
        errorbar=None,
        ax=ax
    )

    ax.set_title("Average Insurance Charges by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Average Charges ($)")

    st.pyplot(fig)
    plt.close(fig)

elif eda_option == "Correlation Heatmap":
    corr_df = df.copy()

    corr_df["sex"] = corr_df["sex"].map({
        "male": 0,
        "female": 1
    })

    corr_df["smoker"] = corr_df["smoker"].map({
        "no": 0,
        "yes": 1
    })

    corr_df = pd.get_dummies(
        corr_df,
        columns=["region"],
        drop_first=True,
        dtype=int
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.heatmap(
        corr_df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Feature Correlation Heatmap")

    st.pyplot(fig)
    plt.close(fig)

# ============================================
# 4. DATA PREPROCESSING
# ============================================

@st.cache_data
def preprocess_data(df):
    data = df.copy()

    # Convert sex
    data["sex"] = data["sex"].map({
        "male": 0,
        "female": 1
    })

    # Convert smoker
    data["smoker"] = data["smoker"].map({
        "no": 0,
        "yes": 1
    })

    # Convert region
    data = pd.get_dummies(
        data,
        columns=["region"],
        drop_first=True
    )

    # Convert boolean columns to integers
    for column in data.columns:
        if data[column].dtype == "bool":
            data[column] = data[column].astype(int)

    X = data.drop("charges", axis=1)
    y = data["charges"]

    return X, y

X, y = preprocess_data(df)

# ============================================
# 5. TRAIN MODELS ONLY ONCE
# ============================================

@st.cache_resource
def train_models(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # LINEAR REGRESSION

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    linear_prediction = linear_model.predict(
        X_test
    )

    linear_r2 = r2_score(
        y_test,
        linear_prediction
    )

    linear_mae = mean_absolute_error(
        y_test,
        linear_prediction
    )

    linear_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            linear_prediction
        )
    )

    # POLYNOMIAL REGRESSION

    poly = PolynomialFeatures(
        degree=2
    )

    X_train_poly = poly.fit_transform(
        X_train
    )

    X_test_poly = poly.transform(
        X_test
    )

    poly_model = LinearRegression()

    poly_model.fit(
        X_train_poly,
        y_train
    )

    poly_prediction = poly_model.predict(
        X_test_poly
    )

    poly_r2 = r2_score(
        y_test,
        poly_prediction
    )

    poly_mae = mean_absolute_error(
        y_test,
        poly_prediction
    )

    poly_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            poly_prediction
        )
    )

    return (
        linear_model,
        poly,
        poly_model,
        linear_r2,
        linear_mae,
        linear_rmse,
        poly_r2,
        poly_mae,
        poly_rmse
    )

# Train models once
(
    linear_model,
    poly,
    poly_model,
    linear_r2,
    linear_mae,
    linear_rmse,
    poly_r2,
    poly_mae,
    poly_rmse
) = train_models(X, y)

# ============================================
# 6. MODEL PERFORMANCE
# ============================================

st.divider()
st.subheader("📊 Model Performance")

performance = pd.DataFrame({
    "Algorithm": [
        "Linear Regression",
        "Polynomial Regression"
    ],
    "R² Score": [
        linear_r2,
        poly_r2
    ],
    "MAE": [
        linear_mae,
        poly_mae
    ],
    "RMSE": [
        linear_rmse,
        poly_rmse
    ]
})

st.dataframe(
    performance,
    use_container_width=True
)

# ============================================
# 7. USER INPUT
# ============================================

st.divider()
st.subheader("🔮 Predict Insurance Charges")

# Age
age = st.number_input(
    "Age",
    min_value=18,
    max_value=64,
    value=25,
    step=1
)

# Sex
sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

# BMI
bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)

# Children
children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

# Smoker
smoker = st.selectbox(
    "Smoker",
    ["no", "yes"]
)

# Region
region = st.selectbox(
    "Region",
    [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)

# Algorithm
algorithm = st.selectbox(
    "Select Algorithm",
    [
        "Linear Regression",
        "Polynomial Regression"
    ]
)

# ============================================
# 8. PREDICTION BUTTON
# ============================================

if st.button(
    "🚀 Predict Insurance Charges",
    use_container_width=True
):
    # Convert user input
    user_data = {
        "age": age,
        "sex": 0 if sex == "male" else 1,
        "bmi": bmi,
        "children": children,
        "smoker": 0 if smoker == "no" else 1
    }

    # Region encoding
    for column in X.columns:
        if column.startswith("region_"):
            region_name = column.replace(
                "region_",
                ""
            )

            user_data[column] = int(
                region == region_name
            )

    # Create input DataFrame
    user_input = pd.DataFrame(
        [user_data]
    )

    # Ensure same column order
    user_input = user_input.reindex(
        columns=X.columns,
        fill_value=0
    )

    # PREDICTION
    if algorithm == "Linear Regression":
        prediction = linear_model.predict(
            user_input
        )[0]
    else:
        user_input_poly = poly.transform(
            user_input
        )

        prediction = poly_model.predict(
            user_input_poly
        )[0]

    # DISPLAY RESULT
    st.success(
        f"💰 Predicted Insurance Charges: "
        f"${prediction:,.2f}"
    )

    st.info(
        f"Algorithm Used: {algorithm}"
    )

# ============================================
# 9. PROJECT INFORMATION
# ============================================

st.divider()
st.subheader("📌 Project Information")

st.write("""
**Project:** Medical Insurance Cost Prediction

**Algorithms Used:**
- Linear Regression
- Polynomial Regression

**Input Features:**
- Age
- Sex
- BMI
- Children
- Smoker
- Region

**Target Variable:**
- Insurance Charges

**Evaluation Metrics:**
- R² Score
- MAE
- RMSE

**EDA Libraries:**
- Seaborn
- Matplotlib
""")

