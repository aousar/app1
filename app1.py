
# ============================================
# INSURANCE COST PREDICTION
# Linear + Polynomial Regression
# Compact Streamlit Interface + Seaborn EDA
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter, MaxNLocator
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
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.3rem;
    padding-bottom: 1rem;
    max-width: 1000px;
}
h1 {
    font-size: 1.9rem !important;
}
h2, h3 {
    font-size: 1.25rem !important;
}
div[data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid", context="notebook")

st.title("💰 Insurance Cost Prediction")
st.caption(
    "Predict medical insurance charges using "
    "Linear Regression and Polynomial Regression."
)


# ============================================
# 2. LOAD DATASET
# ============================================

@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")


df = load_data()


# ============================================
# 3. GRAPH FORMATTING HELPERS
# ============================================

def clean_dollar_axis(ax, axis="y", bins=6):

    formatter = FuncFormatter(
        lambda value, _: f"${value:,.0f}"
    )

    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=bins))
    else:
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=bins))

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)


def finish_plot(fig, ax):

    ax.set_title(
        ax.get_title(),
        fontsize=12,
        fontweight="bold"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ============================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================

with st.expander("📊 Exploratory Data Analysis (EDA)"):

    eda_option = st.selectbox(
        "Select a graph",
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

    # ----------------------------------------
    # VIEW DATASET
    # ----------------------------------------

    if eda_option == "View Dataset":

        st.dataframe(df, use_container_width=True)

        st.caption(f"Dataset rows and columns: {df.shape}")

        with st.expander("Missing Values"):
            st.dataframe(
                df.isnull().sum().rename("Missing Values")
            )

        with st.expander("Statistical Summary"):
            st.dataframe(
                df.describe().round(2),
                use_container_width=True
            )

    # ----------------------------------------
    # AGE DISTRIBUTION
    # ----------------------------------------

    elif eda_option == "Age Distribution":

        fig, ax = plt.subplots(figsize=(7, 3.5))

        sns.histplot(
            data=df,
            x="age",
            bins=15,
            kde=True,
            ax=ax,
            color="steelblue"
        )

        ax.set_title("Age Distribution")
        ax.set_xlabel("Age (Years)")
        ax.set_ylabel("Number of People")

        finish_plot(fig, ax)

    # ----------------------------------------
    # BMI DISTRIBUTION
    # ----------------------------------------

    elif eda_option == "BMI Distribution":

        fig, ax = plt.subplots(figsize=(7, 3.5))

        sns.histplot(
            data=df,
            x="bmi",
            bins=20,
            kde=True,
            ax=ax,
            color="teal"
        )

        ax.set_title("BMI Distribution")
        ax.set_xlabel("Body Mass Index (BMI)")
        ax.set_ylabel("Number of People")

        finish_plot(fig, ax)

    # ----------------------------------------
    # INSURANCE CHARGES DISTRIBUTION
    # Exact $2,000-wide bins
    # ----------------------------------------

    elif eda_option == "Insurance Charges Distribution":

        interval = 2000
        max_charge = float(df["charges"].max())

        upper_limit = int(
            np.ceil(max_charge / interval) * interval
        )

        bins = np.arange(
            0,
            upper_limit + interval,
            interval
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.histplot(
            data=df,
            x="charges",
            bins=bins,
            kde=False,
            ax=ax,
            color="steelblue",
            edgecolor="white",
            linewidth=0.6
        )

        ax.set_title("Insurance Charges Distribution")
        ax.set_xlabel("Insurance Charges ($)")
        ax.set_ylabel("Number of People")

        ax.set_xticks(bins)
        ax.set_xticklabels(
            [f"${value:,.0f}" for value in bins],
            rotation=45,
            ha="right",
            fontsize=8
        )

        ax.set_xlim(0, upper_limit)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

        finish_plot(fig, ax)

    # ----------------------------------------
    # GENDER COUNT
    # ----------------------------------------

    elif eda_option == "Gender Count":

        fig, ax = plt.subplots(figsize=(6, 3.5))

        sns.countplot(
            data=df,
            x="sex",
            hue="sex",
            legend=False,
            ax=ax,
            palette="pastel"
        )

        ax.set_title("Customers by Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Number of Customers")

        finish_plot(fig, ax)

    # ----------------------------------------
    # SMOKER COUNT
    # ----------------------------------------

    elif eda_option == "Smoker Count":

        fig, ax = plt.subplots(figsize=(6, 3.5))

        sns.countplot(
            data=df,
            x="smoker",
            hue="smoker",
            legend=False,
            ax=ax,
            palette="pastel"
        )

        ax.set_title("Smoker vs Non-Smoker")
        ax.set_xlabel("Smoker Status")
        ax.set_ylabel("Number of Customers")

        finish_plot(fig, ax)

    # ----------------------------------------
    # REGION COUNT
    # ----------------------------------------

    elif eda_option == "Region Count":

        fig, ax = plt.subplots(figsize=(7, 3.5))

        sns.countplot(
            data=df,
            x="region",
            hue="region",
            legend=False,
            ax=ax,
            palette="pastel"
        )

        ax.set_title("Customers by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("Number of Customers")

        finish_plot(fig, ax)

    # ----------------------------------------
    # AGE VS CHARGES
    # ----------------------------------------

    elif eda_option == "Age vs Charges":

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.scatterplot(
            data=df,
            x="age",
            y="charges",
            hue="smoker",
            hue_order=["no", "yes"],
            alpha=0.65,
            s=28,
            ax=ax
        )

        ax.set_title("Age vs Insurance Charges")
        ax.set_xlabel("Age (Years)")
        ax.set_ylabel("Insurance Charges ($)")

        clean_dollar_axis(ax, "y", bins=6)

        finish_plot(fig, ax)

    # ----------------------------------------
    # BMI VS CHARGES
    # ----------------------------------------

    elif eda_option == "BMI vs Charges":

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.scatterplot(
            data=df,
            x="bmi",
            y="charges",
            hue="smoker",
            hue_order=["no", "yes"],
            alpha=0.65,
            s=28,
            ax=ax
        )

        ax.set_title("BMI vs Insurance Charges")
        ax.set_xlabel("Body Mass Index (BMI)")
        ax.set_ylabel("Insurance Charges ($)")

        clean_dollar_axis(ax, "y", bins=6)

        finish_plot(fig, ax)

    # ----------------------------------------
    # AVERAGE CHARGES BY SMOKER
    # ----------------------------------------

    elif eda_option == "Average Charges by Smoker":

        fig, ax = plt.subplots(figsize=(6, 3.5))

        sns.barplot(
            data=df,
            x="smoker",
            y="charges",
            hue="smoker",
            legend=False,
            errorbar=None,
            ax=ax,
            palette="pastel"
        )

        ax.set_title("Average Charges by Smoker")
        ax.set_xlabel("Smoker Status")
        ax.set_ylabel("Average Charges ($)")

        clean_dollar_axis(ax, "y", bins=6)

        finish_plot(fig, ax)

    # ----------------------------------------
    # AVERAGE CHARGES BY GENDER
    # ----------------------------------------

    elif eda_option == "Average Charges by Gender":

        fig, ax = plt.subplots(figsize=(6, 3.5))

        sns.barplot(
            data=df,
            x="sex",
            y="charges",
            hue="sex",
            legend=False,
            errorbar=None,
            ax=ax,
            palette="pastel"
        )

        ax.set_title("Average Charges by Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Average Charges ($)")

        clean_dollar_axis(ax, "y", bins=6)

        finish_plot(fig, ax)

    # ----------------------------------------
    # AVERAGE CHARGES BY REGION
    # ----------------------------------------

    elif eda_option == "Average Charges by Region":

        fig, ax = plt.subplots(figsize=(7, 3.5))

        sns.barplot(
            data=df,
            x="region",
            y="charges",
            hue="region",
            legend=False,
            errorbar=None,
            ax=ax,
            palette="pastel"
        )

        ax.set_title("Average Charges by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("Average Charges ($)")

        clean_dollar_axis(ax, "y", bins=6)

        finish_plot(fig, ax)

    # ----------------------------------------
    # CORRELATION HEATMAP
    # ----------------------------------------

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

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            corr_df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.4,
            vmin=-1,
            vmax=1,
            center=0,
            annot_kws={"size": 8},
            ax=ax
        )

        ax.set_title("Feature Correlation Heatmap")

        finish_plot(fig, ax)


# ============================================
# 5. DATA PREPROCESSING
# ============================================

@st.cache_data
def preprocess_data(df):

    data = df.copy()

    data["sex"] = data["sex"].map({
        "male": 0,
        "female": 1
    })

    data["smoker"] = data["smoker"].map({
        "no": 0,
        "yes": 1
    })

    data = pd.get_dummies(
        data,
        columns=["region"],
        drop_first=True,
        dtype=int
    )

    for column in data.columns:
        if data[column].dtype == "bool":
            data[column] = data[column].astype(int)

    X = data.drop("charges", axis=1)
    y = data["charges"]

    return X, y


X, y = preprocess_data(df)


# ============================================
# 6. TRAIN MODELS
# ============================================

@st.cache_resource
def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # LINEAR REGRESSION

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    linear_prediction = linear_model.predict(X_test)

    linear_r2 = r2_score(y_test, linear_prediction)
    linear_mae = mean_absolute_error(y_test, linear_prediction)

    linear_rmse = np.sqrt(
        mean_squared_error(y_test, linear_prediction)
    )

    # POLYNOMIAL REGRESSION

    poly = PolynomialFeatures(degree=2)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    poly_prediction = poly_model.predict(X_test_poly)

    poly_r2 = r2_score(y_test, poly_prediction)
    poly_mae = mean_absolute_error(y_test, poly_prediction)

    poly_rmse = np.sqrt(
        mean_squared_error(y_test, poly_prediction)
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
# 7. MODEL PERFORMANCE
# ============================================

st.divider()
st.subheader("📈 Model Performance")

performance = pd.DataFrame({
    "Algorithm": [
        "Linear Regression",
        "Polynomial Regression"
    ],
    "R² Score": [
        linear_r2,
        poly_r2
    ],
    "MAE ($)": [
        linear_mae,
        poly_mae
    ],
    "RMSE ($)": [
        linear_rmse,
        poly_rmse
    ]
})

st.dataframe(
    performance.style.format({
        "R² Score": "{:.3f}",
        "MAE ($)": "${:,.2f}",
        "RMSE ($)": "${:,.2f}"
    }),
    use_container_width=True,
    hide_index=True
)


# ============================================
# 8. PREDICT INSURANCE CHARGES
# ============================================

st.divider()
st.subheader("🔮 Predict Insurance Charges")

# Use two columns to reduce vertical length.
col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=64,
        value=25,
        step=1
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

with col2:

    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    smoker = st.selectbox(
        "Smoker",
        ["no", "yes"]
    )

    region = st.selectbox(
        "Region",
        [
            "northeast",
            "northwest",
            "southeast",
            "southwest"
        ]
    )

algorithm = st.selectbox(
    "Select Prediction Algorithm",
    [
        "Linear Regression",
        "Polynomial Regression"
    ]
)


# ============================================
# 9. PREDICTION BUTTON
# ============================================

if st.button(
    "🚀 Predict Insurance Charges",
    use_container_width=True
):

    user_data = {
        "age": age,
        "sex": 0 if sex == "male" else 1,
        "bmi": bmi,
        "children": children,
        "smoker": 0 if smoker == "no" else 1
    }

    for column in X.columns:

        if column.startswith("region_"):

            region_name = column.replace("region_", "")

            user_data[column] = int(
                region == region_name
            )

    user_input = pd.DataFrame([user_data])

    user_input = user_input.reindex(
        columns=X.columns,
        fill_value=0
    )

    if algorithm == "Linear Regression":

        prediction = linear_model.predict(
            user_input
        )[0]

    else:

        user_input_poly = poly.transform(user_input)

        prediction = poly_model.predict(
            user_input_poly
        )[0]

    st.success(
        f"💰 Predicted Insurance Charges: ${prediction:,.2f}"
    )

    st.caption(f"Algorithm used: {algorithm}")


# ============================================
# 10. PROJECT INFORMATION
# ============================================

with st.expander("📌 Project Information"):

    st.markdown("""
    **Project:** Medical Insurance Cost Prediction

    **Algorithms:** Linear Regression, Polynomial Regression

    **Input Features:** Age, Sex, BMI, Children, Smoker, Region

    **Target Variable:** Insurance Charges

    **Evaluation Metrics:** R² Score, MAE, RMSE

    **Libraries:** Pandas, NumPy, Seaborn, Matplotlib,
    Scikit-learn, Streamlit
    """)

