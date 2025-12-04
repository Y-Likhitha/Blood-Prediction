import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Blood Donation Prediction", layout="wide")

st.title("🩸 Blood Donation Prediction App")
st.write("Upload your dataset and train ML models to predict blood donation behavior.")

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload Excel (BLD-1.xlsx) and CSV (BLD-2.csv) files",
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    df_list = []
    for file in uploaded_files:
        if file.name.endswith(".xlsx"):
            df_list.append(pd.read_excel(file))
        else:
            df_list.append(pd.read_csv(file))

    df = pd.concat(df_list, ignore_index=True)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Drop useless column if exists
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    # EDA SECTION
    st.subheader("📊 Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df['Months since Last Donation'],
        y=df['Number of Donations'],
        hue=df['Made Donation in March 2007'],
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("📈 Histogram of Features")
    num_cols = df.select_dtypes(include="number").columns
    fig, axes = plt.subplots(len(num_cols), 1, figsize=(6, 3 * len(num_cols)))
    if len(num_cols) == 1:
        axes = [axes]

    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f"Histogram - {col}")

    st.pyplot(fig)

    st.subheader("📦 Boxplots")
    fig, axes = plt.subplots(len(num_cols), 1, figsize=(6, 3 * len(num_cols)))
    if len(num_cols) == 1:
        axes = [axes]

    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f"Boxplot - {col}")

    st.pyplot(fig)

    # Outlier Removal
    outcols = ['Number of Donations', 'Total Volume Donated (c.c.)']
    for col in outcols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        ll = q1 - 1.5 * iqr
        ul = q3 + 1.5 * iqr
        df = df[(df[col] > ll) & (df[col] < ul)]

    # Train/Test Split
    ip = df.drop("Made Donation in March 2007", axis=1)
    op = df["Made Donation in March 2007"]

    x_train, x_test, y_train, y_test = train_test_split(ip, op, random_state=42)

    # Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # MODEL TRAINING
    st.subheader("🤖 Model Training Results")

    models = [
        LogisticRegression(),
        SVC(),
        BernoulliNB()
    ]

    for model in models:
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        acc = accuracy_score(y_test, pred)
        st.write(f"**{model.__class__.__name__} Accuracy:** {acc:.4f}")

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    st.write(f"🌲 **Random Forest Accuracy:** {accuracy_score(y_test, pred):.4f}")

    # Best Estimator Search
    params = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    grid = GridSearchCV(RandomForestClassifier(), params, cv=2)
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_

    st.success(f"Best RandomForest Model: {best_model}")

    # Final Model
    final_rf = best_model
    final_rf.fit(x_train, y_train)

    # ---------------------------
    # PREDICTION FORM
    # ---------------------------
    st.subheader("🔮 Predict Donation for New Data")

    col1, col2, col3, col4 = st.columns(4)

    months = col1.number_input("Months since Last Donation", 0, 100, 10)
    wbd = col2.number_input("Number of Donations", 0, 50, 2)
    tvd = col3.number_input("Total Volume Donated (c.c.)", 0, 3000, 500)
    mos = col4.number_input("Months since First Donation", 0, 500, 50)

    if st.button("Predict"):
        new_data = np.array([[months, wbd, tvd, mos]])
        new_data_scaled = sc.transform(new_data)
        result = final_rf.predict(new_data_scaled)[0]

        if result == 1:
            st.success("🟢 Person is likely to donate!")
        else:
            st.error("🔴 Person is unlikely to donate.")
