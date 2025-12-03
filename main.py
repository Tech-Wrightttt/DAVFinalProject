import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import plotly.express as px

# Load data
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
# Convert "Blood Pressure" like "120/80" to numeric systolic value 120
df["Systolic_BP"] = df["Blood Pressure"].str.split("/").str[0].astype(float)


# st.write("Dataset shape:", df.shape)
# st.write(df.info())
# st.dataframe(df.head())
# st.write(df.describe())

# st.write(df.columns)

# --- Linear Regression: predict Sleep Duration ---
st.header("Linear Regression")

simple_tab, multi_tab = st.tabs(["Simple linear", "Multiple linear"])

with simple_tab:
    st.subheader("Simple linear relationships")
    st.write("Scatter plots with regression lines")

    fig1 = px.scatter(
        df,
        x="Age",
        y="Sleep Duration",
        trendline="ols",
        title="Sleep Duration vs Age"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        df,
        x="Physical Activity Level",
        y="Sleep Duration",
        trendline="ols",
        title="Sleep Duration vs Physical Activity Level"
    )
    st.plotly_chart(fig2, use_container_width=True)


    fig3 = px.scatter(
        df,
        x="Stress Level",
        y="Sleep Duration",
        trendline="ols",
        title="Sleep Duration vs Stress Level"
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.scatter(
        df,
        x="Quality of Sleep",
        y="Stress Level",
        trendline="ols",
        title="Quality of Sleep vs Stress Level"
    )
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.scatter(
        df,
        x="Quality of Sleep",
        y="Physical Activity Level",
        trendline="ols",
        title="Quality of Sleep vs Physical Activity Level"
    )
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.scatter(
        df,
        x="Heart Rate",
        y="Daily Steps",
        trendline="ols",
        title="Heart Rate vs Daily Steps"
    )
    st.plotly_chart(fig6, use_container_width=True)


    fig7 = px.scatter(
        df,
        x="Systolic_BP",
        y="Age",
     trendline="ols",
        title="Age vs Systolic Blood Pressure"
    )
    st.plotly_chart(fig7, use_container_width=True)

with multi_tab:
    st.subheader("Multiple linear regression: predict Sleep Duration")

    # 1) Select features and target
    feature_cols = [
        "Age",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "Systolic_BP"
    ]
    X = df[feature_cols]
    y = df["Sleep Duration"]

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4) Predictions and metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.write("R²:", r2)
    st.write("RMSE:", rmse)

    # 5) Coefficients table
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_
    })
    st.write("Model coefficients")
    st.dataframe(coef_df)

    # 6) Actual vs Predicted plot
    pred_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    fig_ap = px.scatter(
        pred_df,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted Sleep Duration"
    )
    # y = x line
    fig_ap.add_shape(
        type="line",
        x0=pred_df["Actual"].min(),
        y0=pred_df["Actual"].min(),
        x1=pred_df["Actual"].max(),
        y1=pred_df["Actual"].max(),
        line=dict(color="red", dash="dash")
    )
    st.plotly_chart(fig_ap, use_container_width=True)

    # Compute residuals: e_x = y - f(x)
actual = y_test
predicted = y_pred
residuals = actual - predicted

# Use the test-set index as x (just 0,1,2,... visually)
res_df = pd.DataFrame({
    "Index": range(len(residuals)),
    "Residuals": residuals
})

fig_res = px.scatter(
    res_df,
    x="Index",
    y="Residuals",
    title="Residuals of the Regression"
)

# Horizontal line at 0
fig_res.add_shape(
    type="line",
    x0=res_df["Index"].min(),
    y0=0,
    x1=res_df["Index"].max(),
    y1=0,
    line=dict(color="blue")
)

st.plotly_chart(fig_res, use_container_width=True)


