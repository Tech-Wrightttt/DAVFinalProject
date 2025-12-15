import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import plotly.express as px

# Load data
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
# Split "Blood Pressure" (e.g., "120/80") into systolic and diastolic columns
df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].str.split("/", expand=True).astype(float)
df["Pulse_Pressure"] = df["Systolic_BP"] - df["Diastolic_BP"]  # optional derived feature: systolic - diastolic[web:129][web:132]


# st.write("Dataset shape:", df.shape)
# st.write(df.info())
# st.dataframe(df.head())
# st.write(df.describe())

# st.write(df.columns)

# --- Linear Regression: predict Sleep Duration ---
# st.header("Linear Regression")
st.title("Sleep Health and Lifestyle Analysis")

st.header("Overview")

st.markdown(
    """
    This app uses the **Sleep Health and Lifestyle** dataset, which contains information about
    individuals' demographics, daily habits, and cardiovascular measurements.

    The main research question is: *How well can lifestyle and health factors predict an individual's sleep duration?*
    To answer this, the app applies simple linear regression for pairwise relationships
    and multiple linear regression to model sleep duration from several predictors.
    """
)

# Basic structure: shape and columns
st.subheader("Dataset structure")

col1, col2 = st.columns(2)
with col1:
    st.metric("Number of rows", df.shape[0])
with col2:
    st.metric("Number of columns", df.shape[1])

st.write("Column names and data types:")
st.dataframe(
    pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str)
    }),
    use_container_width=True
)

st.write("First 5 rows of the dataset:")
st.dataframe(df.head(), use_container_width=True)

st.write("Summary statistics for numeric columns:")
st.dataframe(df.describe(), use_container_width=True)

data_exploration_tab,simple_tab, multi_tab = st.tabs(["Data Preparation and Exploration","Simple linear", "Multiple linear"])

with data_exploration_tab:
    st.subheader("Data Exploration")

    # 1) Show raw structure
    st.write("Basic info about the dataset:")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.dataframe(df.head(), use_container_width=True)

    # 2) Handling / inspecting missing values
    st.markdown("### Missing values and cleaning")

    missing_counts = df.isna().sum()
    st.write("Missing values per column:")
    st.dataframe(missing_counts.to_frame("Missing count"))

    # Updated: Include both systolic/diastolic + pulse pressure for full BP representation
    feature_cols = [
        "Age",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "Systolic_BP",
        "Diastolic_BP",
        "Pulse_Pressure",
        "Sleep Duration"
    ]
    df_clean = df[feature_cols].dropna()

    st.write(
        "After removing rows with missing values in the main numeric features (now including "
        "both systolic/diastolic BP and pulse pressure), the cleaned dataset has "
        f"{df_clean.shape[0]} rows."
    )

    # 3) Distribution visualizations (histograms) - Updated with BP features
    st.markdown("### Distributions of key numeric variables")

    num_cols = [
        "Sleep Duration",
        "Age",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "Systolic_BP",
        "Diastolic_BP",
        "Pulse_Pressure"
    ]

    selected_hist_col = st.selectbox(
        "Select a numeric variable for histogram:",
        num_cols,
        index=0
    )

    fig_hist = px.histogram(
        df_clean,
        x=selected_hist_col,
        nbins=30,
        title=f"Histogram of {selected_hist_col}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # 4) Correlation heatmap - Automatically includes all 10 features now
    st.markdown("### Correlation heatmap of numeric features")

    corr = df_clean.corr(numeric_only=True)

    fig_heat = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation matrix of numeric variables (incl. systolic, diastolic, pulse pressure)"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # 5) Pie charts for categorical distributions
    st.markdown("### Categorical distributions (pie charts)")

    cat_cols = [
        "Gender",
        "Occupation",
        "BMI Category",
        "Sleep Disorder"
    ]

    available_cat_cols = [c for c in cat_cols if c in df.columns]

    if available_cat_cols:
        selected_cat_col = st.selectbox(
            "Select a categorical variable for pie chart:",
            available_cat_cols,
            index=0
        )

        cat_counts = df[selected_cat_col].value_counts().reset_index()
        cat_counts.columns = [selected_cat_col, "Count"]

        fig_pie = px.pie(
            cat_counts,
            names=selected_cat_col,
            values="Count",
            title=f"Distribution of {selected_cat_col}",
            hole=0.3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No expected categorical columns (Gender, Occupation, BMI Category, Sleep Disorder) found to build pie charts.")

    st.markdown(
        "This section inspects missing values, applies a simple cleaning step, and uses histograms, "
        "a correlation heatmap, and pie charts to understand the data before modeling. Blood pressure "
        "is now fully represented as systolic, diastolic, and derived pulse pressure."
    )




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

    # 1) Select features and target (updated: full BP representation)
    feature_cols = [
        "Age",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "Systolic_BP",
        "Diastolic_BP",
        "Pulse_Pressure"
    ]
    X = df[feature_cols]
    y = df["Sleep Duration"]

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Fit Ridge model
    model = Ridge(alpha=1.0)  # L2-regularized multiple linear regression
    model.fit(X_train, y_train)

    # 4) Predictions and metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.write("R²:", r2)
    st.write("RMSE:", rmse)

    # 5) Coefficients table (now includes all BP terms)
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_
    })
    st.write("Model coefficients (Ridge with systolic, diastolic, pulse pressure)")
    st.dataframe(coef_df, use_container_width=True)

    # 6) Actual vs Predicted plot
    pred_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    fig_ap = px.scatter(
        pred_df,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted Sleep Duration (Ridge Regression)"
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

    # 7) Residuals plot
    residuals = y_test - y_pred

    res_df = pd.DataFrame({
        "Index": range(len(residuals)),
        "Residuals": residuals
    })

    fig_res = px.scatter(
        res_df,
        x="Index",
        y="Residuals",
        title="Residuals of the Ridge Regression"
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

        # 8) User input: predict your own sleep duration
    st.markdown("---")
    st.subheader("Predict your own Sleep Duration")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        age_input = st.number_input("Age", min_value=10, max_value=100, value=30, step=1)
        qsleep_input = st.slider("Quality of Sleep (1–10)", min_value=1, max_value=10, value=7)
        pal_input = st.number_input("Physical Activity Level (minutes/day)", min_value=0, max_value=300, value=60, step=5)

    with col_b:
        stress_input = st.slider("Stress Level (1–10)", min_value=1, max_value=10, value=5)
        hr_input = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=70, step=1)
        steps_input = st.number_input("Daily Steps", min_value=0, max_value=50000, value=8000, step=500)

    with col_c:
        systolic_input = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=220, value=120, step=1)
        diastolic_input = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=130, value=80, step=1)
        pulse_input = systolic_input - diastolic_input
        st.write(f"Pulse Pressure (auto): **{pulse_input}** mmHg")

    if st.button("Predict my Sleep Duration"):
        user_row = pd.DataFrame([{
            "Age": age_input,
            "Quality of Sleep": qsleep_input,
            "Physical Activity Level": pal_input,
            "Stress Level": stress_input,
            "Heart Rate": hr_input,
            "Daily Steps": steps_input,
            "Systolic_BP": systolic_input,
            "Diastolic_BP": diastolic_input,
            "Pulse_Pressure": pulse_input
        }])

        user_pred = model.predict(user_row)[0]

        st.success(f"Estimated Sleep Duration: **{user_pred:.2f} hours per night**")

        # --- Cool visualizations for the prediction ---

        vis_col1, vis_col2 = st.columns(2)

        # 1) Gauge / indicator showing predicted sleep vs typical range
        with vis_col1:
            import plotly.graph_objects as go

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=user_pred,
                title={"text": "Your Predicted Sleep (hours)"},
                delta={"reference": 7.5, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                gauge={
                    "axis": {"range": [4, 10]},
                    "bar": {"color": "dodgerblue"},
                    "steps": [
                        {"range": [4, 6], "color": "#ffcccc"},
                        {"range": [6, 7], "color": "#ffe5b4"},
                        {"range": [7, 9], "color": "#ccffcc"},
                        {"range": [9, 10], "color": "#e0e0ff"}
                    ],
                    "threshold": {
                        "line": {"color": "purple", "width": 4},
                        "thickness": 0.75,
                        "value": user_pred
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # 2) Histogram of dataset sleep with your prediction marked
        with vis_col2:
            fig_user_hist = px.histogram(
                df,
                x="Sleep Duration",
                nbins=30,
                title="Where you sit vs dataset"
            )
            fig_user_hist.add_vline(
                x=user_pred,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Your prediction: {user_pred:.2f}h",
                annotation_position="top"
            )
            st.plotly_chart(fig_user_hist, use_container_width=True)

        # Optional text interpretation
        st.markdown(
            """
            - Green zone on the gauge highlights roughly 7–9 hours, often cited as a healthy adult range.
            - The histogram shows how your predicted sleep compares to others in this dataset.
            """
        )





