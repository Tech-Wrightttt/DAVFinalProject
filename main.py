import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Sleep Health and Lifestyle Analysis",   # text in the browser tab
    page_icon="🌙",                        # emoji favicon
    layout="wide"
)


# Load and clean data
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

    # Safely split Blood Pressure column
    df['Blood Pressure'] = df['Blood Pressure'].astype(str).str.strip()
    bp_split = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic_BP'] = pd.to_numeric(bp_split[0], errors='coerce')
    df['Diastolic_BP'] = pd.to_numeric(bp_split[1], errors='coerce')
    df['Pulse_Pressure'] = df['Systolic_BP'] - df['Diastolic_BP']

    return df


df = load_and_clean_data()

st.title("Sleep Health and Lifestyle Analysis")
st.header("by Team O(1)")

# Fix 1: Correct tabs unpacking (4 tabs, 4 variables)
overview_tab, data_tab, analysis_tab, conclusion_tab = st.tabs([
    "Overview",
    "Data Exploration and Preparation",
    "Analysis and Insights",
    "Conclusion"
])

with overview_tab:
    st.header("Overview")
    st.markdown("""
        This app uses the **Sleep Health and Lifestyle** dataset, which contains information about
        individuals' demographics, daily habits, and cardiovascular measurements.

        The main research question is: *How well can lifestyle and health factors predict an individual's sleep duration?*
        To answer this, the app applies simple linear regression for pairwise relationships
        and multiple linear regression to model sleep duration from several predictors.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of rows", df.shape[0])
    with col2:
        st.metric("Number of columns", df.shape[1])

    st.write("Column names and data types:")
    st.dataframe(pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str)
    }), use_container_width=True)

    st.write("First 5 rows:")
    st.dataframe(df.head(), use_container_width=True)

with data_tab:
    st.subheader("Data Exploration")

    missing_counts = df.isna().sum()
    st.write("Missing values per column:")
    st.dataframe(missing_counts.to_frame("Missing count"), use_container_width=True)

    # Define features consistently
    feature_cols = [
        "Age", "Quality of Sleep", "Physical Activity Level", "Stress Level",
        "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP",
        "Pulse_Pressure", "Sleep Duration"
    ]
    df_clean = df[feature_cols].dropna()

    st.write(f"Cleaned dataset: {df_clean.shape[0]} rows")

    # Histograms
    num_cols = feature_cols[:-1]  # Exclude target
    selected_hist_col = st.selectbox("Select histogram:", num_cols, index=0)
    fig_hist = px.histogram(df_clean, x=selected_hist_col, nbins=30,
                            title=f"Histogram of {selected_hist_col}")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Correlation heatmap
    corr = df_clean.corr()
    fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                         title="Correlation Matrix")
    st.plotly_chart(fig_heat, use_container_width=True)

with analysis_tab:
    # Fix 2: Correct nested tabs syntax
    simple_tab, multi_tab = st.tabs(["Simple linear", "Multiple linear"])

    with simple_tab:
        st.subheader("Simple Linear Relationships")

        plots = [
            ("Age", "Sleep Duration"),
            ("Physical Activity Level", "Sleep Duration"),
            ("Stress Level", "Sleep Duration"),
            ("Quality of Sleep", "Stress Level"),
            ("Quality of Sleep", "Physical Activity Level"),
            ("Heart Rate", "Daily Steps")
        ]

        for x_col, y_col in plots:
            fig = px.scatter(df_clean, x=x_col, y=y_col, trendline="ols",
                             title=f"{y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)

    with multi_tab:
        st.subheader("Multiple Linear Regression")

        # Use cleaned data, no Pulse_Pressure
        model_features = [
            "Age", "Quality of Sleep", "Physical Activity Level", "Stress Level",
            "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP"
        ]

        X = df_clean[model_features]
        y = df_clean["Sleep Duration"]

        # Train/test split on clean data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -------- STANDARDIZATION ADDED HERE --------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # --------------------------------------------

        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        col1, col2 = st.columns(2)
        col1.metric("R² Score", f"{r2:.3f}")
        col2.metric("RMSE", f"{rmse:.3f}")

        # Coefficients now correspond to standardized features
        coef_df = pd.DataFrame({
            "Feature": model_features,
            "Coefficient": model.coef_
        }).sort_values("Coefficient", key=abs, ascending=False)
        st.dataframe(coef_df, use_container_width=True)

        # Plots
        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        fig_ap = px.scatter(pred_df, x="Actual", y="Predicted",
                            title="Actual vs Predicted")
        fig_ap.add_shape(type="line", x0=4, y0=4, x1=10, y1=10,
                         line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_ap, use_container_width=True)

        # User prediction
        st.markdown("---")
        st.subheader("Predict Your Sleep Duration")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            age_input = st.number_input("Age", 10, 100, 30)
            qsleep_input = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
            pal_input = st.number_input("Physical Activity (min/day)", 0, 300, 60, 5)

        with col_b:
            stress_input = st.slider("Stress Level (1-10)", 1, 10, 5)
            hr_input = st.number_input("Heart Rate (bpm)", 40, 200, 70)
            steps_input = st.number_input("Daily Steps", 0, 50000, 8000, 500)

        with col_c:
            systolic_input = st.number_input("Systolic BP", 80, 220, 120)
            diastolic_input = st.number_input("Diastolic BP", 50, 130, 80)

        if st.button("Predict Sleep Duration"):
            # Build a one-row DataFrame with raw values
            user_row = pd.DataFrame([{
                "Age": age_input,
                "Quality of Sleep": qsleep_input,
                "Physical Activity Level": pal_input,
                "Stress Level": stress_input,
                "Heart Rate": hr_input,
                "Daily Steps": steps_input,
                "Systolic_BP": systolic_input,
                "Diastolic_BP": diastolic_input
            }])

            # -------- STANDARDIZE USER INPUT TOO --------
            user_row_scaled = scaler.transform(user_row)
            # --------------------------------------------

            user_pred = model.predict(user_row_scaled)[0]
            st.success(f"**Predicted Sleep: {user_pred:.2f} hours**")

            # Visualizations
            vis_col1, vis_col2 = st.columns(2)

            with vis_col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=user_pred,
                    title={"text": "Predicted Sleep (hours)"},
                    delta={"reference": 7.5},
                    gauge={
                        "axis": {"range": [4, 10]},
                        "bar": {"color": "dodgerblue"},
                        "steps": [
                            {"range": [4, 6], "color": "#ffcccc"},
                            {"range": [6, 7], "color": "#ffe5b4"},
                            {"range": [7, 9], "color": "#ccffcc"},
                            {"range": [9, 10], "color": "#e0e0ff"}
                        ],
                        "threshold": {"line": {"color": "purple", "width": 4}, "value": user_pred}
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with vis_col2:
                fig_hist = px.histogram(df_clean, x="Sleep Duration", nbins=30)
                fig_hist.add_vline(x=user_pred, line_dash="dash", line_color="red",
                                   annotation_text=f"{user_pred:.1f}h")
                st.plotly_chart(fig_hist, use_container_width=True)

with conclusion_tab:
    st.header("Conclusion, Limitations, and Next Steps")

    # --- 1. Answering the overview question ---
    st.subheader("What did this analysis show?")

    st.markdown(
        """
        **Research question recap:**  
        *How well can lifestyle and cardiovascular factors (age, sleep quality, activity, stress, heart rate,
        daily steps, blood pressure) predict nightly sleep duration in this dataset?*

        **Main findings:**

        - A Ridge multiple linear regression model using nine predictors (**Age**, **Quality of Sleep**,
          **Physical Activity Level**, **Stress Level**, **Heart Rate**, **Daily Steps**, **Systolic_BP**,
          **Diastolic_BP**, **Pulse_Pressure**) can explain a substantial portion of the variation in sleep
          duration in this dataset.
        - Higher **sleep quality** and **physical activity** are generally associated with **longer**
          predicted sleep, while higher **stress** and **pulse pressure** are associated with **shorter**
          predicted sleep.
        - The interactive prediction tool demonstrates that changing these inputs (for example, lowering
          stress or improving sleep quality) does change the model’s estimated sleep duration in plausible
          directions.
        """
    )

    # --- 2. Limitations / why the model is not fully accurate ---
    st.subheader("Limitations and what the model is missing")

    st.markdown(
        """
        Even though the model fits this dataset reasonably well, it is **not a complete or fully accurate
        representation of real-world sleep behavior**:

        - **Limited set of predictors**: The model only uses a small subset of factors. In reality, sleep
          duration is also affected by caffeine intake, alcohol, screen time, shift work, noise/light
          exposure, mental health, chronic illnesses, medications, and genetics — none of which are
          captured here.
        - **Self‑reported and simplified measures**: Variables like *Quality of Sleep* and *Stress Level*
          are self-reported on 1–10 scales. These are coarse, subjective measurements and do not capture
          detailed sleep architecture or clinical stress.
        - **Dataset scope and representativeness**: The dataset reflects a specific sample of individuals
          and may not generalize to all age groups, cultures, occupations, or health conditions.
        - **Linear and additive assumptions**: Ridge regression assumes that each factor has a linear,
          additive effect on sleep duration. Real sleep patterns often involve non‑linear effects and
          interactions (for example, stress plus late caffeine plus screen time).
        - **Associational, not causal**: The model finds patterns in this dataset, but it **does not prove
          that changing a single variable will cause a specific change in sleep**. For example, reducing
          pulse pressure or stress may be beneficial, but the estimated impact on sleep duration here is
          only an approximation, not a causal guarantee.
        """
    )

    st.info(
        "In short: the chosen factors do influence the model’s prediction of sleep duration, "
        "but many important determinants of sleep are not included. The tool is best used for "
        "exploration and education, not as a precise or clinical sleep predictor."
    )

    # --- 3. Actionable recommendations (with expanders) ---
    st.subheader("Actionable recommendations")

    with st.expander("Improve sleep quality and routine"):
        st.markdown(
            """
            - Aim for a **consistent bedtime and wake time** to stabilize your sleep schedule.
            - Create a **wind‑down routine** (reduced screens, dim lighting, relaxing activity) to push your
              *Quality of Sleep* score upward in the model inputs.
            - Use the prediction panel to simulate going to bed slightly earlier and see how the estimated
              sleep duration responds.
            """
        )

    with st.expander("Manage stress and mental load"):
        st.markdown(
            """
            - Because higher **Stress Level** is linked to shorter predicted sleep in this model, build
              **small, daily stress‑management habits** (short breaks, breathing exercises, light stretching,
              journaling).
            - In the input panel, try lowering your stress score by 1–2 points and observe how the predicted
              sleep duration changes. Treat this as a *what‑if* exploration, not a promise.
            - If high stress and poor sleep persist, consider talking to a health professional; this app
              cannot assess or diagnose mental health.
            """
        )

    with st.expander("Physical activity and daily movement"):
        st.markdown(
            """
            - Increase **Physical Activity Level** or **Daily Steps** gradually (for example, an extra
              10–15 minutes of walking or +1,000 steps per day) and test those values in the model.
            - Note how moderate increases in activity often lead to slightly longer predicted sleep,
              especially when not scheduled too close to bedtime.
            - Prioritize changes that you can **sustain long‑term** rather than large, short‑lived spikes.
            """
        )

    with st.expander("Cardiovascular health and blood pressure"):
        st.markdown(
            """
            - The model suggests that higher **pulse pressure** and higher blood pressure values are
              associated with modestly shorter sleep duration, reflecting a link between cardiovascular
              strain and sleep.
            - Use the input sliders to see how changes in **Systolic_BP** and **Diastolic_BP** (and therefore
              Pulse_Pressure) affect your predicted sleep, but rely on **actual medical guidance** for any
              decisions about blood pressure management.
            - If you see a pattern of high blood pressure and low predicted sleep, treat it as a prompt to
              monitor your health more closely and consult a professional.
            """
        )

    # --- 4. Interactive insight selector / text box ---
    st.subheader("Explore insights interactively")

    insight_choice = st.selectbox(
        "Choose a question to explore:",
        [
            "How can I increase my predicted sleep?",
            "What happens if I reduce my stress?",
            "What happens if I improve sleep quality?",
            "What happens if I become more active?",
            "How sensitive is the model to blood pressure changes?"
        ]
    )

    if insight_choice == "How can I increase my predicted sleep?":
        st.markdown(
            """
            - Use the **prediction section** to adjust one factor at a time (sleep quality, stress, activity,
              steps) and watch how the gauge and histogram move.
            - Identify which 1–2 changes give the largest increase in predicted sleep for you; focus your
              real-life efforts there.
            - Remember that unmodeled factors (caffeine, light, noise, mental health) can easily override
              these gains in real life.
            """
        )

    elif insight_choice == "What happens if I reduce my stress?":
        st.markdown(
            """
            - Keep all other sliders fixed and lower **Stress Level** step by step (for example, from 8 → 6 → 4).
            - Observe the predicted sleep duration after each change; this illustrates the model’s estimated
              benefit of stress reduction for someone with your profile.
            - Use this as a guide to prioritize stress‑reducing habits (breaks, boundaries, relaxation),
              not as an exact forecast.
            """
        )

    elif insight_choice == "What happens if I improve sleep quality?":
        st.markdown(
            """
            - Increase **Quality of Sleep** in the input panel (e.g., from 5 → 7 → 9) and note how strongly
              predicted sleep responds.
            - This shows that, in this dataset, improving the *experience* of sleep is often as important as
              changing external metrics like steps or heart rate.
            - Experiment in real life with bedroom environment (light, noise, temperature), bedtime routine,
              and screen use, then adjust the slider to match your perceived quality over time.
            """
        )

    elif insight_choice == "What happens if I become more active?":
        st.markdown(
            """
            - Raise **Physical Activity Level** or **Daily Steps** and re-run the prediction to see the
              model’s response.
            - Look for a realistic activity target that nudges predicted sleep upward without being extreme.
            - Keep in mind that very intense late-night exercise can still disrupt sleep even if the model
              predicts more sleep based on total activity minutes.
            """
        )

    elif insight_choice == "How sensitive is the model to blood pressure changes?":
        st.markdown(
            """
            - Adjust **Systolic_BP** and **Diastolic_BP** in the prediction panel and watch how
              **Pulse_Pressure** (auto-calculated) and predicted sleep change.
            - Note that the effects are usually modest; the model treats blood pressure as one contributor
              among many, not the dominant driver of sleep.
            - Use this only for educational insight into the relationship between cardiovascular metrics and
              sleep in this dataset — it does **not** replace proper blood pressure monitoring or care.
            """
        )
