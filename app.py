import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("construction_cost_model.pkl")

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Construction Cost Predictor",
    page_icon="🏗️",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 15px;
}

.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">🏗️ Construction Cost Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Construction Cost Estimation System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# Building Information
# -----------------------------
st.markdown(
    '<div class="section-title">📋 Building Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input(
        "Area (sq.ft)",
        min_value=100,
        value=1000,
        step=100
    )

with col2:
    floors = st.number_input(
        "Number of Floors",
        min_value=1,
        value=1,
        step=1
    )

with col3:
    material = st.selectbox(
        "Material",
        ["RCC", "Steel"]
    )

col4, col5, col6 = st.columns(3)

with col4:
    location = st.selectbox(
        "Location",
        ["Hyderabad", "Warangal"]
    )

with col5:
    labour_cost = st.number_input(
        "Labour Cost",
        min_value=0.0,
        value=600.0,
        step=50.0
    )

with col6:
    duration = st.number_input(
        "Duration (days)",
        min_value=1,
        value=140,
        step=10
    )

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------
predict_button = st.button(
    "🔮 Predict Construction Cost",
    use_container_width=True
)

if predict_button:

    new_data = pd.DataFrame({
        "Area": [area],
        "Floors": [floors],
        "Material": [material],
        "Location": [location],
        "LabourCost": [labour_cost],
        "Duration": [duration]
    })

    # Predict cost
    predicted_cost = model.predict(new_data)[0]

    # Cost per square foot
    cost_per_sqft = predicted_cost / area

    st.success("Prediction completed successfully!")

    # -----------------------------
    # Estimated Cost
    # -----------------------------
    st.markdown(
        '<div class="section-title">💰 Estimated Construction Cost</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Total Estimated Cost",
            f"₹{predicted_cost:,.2f}"
        )

    with result_col2:
        st.metric(
            "Estimated Cost / sq.ft",
            f"₹{cost_per_sqft:,.2f}"
        )

    # -----------------------------
    # Project Summary
    # -----------------------------
    st.markdown(
        '<div class="section-title">📌 Project Summary</div>',
        unsafe_allow_html=True
    )

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write(f"**Area:** {area:,.0f} sq.ft")
        st.write(f"**Number of Floors:** {floors}")
        st.write(f"**Material:** {material}")

    with summary_col2:
        st.write(f"**Location:** {location}")
        st.write(f"**Labour Cost:** ₹{labour_cost:,.2f}")
        st.write(f"**Duration:** {duration} days")

    # -----------------------------
    # Cost Comparison
    # -----------------------------
    st.markdown(
        '<div class="section-title">📊 Cost Comparison by Area</div>',
        unsafe_allow_html=True
    )

    comparison_areas = [500, 1000, 1500, 2000]

    comparison_data = pd.DataFrame({
        "Area": comparison_areas,
        "Floors": [floors] * 4,
        "Material": [material] * 4,
        "Location": [location] * 4,
        "LabourCost": [labour_cost] * 4,
        "Duration": [duration] * 4
    })

    comparison_costs = model.predict(comparison_data)

    chart_data = pd.DataFrame({
        "Area (sq.ft)": comparison_areas,
        "Predicted Cost": comparison_costs
    })

    st.bar_chart(
        chart_data.set_index("Area (sq.ft)")
    )

# -----------------------------
# Model Performance
# -----------------------------
st.divider()

st.markdown(
    '<div class="section-title">📊 Model Performance</div>',
    unsafe_allow_html=True
)

performance_col1, performance_col2 = st.columns(2)

with performance_col1:
    st.metric(
        "R² Score",
        "0.9843"
    )

with performance_col2:
    st.metric(
        "Mean Absolute Error (MAE)",
        "₹18,812.50"
    )

st.info(
    "R² Score of 0.9843 indicates that the model explained about 98.43% "
    "of the variation in construction cost on the test dataset. "
    "The MAE of ₹18,812.50 means that the model's predictions differed "
    "from the actual costs by about ₹18,812 on average on that test set."
)

st.caption(
    "Note: The model was evaluated using the project's current dataset "
    "and test split. These performance values should not be interpreted "
    "as guaranteed real-world prediction accuracy."
)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    '<div class="footer">'
    'B.Tech Civil Engineering Project | Machine Learning Based Cost Prediction'
    '</div>',
    unsafe_allow_html=True
)
