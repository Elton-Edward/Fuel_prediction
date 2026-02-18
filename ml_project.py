
# 🚗 Premium Fuel Consumption Dashboard


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Fuel Consumption Dashboard",
    page_icon="🚗",
    layout="wide"
)


# Custom CSS Styling

st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
h1 {
    color: #1f4e79;
}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.stMetric {
    background-color: rgba(1,56,78,);
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
.sidebar .sidebar-content {
    background-color: #1f4e79;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# Load Models

@st.cache_resource
def load_models():
    lr = joblib.load("model.pkl")
    dt = joblib.load("decision_tree.pkl")
    return lr, dt

lr_model, dt_model = load_models()


# Sidebar

st.sidebar.title("🚗 Navigation")
page = st.sidebar.radio(
    "",
    ["Prediction", "About", "Visualization"]
)

st.sidebar.markdown("---")
st.sidebar.write("Professional ML Dashboard")


# PREDICTION PAGE

if page == "Prediction":

    st.title("🚗 Fuel Consumption Prediction System")

    st.markdown("### 🔧 Enter Vehicle Specifications")

    col1, col2 = st.columns(2)

    with col1:
        cylinders = st.slider("Cylinders", 3, 12, 4)
        displacement = st.number_input("Displacement", 50.0, 500.0, 150.0)
        horsepower = st.number_input("Horsepower", 40.0, 300.0, 100.0)
        weight = st.number_input("Weight (lbs)", 1500.0, 6000.0, 2500.0)

    with col2:
        acceleration = st.number_input("Acceleration", 5.0, 30.0, 15.0)
        model_year = st.slider("Model Year", 70, 82, 76)
        origin_label = st.selectbox("Vehicle Origin",["Europe", "USA", "Japan"])

    selected_model = st.selectbox(
        "Select Model",
        ["Linear Regression", "Decision Tree"]
    )

    st.markdown("")

    if st.button("🔍 Predict Fuel Consumption"):

        origin_usa = 0
        origin_japan = 0

        if origin_label == "USA":
            origin_usa = 1
        elif origin_label == "Japan":
            origin_japan = 1

        input_df = pd.DataFrame([{
            "cylinders": cylinders,
            "displacement": displacement,
            "horsepower": horsepower,
            "weight": weight,
            "acceleration": acceleration,
            "model_year": model_year,
            "origin_usa": origin_usa,
            "origin_japan": origin_japan }])


        input_df = input_df[lr_model.feature_names_in_]
        
        if selected_model == "Linear Regression":
            prediction = lr_model.predict(input_df)[0]
        else:
            prediction = dt_model.predict(input_df)[0]

        st.markdown("### 📊 Prediction Result")

        st.metric(
            label="Estimated Fuel Consumption",
            value=f"{prediction:.2f} L/100km"
        )

        if prediction < 6:
            st.success("✅ Excellent Fuel Efficiency")
        elif prediction < 9:
            st.info("ℹ Moderate Consumption")
        else:
            st.warning("⚠ High Fuel Consumption")

# MODEL COMPARISON PAGE

elif page == "About":

    st.title("📘 About This Project")
    st.markdown("""
    This is AI application that predicts vehicle fuel consumption measured in Miles Per Gallon(MPG).
 
    ### 📊 Dataset:
    Auto MPG Dataset
    """)

   
    

# VISUALIZATION PAGE

elif page == "Visualization":

    st.title("📊 Feature Relationship Visualization")

    hp = np.linspace(50, 250, 50)
    fuel = 15 - (hp * 0.03)

    fig, ax = plt.subplots()
    ax.scatter(hp, fuel)
    ax.set_xlabel("Horsepower")
    ax.set_ylabel("Fuel Consumption (L/100km)")
    ax.set_title("Horsepower vs Fuel Consumption")

    st.pyplot(fig)

    st.markdown("""
    🔎 Observation:
    Vehicles with higher horsepower tend to consume more fuel.
    """)



# Footer

st.markdown("---")
st.caption("🚗 Developed with Streamlit | Machine Learning Project")