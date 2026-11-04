
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Smart Traffic Management",
    page_icon="🚦",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("traffic_model.pkl")

bundle = load_model()

model = bundle["model"]
preprocessor = bundle["preprocessor"]
label_encoder = bundle["label_encoder"]

st.title("🚦 Smart Traffic Management System")
st.subheader("Warangal Tri-Cities")

page = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Traffic Prediction", "Dashboard", "About"]
)

if page == "Home":
    st.header("Welcome!")
    st.write("AI-Based Traffic Prediction System")
    st.write("Locations: Warangal, Hanamkonda and Kazipet")

elif page == "Traffic Prediction":
    st.header("Predict Traffic")

    location = st.selectbox(
        "Location",
        ["Warangal", "Hanamkonda", "Kazipet"]
    )

    hour = st.slider("Hour", 0, 23, 9)

    day = st.selectbox(
        "Day",
        ["Monday", "Tuesday", "Wednesday",
         "Thursday", "Friday", "Saturday", "Sunday"]
    )

    vehicles = st.number_input(
        "Vehicle Count", min_value=0, value=250
    )

    capacity = st.number_input(
        "Road Capacity",
        min_value=1,
        value={"Warangal": 400,
               "Hanamkonda": 350,
               "Kazipet": 300}[location]
    )

    weather = st.selectbox(
        "Weather", ["Clear", "Rainy", "Cloudy"]
    )

    if st.button("Predict Traffic"):
        data = pd.DataFrame([{
            "Location": location,
            "Hour": hour,
            "Day": day,
            "Vehicle_Count": vehicles,
            "Road_Capacity": capacity,
            "Weather": weather
        }])

        encoded = preprocessor.transform(data)
        prediction = model.predict(encoded)
        result = label_encoder.inverse_transform(prediction)[0]

        st.subheader("Prediction Result")

        if result == "High":
            st.error("🔴 HIGH TRAFFIC")
        elif result == "Medium":
            st.warning("🟠 MEDIUM TRAFFIC")
        else:
            st.success("🟢 LOW TRAFFIC")

        st.write("Location:", location)

        st.caption("Predictions are based on simulated training data.")

elif page == "Dashboard":
    st.header("Traffic Dashboard")

    df = pd.read_csv("warangal_traffic_dataset.csv")

    st.metric("Total Records", len(df))

    st.subheader("Traffic Distribution")
    st.bar_chart(df["Traffic_Condition"].value_counts())

    st.subheader("Location-wise Traffic")
    st.bar_chart(pd.crosstab(
        df["Location"], df["Traffic_Condition"]
    ))

    st.dataframe(df.head(20), use_container_width=True)

elif page == "About":
    st.header("About Our Project")

    st.write("""
    Project: AI-Based Smart Traffic Management System

    Locations:
    - Warangal
    - Hanamkonda
    - Kazipet

    Algorithm: Random Forest Classifier

    Predictions:
    - Low Traffic
    - Medium Traffic
    - High Traffic
    """)

st.divider()
st.caption("Smart Traffic Management System | ML Project")
