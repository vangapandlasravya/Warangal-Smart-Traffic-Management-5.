
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Smart Traffic Management",
    page_icon="🚦",
    layout="wide"
)

# Load trained model and encoders
@st.cache_resource
def load_model():
    return joblib.load("traffic_model.pkl")

bundle = load_model()

model = bundle["model"]
le_city = bundle["le_city"]
le_location = bundle["le_location"]
le_day = bundle["le_day"]
le_weather = bundle["le_weather"]
le_traffic = bundle["le_traffic"]
columns = bundle["columns"]

# Website title
st.title("🚦 Smart Traffic Management System")
st.subheader("Warangal Tri-Cities")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Traffic Prediction", "Dashboard", "About"]
)

# Home page
if page == "Home":
    st.header("Welcome!")
    st.write("AI-Based Traffic Prediction System")
    st.write("Locations: Warangal, Hanamkonda and Kazipet")

# Traffic prediction page
elif page == "Traffic Prediction":
    st.header("Predict Traffic")

    city = st.selectbox(
        "Select City",
        ["Warangal", "Hanamkonda", "Kazipet"]
    )

    locations = {
        "Warangal": [
            "Warangal Chowrasta",
            "MGM Hospital",
            "Enumamula",
            "Mulugu Road",
            "Naimnagar",
            "Pochamma Maidan",
            "Fort Warangal"
        ],
        "Hanamkonda": [
            "Hanamkonda Chowrasta",
            "Nakkalagutta",
            "Kakatiya University",
            "Subedari",
            "Public Garden",
            "Adalath"
        ],
        "Kazipet": [
            "Kazipet Railway Station",
            "Fathimanagar",
            "Vishnupuri",
            "Madikonda",
            "Ayodhya Puram",
            "Siddipet Road",
            "Railway Quarters"
        ]
    }

    location = st.selectbox(
        "Select Location",
        locations[city]
    )

    hour = st.slider("Hour", 0, 23, 9)

    day = st.selectbox(
        "Day",
        [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ]
    )

    vehicles = st.number_input(
        "Vehicle Count",
        min_value=0,
        value=250
    )

    capacity = st.number_input(
        "Road Capacity",
        min_value=1,
        value=400
    )

    weather = st.selectbox(
        "Weather",
        ["Clear", "Rainy", "Cloudy"]
    )

    if st.button("Predict Traffic"):

        try:
            data = pd.DataFrame([{
                "City": le_city.transform([city])[0],
                "Location": le_location.transform([location])[0],
                "Hour": hour,
                "Day": le_day.transform([day])[0],
                "Vehicle_Count": vehicles,
                "Road_Capacity": capacity,
                "Weather": le_weather.transform([weather])[0]
            }])

            data = data[columns]

            prediction = model.predict(data)

            result = le_traffic.inverse_transform(
                prediction
            )[0]

            st.subheader("Prediction Result")

            if result == "High":
                st.error("🔴 HIGH TRAFFIC")

            elif result == "Medium":
                st.warning("🟠 MEDIUM TRAFFIC")

            else:
                st.success("🟢 LOW TRAFFIC")

            st.write("City:", city)
            st.write("Location:", location)

            st.caption(
                "Predictions are based on simulated training data."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")

# Dashboard page
elif page == "Dashboard":
    st.header("Traffic Dashboard")

    df = pd.read_csv("warangal_traffic_dataset.csv")

    st.metric("Total Records", len(df))

    st.subheader("Traffic Distribution")

    st.bar_chart(
        df["Traffic_Condition"].value_counts()
    )

    st.subheader("Location-wise Traffic")

    st.bar_chart(
        pd.crosstab(
            df["Location"],
            df["Traffic_Condition"]
        )
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

# About page
elif page == "About":
    st.header("About Our Project")

    st.write("""
    Project: AI-Based Smart Traffic Management System

    Cities:
    - Warangal
    - Hanamkonda
    - Kazipet

    Algorithm: Random Forest Classifier

    Predictions:
    - Low Traffic
    - Medium Traffic
    - High Traffic

    Dataset: Simulated traffic data
    """)

st.divider()

st.caption(
    "Smart Traffic Management System | ML Project"
)
