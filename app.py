

import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Warangal Smart Traffic Management",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------
# LOAD MACHINE LEARNING MODEL
# --------------------------------

@st.cache_resource
def load_model():
    return joblib.load("traffic_model.pkl")

try:
    bundle = load_model()

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    label_encoder = bundle["label_encoder"]

except Exception as e:
    st.error("Unable to load the traffic prediction model.")
    st.exception(e)
    st.stop()


# --------------------------------
# TRAFFIC LOCATIONS
# --------------------------------

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


# --------------------------------
# WEBSITE HEADER
# --------------------------------

st.title("🚦 Smart Traffic Management System")
st.subheader("Warangal Tri-Cities")

st.write(
    "An AI-based system for predicting traffic "
    "conditions in Warangal, Hanamkonda and Kazipet."
)

# --------------------------------
# SIDEBAR
# --------------------------------

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Home",
        "Traffic Prediction",
        "Dashboard",
        "About"
    ]
)


# --------------------------------
# HOME PAGE
# --------------------------------

if page == "Home":

    st.header("Welcome to Smart Traffic Management!")

    st.write(
        "Our project uses Machine Learning to predict "
        "traffic conditions and help understand traffic "
        "patterns across the Warangal Tri-Cities."
    )

    st.subheader("Areas Covered")

    col1, col2, col3 = st.columns(3)

    col1.metric("City 1", "Warangal")
    col2.metric("City 2", "Hanamkonda")
    col3.metric("City 3", "Kazipet")

    st.subheader("Key Features")

    st.markdown("""
    - Traffic condition prediction
    - Selection of traffic locations
    - Vehicle count analysis
    - Weather-based input
    - Traffic distribution dashboard
    - Machine Learning classification
    """)

    st.info(
        "Select Traffic Prediction from the sidebar "
        "to test the prediction system."
    )


# --------------------------------
# TRAFFIC PREDICTION PAGE
# --------------------------------

elif page == "Traffic Prediction":

    st.header("Traffic Condition Prediction")

    city = st.selectbox(
        "Select City",
        list(locations.keys())
    )

    location = st.selectbox(
        "Select Traffic Location",
        locations[city]
    )

    hour = st.slider(
        "Select Hour",
        0, 23, 9
    )

    day = st.selectbox(
        "Select Day",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
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
            # Read the input columns expected by the model
            expected_columns = list(
                preprocessor.feature_names_in_
            )

            # Build input according to the saved preprocessor
            input_values = {}

            for column in expected_columns:

                if column == "City":
                    input_values[column] = city

                elif column == "Location":
                    input_values[column] = location

                elif column == "Hour":
                    input_values[column] = hour

                elif column == "Day":
                    input_values[column] = day

                elif column == "Vehicle_Count":
                    input_values[column] = vehicles

                elif column == "Road_Capacity":
                    input_values[column] = capacity

                elif column == "Weather":
                    input_values[column] = weather

                else:
                    raise ValueError(
                        f"Unknown model input column: {column}"
                    )

            data = pd.DataFrame(
                [input_values],
                columns=expected_columns
            )

            # Transform input
            encoded_data = preprocessor.transform(data)

            # Predict traffic
            prediction = model.predict(encoded_data)

            result = label_encoder.inverse_transform(
                prediction
            )[0]

            st.subheader("Prediction Result")

            if result == "High":
                st.error("🔴 HIGH TRAFFIC")

            elif result == "Medium":
                st.warning("🟠 MEDIUM TRAFFIC")

            elif result == "Low":
                st.success("🟢 LOW TRAFFIC")

            else:
                st.write("Predicted Traffic:", result)

            st.write("City:", city)
            st.write("Location:", location)

            st.caption(
                "Predictions are based on simulated training data, "
                "not live traffic measurements."
            )

        except Exception as e:
            st.error("Unable to predict traffic.")
            st.write("Error details:", str(e))


# --------------------------------
# DASHBOARD PAGE
# --------------------------------

elif page == "Dashboard":

    st.header("Traffic Dashboard")

    try:
        df = pd.read_csv(
            "warangal_traffic_dataset.csv"
        )

        st.metric(
            "Total Traffic Records",
            len(df)
        )

        st.subheader("Traffic Condition Distribution")

        traffic_counts = (
            df["Traffic_Condition"].value_counts()
        )

        st.bar_chart(traffic_counts)

        st.subheader("Location-wise Traffic")

        location_data = pd.crosstab(
            df["Location"],
            df["Traffic_Condition"]
        )

        st.bar_chart(location_data)

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

    except Exception as e:
        st.error("Unable to load the traffic dataset.")
        st.write(str(e))


# --------------------------------
# ABOUT PAGE
# --------------------------------

elif page == "About":

    st.header("About Our Project")

    st.write("""
    Project Title:
    AI-Based Smart Traffic Management System

    Areas Covered:
    - Warangal
    - Hanamkonda
    - Kazipet

    Machine Learning Algorithm:
    Random Forest Classifier

    Traffic Categories:
    - Low Traffic
    - Medium Traffic
    - High Traffic

    Dataset:
    Simulated Traffic Dataset
    """)

    st.subheader("Project Objective")

    st.write(
        "The objective is to predict traffic conditions "
        "at selected locations using Machine Learning."
    )


# --------------------------------
# FOOTER
# --------------------------------

st.divider()

st.caption(
    "Smart Traffic Management System | Machine Learning Project"
)
