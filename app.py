
import streamlit as st
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="IPL Score Prediction",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 IPL Score Prediction")
st.write("Predict Team Runs using Machine Learning")
st.divider()

# Model file path
MODEL_PATH = "cricket_score_prediction.pkl"

# Load trained model
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

# Check model file
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found!")
    st.info("Keep cricket_score_prediction.pkl in the same folder as app.py")
    st.stop()

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

st.subheader("📊 Enter Match Details")

# Input features
runs_batter = st.number_input(
    "Runs Batter",
    min_value=0,
    max_value=6,
    value=1,
    step=1
)

team_balls = st.number_input(
    "Team Balls",
    min_value=0,
    value=0,
    step=1
)

# Prediction button
if st.button("🔮 Predict Team Runs", use_container_width=True):

    try:
        # Feature order must match model training
        input_data = pd.DataFrame(
            [[runs_batter, team_balls]],
            columns=["runs_batter", "team_balls"]
        )

        prediction = model.predict(input_data)

        st.success("Prediction Completed!")

        st.metric(
            label="Predicted Team Runs",
            value=f"{float(prediction[0]):.0f} Runs"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()
st.caption("IPL Score Prediction | Machine Learning Project")
