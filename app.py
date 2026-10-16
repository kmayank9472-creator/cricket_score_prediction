
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="IPL Score Prediction",
    page_icon="🏏",
    layout="centered"
)

# Interface design
st.title("🏏 IPL Score Prediction")
st.markdown("### Machine Learning Based Prediction")
st.write("Enter match details to predict the score.")

# Model file path
MODEL_PATH = Path(__file__).with_name(
    "cricket_score_prediction (1).pkl"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Model loading error!")
    st.code(str(e))
    st.info(
        "Check that the model file is in the same folder as app.py."
    )
    st.stop()

st.divider()
st.subheader("Enter Match Details")

# Input fields
runs_batter = st.number_input(
    "Batter Runs",
    min_value=0,
    max_value=300,
    value=50,
    step=1
)

team_balls = st.number_input(
    "Team Balls Faced",
    min_value=1,
    max_value=120,
    value=60,
    step=1
)

# Prediction button
if st.button("🏏 Predict Score", type="primary",
             use_container_width=True):
    try:
        input_data = pd.DataFrame(
            [[runs_batter, team_balls]],
            columns=["runs_batter", "team_balls"]
        )

        prediction = model.predict(input_data)[0]

        st.success("Prediction completed!")
        st.metric(
            label="Predicted Score",
            value=f"{prediction:.2f} Runs"
        )

    except Exception as e:
        st.error("Prediction failed!")
        st.code(str(e))

st.divider()
st.caption("IPL Score Prediction | Machine Learning Project")
