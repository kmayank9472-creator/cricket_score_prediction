
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="IPL Score Prediction",
    page_icon="🏏",
    layout="centered"
)

st.title("🏏 IPL Score Prediction")
st.write("Machine Learning se score predict karein.")

# Model file load karein
MODEL_PATH = Path(__file__).with_name(
    "cricket_score_prediction(1).pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Model loading error!")
    st.code(str(e))
    st.info(
        "Model ki .pkl file ko app.py ke same folder mein rakhein."
    )
    st.stop()

st.subheader("Match Details")

runs_batter = st.number_input(
    "Batter Runs (runs_batter)",
    min_value=0,
    max_value=300,
    value=45
)

team_balls = st.number_input(
    "Team Balls (team_balls)",
    min_value=1,
    max_value=120,
    value=60
)

if st.button("Predict Score", type="primary"):
    try:
        # Model ke feature names same rakhein
        features = pd.DataFrame(
            [[runs_batter, team_balls]],
            columns=["runs_batter", "team_balls"]
        )

        prediction = model.predict(features)[0]

        st.success("Prediction Successful!")
        st.metric(
            label="Predicted Score",
            value=f"{prediction:.2f} Runs"
        )

    except Exception as e:
        st.error("Prediction error!")
        st.code(str(e))
