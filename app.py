
import streamlit as st
import pickle
import pandas as pd
import os

# Page settings
st.set_page_config(
    page_title="IPL Score Prediction",
    page_icon="🏏",
    layout="wide"
)

# Load trained model
MODEL_PATH = "cricket_score_prediction.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

st.title("🏏 IPL Score Prediction")
st.write("Predict cricket scores using your Machine Learning model.")
st.divider()

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found! "
        "Keep cricket_score_prediction.pkl "
        "in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Read model input feature names, if available
features = getattr(model, "feature_names_in_", None)

if features is None:
    st.warning(
        "The model does not expose feature names. "
        "Check your training code for the exact input columns."
    )
    st.stop()

st.subheader("📋 Enter Match Details")

teams = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans"
]

with st.form("prediction_form"):
    values = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        name = str(feature)
        key = name.lower()

        with (col1 if i % 2 == 0 else col2):
            if "team" in key:
                values[name] = st.selectbox(
                    name.replace("_", " ").title(),
                    teams,
                    key=name
                )
            elif any(word in key for word in
                     ["venue", "city", "stadium"]):
                values[name] = st.text_input(
                    name.replace("_", " ").title(),
                    key=name
                )
            else:
                values[name] = st.number_input(
                    name.replace("_", " ").title(),
                    value=0.0,
                    key=name
                )

    submitted = st.form_submit_button(
        "🔮 Predict Score",
        use_container_width=True
    )

if submitted:
    try:
        input_data = pd.DataFrame(
            [values],
            columns=list(features)
        )

        prediction = model.predict(input_data)

        st.success("Prediction completed!")
        st.metric(
            "Predicted Score",
            f"{float(prediction[0]):.0f} Runs"
        )

    except Exception as e:
        st.error("Prediction failed. Check the model's "
                 "expected input features and data types.")
        st.code(str(e))

st.divider()
st.caption("IPL Score Prediction | Machine Learning Project")
