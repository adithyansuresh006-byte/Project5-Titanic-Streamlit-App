
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please add model.joblib to this folder.")
    st.stop()

# App title
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict their survival probability.")

st.divider()

# Passenger inputs
st.subheader("Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Gender", ["Female", "Male"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses", 0, 8, 0)
parch = st.number_input("Number of Parents/Children", 0, 6, 0)
fare = st.number_input("Ticket Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Prediction
if st.button("Predict Survival", type="primary"):

    # Convert inputs into the exact training feature format
    passenger = pd.DataFrame([{
        "pclass": pclass,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "sex_female": 1 if sex == "Female" else 0,
        "sex_male": 1 if sex == "Male" else 0,
        "embarked_C": 1 if embarked == "C" else 0,
        "embarked_Q": 1 if embarked == "Q" else 0,
        "embarked_S": 1 if embarked == "S" else 0
    }])

    passenger = passenger[[
        "pclass", "age", "sibsp", "parch", "fare",
        "sex_female", "sex_male",
        "embarked_C", "embarked_Q", "embarked_S"
    ]]

    # Get prediction probability
    probability = model.predict_proba(passenger)[0][1]
    prediction = model.predict(passenger)[0]

    st.divider()
    st.subheader("Prediction Result")

    st.metric("Estimated Survival Probability", f"{probability * 100:.2f}%")

    if prediction == 1:
        st.success("The model predicts that this passenger may survive.")
    else:
        st.warning("The model predicts that this passenger may not survive.")

    st.caption("This is an educational prediction based on a Titanic dataset, not a real-world prediction or guarantee.")
