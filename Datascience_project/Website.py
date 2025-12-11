import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt

model = joblib.load("Datascience_project/glucose_model.pkl")
gender_encoder = joblib.load("Datascience_project/gender_encoder.pkl")

st.title("Glucose Spike Predictor")
st.write("Please enter your meal and personal information to predict if your meal will cause a glucose spike.")

carbs = st.number_input("Carbohydrates (g)", value=None)
fat = st.number_input("Fat (g)", value=None)
protein = st.number_input("Protein (g)", value=None)
calories = st.number_input("Calories", value=None)
fiber = st.number_input("Fiber (g)", value=None)
age = st.number_input("Age", value=None)
bmi = st.number_input("BMI", value=None)
gender = st.selectbox("Gender", options=["Male", "Female"])

#This model is trained on "M" and "F" instead of male and female this is so that streamlit knows what Male and Female is
gender_mapping= {"Male": "M", "Female": "F"}
gender_on_website= gender_mapping[gender]

gender_encoded= gender_encoder.transform([gender_on_website])[0]
input_data= np.array([[carbs, fat, protein, calories, fiber, age, bmi, gender_encoded]])
live_probobility= model.predict_proba(input_data)[0][1]
live_percent = live_probobility * 100                        

st.write("### 🔍 Live Spike Probability Preview")
st.write(f"**{live_percent:.1f}% Spike Risk**")
st.progress(live_probobility)

if st.button("Predict Spike"):
    prediction= model.predict(input_data)
    probability= model.predict_proba(input_data)[0]

    if prediction== 1:
        st.error("This meal will likely cause a glucose spike.")
    else:
        st.success("This meal will likely not cause a glucose spike.")

    prob_df = pd.DataFrame({
        "Outcome": ["No Spike", "Spike"],
        "Probability": [probability[0], probability[1]]
    })

    chart = (
        alt.Chart(prob_df)
        .mark_bar(size=60)
        .encode(
            x=alt.X("Outcome", sort=None),
            y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Outcome", scale=alt.Scale(scheme="blues")),
            tooltip=["Outcome", "Probability"]
        )
    )

    st.altair_chart(chart, use_container_width=True)
