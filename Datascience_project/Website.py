import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt
import requests

page= st.query_params.get("page", "input")

model = joblib.load("glucose_model.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")

if page=="input":
    st.title("Glucose Spike Predictor")
    st.write("Please enter your meal and personal information to predict if your meal will cause a glucose spike. If no value is entered it is defaulted as zero. ")

    carbs = st.number_input("Carbohydrates (g)", min_value=None, value=None, placeholder="ex: 45")
    fat = st.number_input("Fat (g)", min_value=None, value=None, placeholder="ex: 20")
    protein = st.number_input("Protein (g)", min_value=None, value=None, placeholder="ex: 30")
    calories = st.number_input("Calories", min_value=None, value=None, placeholder="ex: 500")
    fiber = st.number_input("Fiber (g)", min_value=None, value=None, placeholder="ex: 7")

    age = st.number_input("Age", min_value=None, value=None, placeholder="ex: 22")
    bmi = st.number_input("BMI", min_value=None, value=None, placeholder="ex: 25")
    gender = st.selectbox("Gender", ["Male", "Female"])

    carbs = carbs or 0
    fat = fat or 0
    protein = protein or 0
    calories = calories or 0
    fiber = fiber or 0
    age = age or 0
    bmi = bmi or 0

    #This model is trained on "M" and "F" instead of male and female this is so that streamlit knows what Male and Female is
    gender_mapping= {"Male": "M", "Female": "F"}
    gender_on_website= gender_mapping[gender]
    gender_encoded= gender_encoder.transform([gender_on_website])[0]

    input_data= np.array([[carbs, fat, protein, calories, fiber, age, bmi, gender_encoded]])
    live_probobility= model.predict_proba(input_data)[0][1]
    live_percent = live_probobility * 100                        

    st.write("### Live Spike Probability Preview")
    st.write(f"**{live_percent:.1f}% Spike Risk**")
    st.progress(live_probobility)
 

    if st.button("Predict Spike"):
        prediction= model.predict(input_data)
        probability= model.predict_proba(input_data)[0]

        st.session_state["carbs"] = carbs
        st.session_state["fat"]= fat
        st.session_state["protein"]= protein
        st.session_state["calories"]= calories
        st.session_state["fiber"]= fiber
        st.session_state["age"]= age
        st.session_state["bmi"]= bmi
        st.session_state["gender"]= gender
        
        st.query_params= {"page": "results"}
        st.rerun() 

if page== "results":
    st.title("Prediction Results")

    carbs = st.session_state.get("carbs", 0)
    fat = st.session_state.get("fat", 0)
    protein = st.session_state.get("protein", 0)
    calories = st.session_state.get("calories", 0)
    fiber = st.session_state.get("fiber", 0)
    age = st.session_state.get("age", 0)
    bmi = st.session_state.get("bmi", 0)
    gender = st.session_state.get("gender", "Male")


    gender_mapping= {"Male": "M", "Female": "F"}
    gender_encoded= gender_encoder.transform([gender_mapping[gender]])[0]

    input_data = np.array([[carbs, fat, protein, calories, fiber, age, bmi, gender_encoded]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]


    if prediction == 1:
        st.error("This meal will likely cause a glucose spike.")
    else:
        st.success("This meal will likely not cause a glucose spike.")

    # Probability chart
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

    st.altair_chart(chart, width="stretch")

    # Advice
    st.write("### Personalized Meal Advice")

    if prediction == 1:
        if carbs > 40:
            st.write("- Carbs are quite high. Reduce portion size or choose lower-GI carbs.")
        if fiber < 5:
            st.write("- Increasing fiber can help reduce glucose spikes.")
        if protein < 15:
            st.write("- Adding more protein can slow the spike.")
        if fat < 10:
            st.write("- Healthy fats (olive oil, nuts) help slow digestion.")
        st.write("- Light activity after eating can help reduce spike.")
    else:
        st.write("- Great macro balance! Protein and fiber helps keep glucose stable.")
        if carbs < 30:
            st.write("- Moderate carbs contribute to stability.")
        if fat < 20:
            st.write("- Healthy fat levels help with glucose control.")

    # Back button
    if st.button("Go Back"):
        st.query_params = {"page": "input"}
        st.rerun()
