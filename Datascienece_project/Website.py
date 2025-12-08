import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt

model= joblib.load("glucose_model.pkl")
gender_encoder= joblib.load("gender_encoder.pkl")

st.title("Glucose Spike Predictor")
st.write("Please enter your meal and personal information to predict if your meal will cause a glucose spike.")

carbs= st.number_input("Carbohydrates (g)", min_value=0.0)
fat= st.number_input("Fat (g)", min_value=0.0)
protein= st.number_input("Protein (g)", min_value=0.0)
calories= st.number_input("Calories", min_value=0.0)
fiber= st.number_input("Fiber (g)", min_value=0.0)
age= st.number_input("Age", min_value=1.0)
bmi= st.number_input("BMI", min_value=10.0)
gender= st.selectbox("Gender", options=["Male", "Female"])

#the model is trained on "M" and "F" instead of male and female this is so that streamlit knows what Male and Female is
gender_mapping={"Male": "M", "Female": "F"}
gender_on_website=gender_mapping[gender]

gender_encoded= gender_encoder.transform([gender_on_website])[0]
input_data= np.array([[carbs, fat, protein, calories, fiber, age, bmi, gender_encoded]])


if st.button("Predict Spike"):
    prediction=model.predict(input_data)
    probability=model.predict_proba(input_data)[0]
    if prediction==1:
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
            x = alt.X("Outcome", sort=None),
            y = alt.Y("Probability", scale=alt.Scale(domain=[0,1])),
            color = alt.Color("Outcome", scale=alt.Scale(scheme="blues")),
            tooltip=["Outcome", "Probability"]
        )
    )

    st.altair_chart(chart, use_container_width=True)
