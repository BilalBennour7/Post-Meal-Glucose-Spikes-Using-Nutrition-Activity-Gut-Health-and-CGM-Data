import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Datascience_project/meals_with_spike_labels.csv")
df = df.dropna(subset=["Spike"])

features = ["Carbs", "Fat", "Protein", "Calories", "Fiber", "Age", "BMI", "Gender"]
df = df[features + ["Spike"]]

gender_encoder = LabelEncoder()
df["Gender"] = gender_encoder.fit_transform(df["Gender"].astype(str))

X = df[features]
y = df["Spike"]

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)
model.fit(X, y)

st.title("Glucose Spike Predictor")
st.write("Please enter your meal and personal information to predict if your meal will cause a glucose spike.")

carbs = st.number_input("Carbohydrates (g)", min_value=0.0)
fat = st.number_input("Fat (g)", min_value=0.0)
protein = st.number_input("Protein (g)", min_value=0.0)
calories = st.number_input("Calories", min_value=0.0)
fiber = st.number_input("Fiber (g)", min_value=0.0)
age = st.number_input("Age", min_value=1.0)
bmi = st.number_input("BMI", min_value=10.0)
gender = st.selectbox("Gender", ["Male", "Female"])

gender_map = {"Male": "M", "Female": "F"}
gender_letter = gender_map[gender]
gender_encoded = gender_encoder.transform([gender_letter])[0]

# Build input array
input_data = np.array([[carbs, fat, protein, calories, fiber, age, bmi, gender_encoded]])


if st.button("Predict Spike"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
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
            x="Outcome",
            y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
            color="Outcome",
            tooltip=["Outcome", "Probability"]
        )
    )

    st.altair_chart(chart, use_container_width=True)

