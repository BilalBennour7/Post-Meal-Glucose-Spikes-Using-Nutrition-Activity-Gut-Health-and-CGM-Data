import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt
import requests

USDA_API_KEY = st.secrets["USDA_API_KEY"]

@st.cache_data(ttl=3600)
def search_usda_food(query):
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": query,
        "pageSize": 5,
        "dataType": ["Foundation", "SR Legacy"],
        "api_key": USDA_API_KEY
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return []
    return r.json().get("foods", [])

@st.cache_data(ttl=3600)
def get_usda_food(fdc_id):
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
    params = {"api_key": USDA_API_KEY}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return {}
    return r.json()

def extract_macros(nutrients):
    macros = {"carbs": 0, "fat": 0, "protein": 0, "calories": 0, "fiber": 0}
    for n in nutrients:
        nid = n["nutrient"]["id"]
        val = n.get("amount", 0)
        if nid == 1005:
            macros["carbs"] = val
        elif nid == 1004:
            macros["fat"] = val
        elif nid == 1003:
            macros["protein"] = val
        elif nid == 1008:
            macros["calories"] = val
        elif nid == 1079:
            macros["fiber"] = val
    return macros

page= st.query_params.get("page", "input")

model = joblib.load("Datascience_project/glucose_model.pkl")
gender_encoder = joblib.load("Datascience_project/gender_encoder.pkl")


def compute_feature_impact(model, base_input, feature_names):
    base_prob = model.predict_proba(base_input)[0][1]
    rows = []

    for i, name in enumerate(feature_names):
        modified= base_input.copy()
        modified[0, i]= 0
        new_prob= model.predict_proba(modified)[0][1]

        delta= (base_prob- new_prob) * 100

        rows.append({
            "Feature": name,
            "Effect (%)": abs(delta),
            "Direction": "Increases Risk" if delta> 0 else "Reduces Risk"
        })

    return pd.DataFrame(rows)


if "meal_log" not in st.session_state:
    st.session_state["meal_log"] = []

if page=="input":
    st.title("Glucose Spike Predictor")

    if "input_mode" not in st.session_state:
        st.session_state["input_mode"] = "Manual Food Logger"

    st.sidebar.title("Tabs")

    mode = st.sidebar.radio(
        "Choose which tab you'd like to go to!",
        ["Manual Food Logger", "Food Lookup"],
        index=["Manual Food Logger", "Food Lookup"].index(st.session_state["input_mode"])
    )

    st.session_state["input_mode"] = mode


    if mode=="Manual Food Logger":
        st.write("Please enter your meal and personal information to predict if your meal will cause a glucose spike. If no value is entered it is defaulted as zero. ")

        st.markdown("Carbohydrates (g)", help="Carbs raise blood glucose the fastest.")

        carbs = st.number_input(
            "", 
            key="carbs_input",
            min_value=None, 
            value=None, 
            placeholder="ex: 45",
            label_visibility="collapsed"
        )

        st.markdown("Fat (g) ", help="Fat slows digestion and reduces glucose spike speed.")

        fat = st.number_input(
            "",
            key="fat_input",
            min_value=None,
            value=None,
            placeholder="ex: 20",
            label_visibility="collapsed"
        )
        st.markdown("Protein (g) ", help="Protein stabilizes glucose by slowing digestion.")

        protein = st.number_input(
            "",
            key="protein_input",
            min_value=None,
            value=None,
            placeholder="ex: 30",
            label_visibility="collapsed"
        )

        st.markdown("Calories ", help="Higher calories often mean more carbs, which can affect glucose levels.")

        calories = st.number_input(
            "",
            key="cal_input",
            min_value=None,
            value=None,
            placeholder="ex: 500",
            label_visibility="collapsed"
        )
        st.markdown("Fiber (g) ", help="Fiber reduces glucose spikes by slowing carb absorption.")

        fiber = st.number_input(
            "",
            key="fiber_input",
            min_value=None,
            value=None,
            placeholder="ex: 7",
            label_visibility="collapsed"
        )

        st.markdown("Age ", help="Age impacts insulin sensitivity.")

        age = st.number_input(
            "",
            key="age_input",
            min_value=None,
            value=None,
            placeholder="ex: 22",
            label_visibility="collapsed"
        )    

        st.markdown("BMI ", help="Higher BMI may increase spike risk.")

        bmi = st.number_input(
            "",
            key="bmi_input",
            min_value=None,
            value=None,
            placeholder="ex: 25",
            label_visibility="collapsed"
        )

        st.markdown("Gender")

        gender = st.selectbox(
            "",
            ["Male", "Female"],
            label_visibility="collapsed"
        )

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

        st.markdown(
            "### Live Spike Probability Preview",
            help=(
                "This shows a real-time glucose spike risk prediction based on the "
                "values you enter. Adjust carbs, fat, protein, fiber, or other inputs "
                "to instantly see how the spike percentage changes!"
            )
        )

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

    if mode=="Food Lookup":
 
        st.subheader("Food & Beverage Lookup")

        food_query = st.text_input("Search for any food or drink", help="Please make sure to be as specific as possible. For example, instead of just saying 'steak' say 'ribeye steak'.")

        if st.button("Search for Food"):
            st.session_state["usda_results"] = search_usda_food(food_query)

        if "usda_results" in st.session_state and st.session_state["usda_results"]:
            names = [f["description"] for f in st.session_state["usda_results"]]
            selected = st.selectbox("Please select one of the following", names)
            grams = st.slider("Amount (grams)", 1, 500, 100, 1,help= "Use the silder to select how much of this food you want in grams.")

            
            if st.button("Add to meal"):
                st.subheader("Meal Logger", help="Please note that any entry that has zero in it means there is no data on it.")

                food = next(f for f in st.session_state["usda_results"] if f["description"] == selected)
                data = get_usda_food(food["fdcId"])
                nutrients = data.get("foodNutrients", [])

                macros = extract_macros(nutrients)
                scale = grams / 100

                row = {
                    "Food": selected,
                    "Grams": grams,
                    "Carbs (g)": round(macros["carbs"] * scale, 2),
                    "Fat (g)": round(macros["fat"] * scale, 2),
                    "Protein (g)": round(macros["protein"] * scale, 2),
                    "Calories": round(macros["calories"] * scale, 1),
                    "Fiber (g)": round(macros["fiber"] * scale, 2),
                }

                st.session_state["meal_log"].append(row)
                st.success(f"Added {grams}g of {selected}")

        if st.session_state["meal_log"]:
            df = pd.DataFrame(st.session_state["meal_log"])

            totals = df[[
                "Carbs (g)", "Fat (g)", "Protein (g)", "Calories", "Fiber (g)"]].sum()

            total_row = {
                "Food": "TOTAL",
                "Grams": df["Grams"].sum(),
                "Carbs (g)": round(totals["Carbs (g)"], 2),
                "Fat (g)": round(totals["Fat (g)"], 2),
                "Protein (g)": round(totals["Protein (g)"], 2),
                "Calories": round(totals["Calories"], 1),
                "Fiber (g)": round(totals["Fiber (g)"], 2),
            }

            df_display = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

            if "Delete" not in df_display.columns:
                df_display["Delete"] = False

            edited_df = st.data_editor(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Delete": st.column_config.CheckboxColumn(
                        "Delete row",
                        help="Check row(s) to remove",
                        default=False
                    )
                },
                disabled=["Food"] if "TOTAL" in df_display["Food"].values else []
            )

            if st.button("Remove Selected Rows"):
                cleaned = edited_df[
                    (~edited_df["Delete"]) & (edited_df["Food"] != "TOTAL")].drop(columns=["Delete"])

                st.session_state["meal_log"] = cleaned.to_dict("records")
                st.rerun()

            

            age = st.number_input(
                "Age",
                min_value=None,
                step=1,
                value=None
            )

            bmi = st.number_input(
                "BMI",
                min_value=None,
                step=0.1,
                value=None
            )

            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
                index=0
            )

                
            gender_mapping = {"Male": "M", "Female": "F"}
            gender_encoded = gender_encoder.transform([gender_mapping[gender]])[0]

            input_data = np.array([[
                totals["Carbs (g)"],
                totals["Fat (g)"],
                totals["Protein (g)"],
                totals["Calories"],
                totals["Fiber (g)"],
                age,
                bmi,
                gender_encoded
            ]])

            live_prob = model.predict_proba(input_data)[0][1]

            st.write(f"**There is a {live_prob * 100:.1f}% chance of a glucose spike**")
            st.progress(live_prob)


            if st.button("Spike prediction From This Meal"):
                st.session_state["previous_page"] = "input"

                st.session_state.update({
                    "carbs": total_row["Carbs (g)"],
                    "fat": total_row["Fat (g)"],
                    "protein": total_row["Protein (g)"],
                    "calories": total_row["Calories"],
                    "fiber": total_row["Fiber (g)"],
                    "age": age,
                    "bmi": bmi,
                    "gender": "Male"
                })


                st.query_params = {"page": "results"}
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

    st.write("### How Each Factor Affects The Spike Risk")

    feature_names = [
        "Carbs", "Fat", "Protein", "Calories",
        "Fiber", "Age", "BMI", "Gender"
    ]

    impact_df = compute_feature_impact(model, input_data, feature_names)

    impact_chart = (
        alt.Chart(impact_df)
        .mark_bar()
        .encode(
            x=alt.X("Effect (%)", title="Effect on Spike Risk (%)"),
            y=alt.Y("Feature", sort="-x"),
            color=alt.Color(
                "Direction",
                scale=alt.Scale(
                    domain=["Increases Risk", "Reduces Risk"],
                    range=["crimson", "seagreen"]
                )
            ),
            tooltip=["Feature", "Effect (%)", "Direction"]
        )
    )

    st.altair_chart(impact_chart, width="stretch")


    

    if prediction == 1:
        st.write("### Personalized Meal Advice")
        if carbs > 40:
            st.write("- Carbs are quite high. Reduce portion size or choose lower-GI carbs.")
        if fiber < 5:
            st.write("- Increasing fiber can help reduce glucose spikes.")
        if protein < 15:
            st.write("- Adding more protein can slow the spike.")
        if fat < 10:
            st.write("- Healthy fats (olive oil, nuts) help slow digestion.")
        st.write("- Light activity after eating can help reduce spike.")
    

    if st.button("Go Back"):
        st.query_params = {"page": "input"}
        st.rerun()


