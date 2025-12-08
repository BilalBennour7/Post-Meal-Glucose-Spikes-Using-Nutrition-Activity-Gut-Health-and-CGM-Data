import pandas as pd 

df=pd.read_csv("final_dataset.csv", low_memory=False)

df["Timestamp"]=pd.to_datetime(df["Timestamp"])
df=df.sort_values(["subject","Timestamp"])

meals= df.dropna(subset=["Carbs", "Calories"]).copy()

spike_labels= []

for index, meal in meals.iterrows():
    subject=meal["subject"]
    meal_time=meal["Timestamp"]
    start_glucose=meal["Libre GL"]

    future=df[
        (df["subject"]==subject) &
        (df["Timestamp"]>= meal_time) &
        (df["Timestamp"]<= meal_time+ pd.Timedelta(hours=2))
    ]

    max_glucose= future["Libre GL"].max()

    if pd.isna(start_glucose)or pd.isna(max_glucose):
        spike_labels.append(None)
    else:
        spike=1 if (max_glucose - start_glucose) >=30 else 0
        spike_labels.append(spike)

meals["Spike"]=spike_labels
meals.to_csv("meals_with_spike_labels.csv", index=False)
