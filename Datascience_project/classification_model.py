import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
df=pd.read_csv("meals_with_spike_labels.csv")
df=df.dropna(subset=["Spike"])

#These are the columns we will use to PREDICT whether a meal causes a spike
categories=["Carbs", "Fat", "Protein", "Calories", "Fiber", "Age", "BMI", "Gender"]
df=df[categories + ["Spike"]]

#turns gender into a numerical value
encoder=LabelEncoder()
df["Gender"]=encoder.fit_transform(df["Gender"].astype(str))

#removes any remaining missing values
df=df.dropna()

X=df[categories]
y=df["Spike"]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

model= RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "glucose_model.pkl")
joblib.dump(encoder, "gender_encoder.pkl")

prediction=model.predict(X_test)


#tells us how good the model is 
print("\nAccuracy: ", accuracy_score(y_test, prediction))
#tells us whether the model does better at predicting spikes or non-spikes
print("\nClassification Report:", classification_report(y_test, prediction))
#tells us how many correct and incorrect predictions were made
print("\nConfusion Matrix:", confusion_matrix(y_test, prediction))

#ploting features 
imporances=model.feature_importances_
plt.figure(figsize=(8,5))
sns.barplot(x=imporances, y=categories)
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()
