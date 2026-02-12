import streamlit as st
import pickle
import pandas as pd

# Load model
with open("models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details below:")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Feature Engineering
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# Convert input into dataframe
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "FamilySize": [family_size],
    "IsAlone": [is_alone],
    "Sex_male": [1 if sex == "male" else 0],
    "Embarked_Q": [1 if embarked == "Q" else 0],
    "Embarked_S": [1 if embarked == "S" else 0]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("🎉 Passenger Survived!")
    else:
        st.error("💀 Passenger Did Not Survive")
prob = model.predict_proba(input_data)[0][1]
st.write(f"Survival Probability: {prob:.2f}")

st.markdown("---")
st.caption("Built by Abhishek Yadav | Logistic Regression Model")
