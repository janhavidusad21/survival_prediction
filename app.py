import streamlit as st
import numpy as np
import joblib

# Load model and encoder
model = joblib.load('model.pkl')
le = joblib.load('label_encoder.pkl')

st.title("🚢 Titanic Survival Predictor")

# User inputs
Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Sex = st.selectbox("Sex", ['male', 'female'])
Age = st.slider("Age", 0, 80, 25)
Fare = st.number_input("Fare", min_value=0.0, value=32.0)
Embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Encode inputs
Sex_encoded = le.transform([Sex])[0]
Embarked_Q = 1 if Embarked == 'Q' else 0
Embarked_S = 1 if Embarked == 'S' else 0

# Predict
if st.button("Predict Survival"):
    input_data = np.array([[Pclass, Sex_encoded, Age, Fare, Embarked_Q, Embarked_S]])
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.success(f"🎉 The passenger is predicted to SURVIVE! Probability: {prob:.2%}")
    else:
        st.error(f"❌ The passenger is predicted NOT to survive. Probability: {prob:.2%}")
