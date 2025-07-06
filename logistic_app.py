import streamlit as st
import numpy as np
import joblib

# Load the trained model (no need for encoder)
model = joblib.load('model.pkl')

st.title("🚢 Titanic Survival Predictor")

Pclass = st.selectbox('Passenger Class', [1, 2, 3])
Sex = st.selectbox('Sex', ['male', 'female'])
Age = st.slider('Age', 0, 80, 25)
Fare = st.number_input('Fare', 0.0, 600.0, 32.0)
Embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Safely encode Sex
Sex_encoded = 1 if Sex == 'male' else 0

# One-hot encode Embarked
Embarked_Q = 1 if Embarked == 'Q' else 0
Embarked_S = 1 if Embarked == 'S' else 0

if st.button('Predict Survival'):
    input_data = np.array([[Pclass, Sex_encoded, Age, Fare, Embarked_Q, Embarked_S]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🎉 The passenger is predicted to SURVIVE! Probability: {probability:.2%}")
    else:
        st.error(f"❌ The passenger is predicted NOT to survive. Probability: {probability:.2%}")
