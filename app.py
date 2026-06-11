import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)
# Load the trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
st.title("Iris Flower Prediction")
st.write("Enter the measurements of the iris flower to predict its type.")
st.divider()
#input sliders
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

    st.divider()
    #Predict button
    if st.button("Predict Species",use_container_width=True):
        #Prepare input data for the model
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        #Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        #Map number to flower name
        flower_names = ["Setosa", "Versicolor", "Virginica"]
        predicted_flower = flower_names[prediction]
        confidence = probability[prediction] * 100
        #Display result
        st.success(f"Predicted Species: {predicted_flower}**")
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        #Show probability chart
        st.write("Prediction Probabilities:**")
        prob_dict = {flower_names[i]: f"{probability[i] * 100:.2f}%" for i in range(len(flower_names))}
        st.bar_chart(prob_dict)

       