import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import streamlit as st


def generate_dataset(num_samples=5000):
    np.random.seed(42)
    temperature = np.random.uniform(15, 45, num_samples)
    rainfall = np.random.uniform(0, 300, num_samples)
    seismic_activity = np.random.uniform(0, 10, num_samples)
    wind_speed = np.random.uniform(0, 200, num_samples)
    humidity = np.random.uniform(20, 100, num_samples)
    
    disaster_type = []
    for temp, rain, seismic, wind, hum in zip(temperature, rainfall, seismic_activity, wind_speed, humidity):
        if temp > 35 and rain < 50 and hum < 30:
            disaster_type.append('Drought')
        elif rain > 200:
            disaster_type.append('Flood')
        elif seismic > 7:
            disaster_type.append('Earthquake')
        elif wind > 150 and rain > 100:
            disaster_type.append('Hurricane')
        else:
            disaster_type.append('None')
    
    disaster_occurred = [1 if d != 'None' else 0 for d in disaster_type]
    data = pd.DataFrame({
        'temperature': temperature,
        'rainfall': rainfall,
        'seismic_activity': seismic_activity,
        'wind_speed': wind_speed,
        'humidity': humidity,
        'disaster_type': disaster_type,
        'disaster_occurred': disaster_occurred
    })
    return data


def train_models(data):
    X = data[['temperature', 'rainfall', 'seismic_activity', 'wind_speed', 'humidity']]
    y_disaster = data['disaster_occurred']
    y_type = data['disaster_type']

   
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    
    smote = SMOTE(random_state=42)
    X_resampled, y_disaster_resampled = smote.fit_resample(X_normalized, y_disaster)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_disaster_resampled, test_size=0.3, random_state=42)
    disaster_model = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=15, class_weight='balanced')
    disaster_model.fit(X_train, y_train)
    print(f"Disaster Occurrence Model Accuracy: {accuracy_score(y_test, disaster_model.predict(X_test)):.2f}")
    
   
    type_data = data[data['disaster_occurred'] == 1]
    X_type = scaler.transform(type_data[['temperature', 'rainfall', 'seismic_activity', 'wind_speed', 'humidity']])
    y_type_filtered = type_data['disaster_type']
    
    X_type_train, X_type_test, y_type_train, y_type_test = train_test_split(X_type, y_type_filtered, test_size=0.3, random_state=42)
    type_model = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=15, class_weight='balanced')
    type_model.fit(X_type_train, y_type_train)
    print(f"Disaster Type Model Accuracy: {accuracy_score(y_type_test, type_model.predict(X_type_test)):.2f}")
    
    
    joblib.dump(disaster_model, 'disaster_occurrence_model.pkl')
    joblib.dump(type_model, 'disaster_type_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')


def get_precautionary_measures(disaster_type):
    precautions = {
        "Drought": [
            "Conserve water and avoid wastage.",
            "Store water in tanks for essential use.",
            "Avoid outdoor activities during peak heat hours.",
            "Use drought-resistant crops in agriculture."
        ],
        "Flood": [
            "Move to higher ground immediately.",
            "Avoid walking or driving through floodwaters.",
            "Turn off electricity to prevent electrocution.",
            "Keep an emergency kit with food, water, and first aid supplies."
        ],
        "Earthquake": [
            "Take cover under sturdy furniture during shaking.",
            "Stay away from windows and heavy objects.",
            "Identify safe spots in advance, such as door frames.",
            "Have an emergency kit and know your evacuation plan."
        ],
        "Hurricane": [
            "Evacuate if instructed by authorities.",
            "Reinforce windows and doors of your home.",
            "Stock up on food, water, and essential supplies.",
            "Stay indoors and avoid windows during the storm."
        ],
        "None": [
            "No precautions needed. Stay alert and safe!"
        ]
    }
    return precautions.get(disaster_type, ["No specific precautions available."])


def deploy_with_streamlit():
    st.title("Welcome to the Natural Disaster Prediction ")
    
    
    st.sidebar.header("Adjust Input Parameters")
    temperature = st.sidebar.slider("Temperature (°C)", 15.0, 45.0, value=25.0, step=0.1)
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, value=50.0, step=1.0)
    seismic_activity = st.sidebar.slider("Seismic Activity (Magnitude)", 0.0, 10.0, value=0.0, step=0.1)
    wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 200.0, value=10.0, step=1.0)
    humidity = st.sidebar.slider("Humidity (%)", 20.0, 100.0, value=40.0, step=1.0)
    
   
    disaster_model = joblib.load('disaster_occurrence_model.pkl')
    type_model = joblib.load('disaster_type_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
   
    input_data = pd.DataFrame({
        'temperature': [temperature],
        'rainfall': [rainfall],
        'seismic_activity': [seismic_activity],
        'wind_speed': [wind_speed],
        'humidity': [humidity]
    })
    input_normalized = scaler.transform(input_data)
    
  
    disaster_occurrence = disaster_model.predict(input_normalized)[0]
    if disaster_occurrence == 1:
        
        disaster_type = type_model.predict(input_normalized)[0]
        st.warning(f"Disaster Predicted: {disaster_type}")
        precautions = get_precautionary_measures(disaster_type)
        st.subheader("Precautionary Measures")
        for precaution in precautions:
            st.write(f"- {precaution}")
    else:
        st.success("No disaster predicted.")
        st.subheader("Precautionary Measures")
        st.write("- No precautions needed. Stay alert and safe!")

if __name__ == "__main__":
    dataset = generate_dataset()
    train_models(dataset)
    deploy_with_streamlit()
