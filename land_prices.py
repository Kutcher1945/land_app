import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create a DataFrame with the parameters and land prices
data = pd.DataFrame({
    'distance_1000m_ddo': [1, 0, 1, 1, 0],  # Example values
    'distance_1000m_schools': [1, 0, 0, 1, 1],  # Example values
    'distance_1000m_medical': [0, 1, 1, 0, 1],  # Example values
    'deficit_ddo': [10, 5, 8, 12, 7],  # Example values
    'deficit_schools': [15, 8, 10, 14, 9],  # Example values
    'amount_dosug_1000m': [3, 2, 4, 5, 1],  # Example values
    'is_parking_exists': [1, 0, 1, 0, 1],  # Example values
    'distance_park_1000m': [1, 0, 1, 0, 0],  # Example values
    'amount_of_cameras_1000m': [4, 3, 2, 5, 1],  # Example values
    'distance_bikeroad_1000m': [1, 1, 0, 0, 1],  # Example values
    'amount_of_bins_1000m': [2, 1, 3, 2, 0],  # Example values
    'amount_of_poi_1000m': [8, 6, 10, 7, 9],  # Example values
    'amount_of_business19_1000m': [12, 9, 15, 11, 14],  # Example values
    'index_of_nearest_sensor_pm': [20, 25, 18, 22, 23],  # Example values
    'index_of_nearest_sensor_am': [30, 28, 35, 32, 31],  # Example values
    'price': [150000, 160000, 145000, 155000, 158000]  # Example land prices
})

# Split the dataset into input features (X_train) and target (y_train)
X_train = data.drop(columns=['price']).values
y_train = data['price'].values

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Set the app title and page icon
st.set_page_config(
    page_title="Прогнозирование стоимости земельных участков",
    page_icon=":house:"
)

# Create the Streamlit app interface
st.title("Прогнозирование стоимости земельных участков")

# Description
st.markdown("""
##### Веб-приложение для прогнозирования стоимости земельных участков в Алматы.

Это приложение использует машинное обучение для прогнозирования стоимости земельных участков. 
Оно загружает предварительно обученную модель линейной регрессии, которая принимает на вход различные характеристики участка, 
такие как расстояние до ДДО, школы, медучреждения, дефицит ДДО и школ, количество объектов досуга, наличие парковки, парка, камер видеонаблюдения, 
наличие велодорожки, количество мусорных контейнеров, количество точек интереса и предприятий общественного питания в радиусе 1000 метров, 
индекс ближайшего датчика после полудня и до полудня. 
Приложение предварительно обрабатывает входные данные, объединяя некоторые характеристики и добавляя новые, например, расстояние до ближайшего города.
""")

# Input fields for property characteristics
st.header("Характеристики земельного участка")
distance_1000m_ddo = st.checkbox("Наличие ДДО в радиусе 1 км")
distance_1000m_schools = st.checkbox("Наличие школы в радиусе 1 км")
distance_1000m_medical = st.checkbox("Наличие медучреждения в радиусе 1 км")
is_parking_exists = st.checkbox("Наличие парковки в радиусе 1 км")
distance_park_1000m = st.checkbox("Наличие парка в радиусе 1 км")
distance_bikeroad_1000m = st.checkbox("Наличие велодорожки в радиусе 1 км")
deficit_ddo = st.number_input("Дефицит ДДО", min_value=0)
deficit_schools = st.number_input("Дефицит школ", min_value=0)
amount_dosug_1000m = st.number_input("Количество объектов досуга в радиусе 1000м", min_value=0)
amount_of_cameras_1000m = st.number_input("Количество камер видеонаблюдения в радиусе 1000м", min_value=0)
amount_of_bins_1000m = st.number_input("Количество коммерческих организации в радиусе 1000м", min_value=0)
amount_of_poi_1000m = st.number_input("Количество объектов благоустройства в радиусе 1000м", min_value=0)
amount_of_business19_1000m = st.number_input("Количество предприятий общественного питания в радиусе 1000м", min_value=0)
index_of_nearest_sensor_pm = st.number_input("Показатель датчика качества воздуха после полудня", min_value=0)
index_of_nearest_sensor_am = st.number_input("Показатель датчика качества воздуха до полудня", min_value=0)

# Button to make predictions
if st.button("Предсказать стоимость", key="prediction_button"):
    # Data preprocessing
    features = np.array([distance_1000m_ddo, distance_1000m_schools,
                         distance_1000m_medical, deficit_ddo, deficit_schools, amount_dosug_1000m, is_parking_exists,
                         distance_park_1000m, amount_of_cameras_1000m, distance_bikeroad_1000m, amount_of_bins_1000m,
                         amount_of_poi_1000m, amount_of_business19_1000m, index_of_nearest_sensor_pm,
                         index_of_nearest_sensor_am]).reshape(1, -1)
    
    # Predict property price
    predicted_price = model.predict(features)[0]
    
    # Apply the minimum and maximum limits
    predicted_price = min(max(predicted_price, 80000), 350000)
    
    # Display the prediction result
    st.header("Результат прогноза")
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] {
            font-size: 40px;
            color: green;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.metric(label='Стоимость земли (м²)', value=f"₸ {predicted_price:.0f}", delta=None)

# Add a section for the Leaflet map
st.header("Карта земельных участков")

# Load a sample dataset with latitude, longitude, and land prices (replace with your data)
map_data = pd.DataFrame({
    'latitude': [43.238293, 43.255052, 43.274656],
    'longitude': [76.912471, 76.943142, 76.996291],
    'price': [20000, 25000, 30000]
})

# Display the map using st.map()
st.map(map_data)
