import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create a DataFrame with the parameters and land prices
data = pd.read_csv('data/train_data.csv')
print(data.columns)

# Split the dataset into input features (X_train) and target (y_train)
X_train = data.drop(columns=['id','price']).values
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

# Add an image after the title
st.image("pic1.jpeg", use_column_width=True)

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
# List of property characteristics
property_characteristics = [
    "Участок, прилегающий к Роще Баума, в квадрате улиц Сейфуллина, Акан Серы, Успенского, Стоимость, тенге/м 73 000",
    "Участок на квадрате улиц Кассина, Акан Сери, Сейфуллина, Хетагурова, Стоимость, тенге/м 75 000",
    "Участок в квадрате улиц Сейфуллина, Котельникова, Акан Серы, Стоимость, тенге/м 70 000",
    "Участок в квадрате улиц Ауэзова, Жандосова, Манаса, М.Озтюрка, Стоимость, тенге/м 232 000",
    "Участок в квадрате улиц Манаса, Бухар жырау, Ауэзова, Габдуллина, Стоимость, тенге/м 231 000",
    "Участок на пересечении улиц Бенберина - Тополиная и Бенберина - Шугыла в мкр.Айгерим-1, Стоимость, тенге/м 56 000",
    "Участок на пересечении улиц Кисловодская-Левского и 2-ая Кисловодская-Отрарская, Стоимость, тенге/м 72 000",
    "Участок в квадрате улиц Абая, Тургут Озала, Брусиловского, Толе би, Стоимость, тенге/м 82 000",
    "Участок в квадрате улиц Розыбакиева, Толе би, И.Каримова, Карасай батыра, Стоимость, тенге/м 90 000",
    "Участок в мкр. Мамыр - западнее улицы Яссауи, Стоимость, тенге/м 108 000"
]
# Create a dropdown to select property characteristics
selected_property = st.selectbox("Выберите участок", property_characteristics)

# You can use the selected_property in your application as needed
st.write("Вы выбрали следующий участок:", selected_property)
# Original property price
selected_property_price = float(selected_property.split("Стоимость, тенге/м ")[1].replace(" ", "").replace("₸", "").replace(",", ""))
original_price = st.number_input("Исходная стоимость (₸/м²)", value=selected_property_price, min_value=0.0, key="original_price", disabled=True)

distance_1000m_ddo = st.checkbox("Наличие ДДО в радиусе 1 км", key="distance_1000m_ddo")
distance_1000m_schools = st.checkbox("Наличие школы в радиусе 1 км", key="distance_1000m_schools")
distance_1000m_medical = st.checkbox("Наличие медучреждения в радиусе 1 км", key="distance_1000m_medical")
is_parking_exists = st.checkbox("Наличие парковки в радиусе 1 км", key="is_parking_exists")
distance_park_1000m = st.checkbox("Наличие парка в радиусе 1 км", key="distance_park_1000m")
distance_bikeroad_1000m = st.checkbox("Наличие велодорожки в радиусе 1 км", key="distance_bikeroad_1000m")
deficit_ddo = st.number_input("Дефицит ДДО", min_value=0, key="deficit_ddo")
deficit_schools = st.number_input("Дефицит школ", min_value=0, key="deficit_schools")
amount_dosug_1000m = st.number_input("Количество объектов досуга в радиусе 1000м", min_value=0, key="amount_dosug_1000m")
amount_of_cameras_1000m = st.number_input("Количество камер видеонаблюдения в радиусе 1000м", min_value=0, key="amount_of_cameras_1000m")
amount_of_bins_1000m = st.number_input("Количество коммерческих организации в радиусе 1000м", min_value=0, key="amount_of_bins_1000m")
amount_of_poi_1000m = st.number_input("Количество объектов благоустройства в радиусе 1000м", min_value=0, key="amount_of_poi_1000m")
amount_of_business19_1000m = st.number_input("Количество предприятий общественного питания в радиусе 1000м", min_value=0, key="amount_of_business19_1000m")
index_of_nearest_sensor_pm = st.number_input("Показатель датчика качества воздуха после полудня", min_value=0, key="index_of_nearest_sensor_pm")
index_of_nearest_sensor_am = st.number_input("Показатель датчика качества воздуха до полудня", min_value=0, key="index_of_nearest_sensor_am")

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
    
    # Calculate the final price by adding or subtracting from the original price
    final_price = original_price + predicted_price
    
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
    st.metric(label='Стоимость земли (м²)', value=f"₸ {final_price:.0f}", delta=None)

st.markdown("""
##### Прогноз стоимости земли после программы реновации
            """)

# Input fields for property characteristics
st.header("Характеристики земельного участка")
new_distance_1000m_ddo = st.checkbox("Наличие ДДО в радиусе 1 км", key="new_distance_1000m_ddo")
new_distance_1000m_schools = st.checkbox("Наличие школы в радиусе 1 км", key="new_distance_1000m_schools")
new_distance_1000m_medical = st.checkbox("Наличие медучреждения в радиусе 1 км", key="new_distance_1000m_medical")
new_is_parking_exists = st.checkbox("Наличие парковки в радиусе 1 км", key="new_is_parking_exists")
new_distance_park_1000m = st.checkbox("Наличие парка в радиусе 1 км", key="new_distance_park_1000m")
new_distance_bikeroad_1000m = st.checkbox("Наличие велодорожки в радиусе 1 км", key="new_distance_bikeroad_1000m")
new_deficit_ddo = st.number_input("Дефицит ДДО", min_value=0, key="new_deficit_ddo")
new_deficit_schools = st.number_input("Дефицит школ", min_value=0, key="new_deficit_schools")
new_amount_dosug_1000m = st.number_input("Количество объектов досуга в радиусе 1000м", min_value=0, key="new_amount_dosug_1000m")
new_amount_of_cameras_1000m = st.number_input("Количество камер видеонаблюдения в радиусе 1000м", min_value=0, key="new_amount_of_cameras_1000m")
new_amount_of_bins_1000m = st.number_input("Количество коммерческих организации в радиусе 1000м", min_value=0, key="new_amount_of_bins_1000m")
new_amount_of_poi_1000m = st.number_input("Количество объектов благоустройства в радиусе 1000м", min_value=0, key="new_amount_of_poi_1000m")
new_amount_of_business19_1000m = st.number_input("Количество предприятий общественного питания в радиусе 1000м", min_value=0, key="new_amount_of_business19_1000m")
new_index_of_nearest_sensor_pm = st.number_input("Показатель датчика качества воздуха после полудня", min_value=0, key="new_index_of_nearest_sensor_pm")
new_index_of_nearest_sensor_am = st.number_input("Показатель датчика качества воздуха до полудня", min_value=0, key="new_index_of_nearest_sensor_am")

# Original property price after renovation
original_price_after_renovation = st.number_input("Исходная стоимость после реновации (₸/м²)", min_value=0, key="original_price_after_renovation")

# Button to make predictions
if st.button("Предсказать стоимость", key="new_prediction_button"):
    # Data preprocessing
    new_features = np.array([new_distance_1000m_ddo, new_distance_1000m_schools,
                         new_distance_1000m_medical, new_deficit_ddo, new_deficit_schools, new_amount_dosug_1000m, new_is_parking_exists,
                         new_distance_park_1000m, new_amount_of_cameras_1000m, new_distance_bikeroad_1000m, new_amount_of_bins_1000m,
                         new_amount_of_poi_1000m, new_amount_of_business19_1000m, new_index_of_nearest_sensor_pm,
                         new_index_of_nearest_sensor_am]).reshape(1, -1)
    
    # Predict property price
    new_predicted_price = model.predict(new_features)[0]
    
    # Apply the minimum and maximum limits
    new_predicted_price = min(max(new_predicted_price, 80000), 350000)
    
    # Calculate the final price by adding or subtracting from the original price after renovation
    final_price_after_renovation = original_price_after_renovation + new_predicted_price
    
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
    st.metric(label='Стоимость земли (м²)', value=f"₸ {final_price_after_renovation:.0f}", delta=None)

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