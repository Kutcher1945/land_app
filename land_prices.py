import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
from sklearn.linear_model import LinearRegression
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster  # Import MarkerCluster for clustering
import plotly.express as px


# Set the app title and page icon
st.set_page_config(
    page_title="Прогнозирование стоимости земельных участков",
    page_icon=":house:"
)


# Create a Streamlit caching decorator for data loading and model training
@st.cache_data  # Cache the data and model
def load_data_and_train_model():
    # Load your CSV data
    data = pd.read_csv('data/train_data.csv')

    # Split the dataset into input features (X_train) and target (y_train)
    X_train = data.drop(columns=['price']).values
    y_train = data['price'].values

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return data, model

# Call the cached function to load data and train model
data, model = load_data_and_train_model()

# Create the Streamlit app interface
st.title("Прогнозирование стоимости земельных участков")

# Add an image after the title
st.image("pic1.jpeg", use_column_width=True)

@st.cache_data  # Cache the data and model
def load_land_area():
    # Load your CSV data
    csv_data = pd.read_csv('data/land_area_updated.csv')
    return csv_data


# # Check if data and model are loaded in the session state, and load them if not
# if 'data' not in st.session_state or 'model' not in st.session_state:
#     st.session_state.data, st.session_state.model = load_data_and_train_model()

# # Check if csv_data is loaded in the session state, and load it if not
# if 'csv_data' not in st.session_state:
#     st.session_state.csv_data = load_land_area()


# Load your CSV data
csv_data = load_land_area()

# Function to format price as an integer (removing extra zeros)
def format_price(price):
    return f"₸ {int(price):,}"  # Format as an integer

# # Create a container for the map
# with st.container():
#     # Create a base map
#     @st.cache_resource
#     def create_map():
#         m = folium.Map(location=[43.238293, 76.912471], zoom_start=9, control_scale=True, width=700)

#         # Create a MarkerCluster for clustering
#         marker_cluster = MarkerCluster().add_to(m)

#         # Add markers for each land plot with popups
#         for index, row in csv_data.iterrows():
#             formatted_price = format_price(row['price'])  # Format the price using the function
#             folium.Marker(
#                 location=[row['latitude'], row['longitude']],
#                 popup=f"<b>Адрес:</b> {row['address']}<br><b>Площадь:</b> {row['area']} sq.m<br><b>Цена:</b> {formatted_price}",
#             ).add_to(marker_cluster)  # Add markers to the MarkerCluster for clustering

#         return m

# # Call the create_map function to create or retrieve the cached map
# m = create_map()

# # Display the map in Streamlit using HTML with responsive height
# st.header("Карта земельных участков")
# st.components.v1.html(m._repr_html_(), width=710, height=400)

# ##############NEW MAP###################

# # Create a container for the map
# st.header("Карта земельных участков")

# # Create a DataFrame with columns 'LAT' and 'LON' for latitude and longitude
# marker_data = pd.DataFrame({
#     'LAT': csv_data['latitude'],
#     'LON': csv_data['longitude'],
#     'TOOLTIP': csv_data.apply(lambda row: f"<b>Адрес:</b> {row['address']}<br><b>Площадь:</b> {row['area']} sq.m<br><b>Цена:</b> {format_price(row['price'])}", axis=1)
# })

# # Display the map using st.map with marker cluster
# st.map(marker_data, use_container_width=True)

##############NEW MAP##################


##############NEW MAP###################

# Create a container for the map
st.header("Карта земельных участков")

# Create a function to generate the map
@st.cache_resource
def create_map():
    csv_data['formatted_price'] = csv_data['price'].apply(format_price)
    
    fig = px.scatter_mapbox(csv_data,
                            lat="latitude",
                            lon="longitude",
                            hover_name="address",
                            hover_data=["area", "formatted_price"],
                            zoom=10)
    
    fig.update_layout(mapbox_style="open-street-map")

    return fig

# Call the create_map function to create or retrieve the cached map
map_fig = create_map()

# Display the map in Streamlit
st.plotly_chart(map_fig)

##############NEW MAP##################

# Description
st.markdown("""
##### Веб-приложение для прогнозирования стоимости земельных участков в Алматы
        
Это приложение использует машинное обучение для прогнозирования стоимости земельных участков. 
Оно загружает предварительно обученную модель линейной регрессии, которая принимает на вход различные характеристики участка, 
такие как расстояние до ДДО, школы, медучреждения, дефицит ДДО и школ, количество объектов досуга, наличие парковки, парка, камер видеонаблюдения, 
наличие велодорожки, количество мусорных контейнеров, количество точек интереса и предприятий общественного питания в радиусе 1000 метров, 
индекс ближайшего датчика после полудня и до полудня. 
Приложение предварительно обрабатывает входные данные, объединяя некоторые характеристики и добавляя новые, например, расстояние до ближайшего города.
""")



# # Input fields for property characteristics
# st.header("Характеристики земельного участка")
# # List of property characteristics
# property_characteristics = [
#     "Участок, прилегающий к Роще Баума, в квадрате улиц Сейфуллина, Акан Серы, Успенского, Стоимость, тенге/м 73 000",
#     "Участок на квадрате улиц Кассина, Акан Сери, Сейфуллина, Хетагурова, Стоимость, тенге/м 75 000",
#     "Участок в квадрате улиц Сейфуллина, Котельникова, Акан Серы, Стоимость, тенге/м 70 000",
#     "Участок в квадрате улиц Ауэзова, Жандосова, Манаса, М.Озтюрка, Стоимость, тенге/м 232 000",
#     "Участок в квадрате улиц Манаса, Бухар жырау, Ауэзова, Габдуллина, Стоимость, тенге/м 231 000",
#     "Участок на пересечении улиц Бенберина - Тополиная и Бенберина - Шугыла в мкр.Айгерим-1, Стоимость, тенге/м 56 000",
#     "Участок на пересечении улиц Кисловодская-Левского и 2-ая Кисловодская-Отрарская, Стоимость, тенге/м 72 000",
#     "Участок в квадрате улиц Абая, Тургут Озала, Брусиловского, Толе би, Стоимость, тенге/м 82 000",
#     "Участок в квадрате улиц Розыбакиева, Толе би, И.Каримова, Карасай батыра, Стоимость, тенге/м 90 000",
#     "Участок в мкр. Мамыр - западнее улицы Яссауи, Стоимость, тенге/м 108 000"
# ]
# # Create a dropdown to select property characteristics
# selected_property = st.selectbox("Выберите участок", property_characteristics)

# # You can use the selected_property in your application as needed
# st.write("Вы выбрали следующий участок:", selected_property)
# distance_1000m_ddo = st.checkbox("Наличие ДДО в радиусе 1 км", key="distance_1000m_ddo")
# distance_1000m_schools = st.checkbox("Наличие школы в радиусе 1 км", key="distance_1000m_schools")
# distance_1000m_medical = st.checkbox("Наличие медучреждения в радиусе 1 км", key="distance_1000m_medical")
# is_parking_exists = st.checkbox("Наличие парковки в радиусе 1 км", key="is_parking_exists")
# distance_park_1000m = st.checkbox("Наличие парка в радиусе 1 км", key="distance_park_1000m")
# distance_bikeroad_1000m = st.checkbox("Наличие велодорожки в радиусе 1 км", key="distance_bikeroad_1000m")
# deficit_ddo = st.number_input("Дефицит ДДО", min_value=0, key="deficit_ddo")
# deficit_schools = st.number_input("Дефицит школ", min_value=0, key="deficit_schools")
# amount_dosug_1000m = st.number_input("Количество объектов досуга в радиусе 1000м", min_value=0, key="amount_dosug_1000m")
# amount_of_cameras_1000m = st.number_input("Количество камер видеонаблюдения в радиусе 1000м", min_value=0, key="amount_of_cameras_1000m")
# amount_of_bins_1000m = st.number_input("Количество коммерческих организации в радиусе 1000м", min_value=0, key="amount_of_bins_1000m")
# amount_of_poi_1000m = st.number_input("Количество объектов благоустройства в радиусе 1000м", min_value=0, key="amount_of_poi_1000m")
# amount_of_business19_1000m = st.number_input("Количество предприятий общественного питания в радиусе 1000м", min_value=0, key="amount_of_business19_1000m")
# index_of_nearest_sensor_pm = st.number_input("Показатель датчика качества воздуха после полудня", min_value=0, key="index_of_nearest_sensor_pm")
# index_of_nearest_sensor_am = st.number_input("Показатель датчика качества воздуха до полудня", min_value=0, key="index_of_nearest_sensor_am")

# # Button to make predictions
# if st.button("Предсказать стоимость", key="prediction_button"):
#     # Data preprocessing
#     features = np.array([distance_1000m_ddo, distance_1000m_schools,
#                      distance_1000m_medical, deficit_ddo, deficit_schools, amount_dosug_1000m, is_parking_exists,
#                      distance_park_1000m, amount_of_cameras_1000m, distance_bikeroad_1000m, amount_of_bins_1000m,
#                      amount_of_poi_1000m, amount_of_business19_1000m, index_of_nearest_sensor_pm,
#                      index_of_nearest_sensor_am]).reshape(1, -1)
    
#     # Predict property price
#     predicted_price = model.predict(features)[0]
    
#     # Apply the minimum and maximum limits
#     predicted_price = min(max(predicted_price, 80000), 350000)
    
#     # Display the prediction result
#     st.header("Результат прогноза")
#     st.markdown(
#         """
#         <style>
#         [data-testid="stMetricValue"] {
#             font-size: 40px;
#             color: green;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )
#     st.metric(label='Стоимость земли (м²)', value=f"₸ {predicted_price:.0f}", delta=None)


# Initialize the state variables
# if 'prediction_button_clicked' not in st.session_state:
#     st.session_state.prediction_button_clicked = False

# # Button to make predictions
# if st.button("Предсказать стоимость", key="prediction_button"):
#     st.session_state.prediction_button_clicked = True


st.markdown("""
##### Прогноз стоимости земли после программы реновации
            """)

# Input fields for property characteristics
st.header("Характеристики земельного участка")
with st.form(key="input_form"):
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
    submitted = st.form_submit_button("Предсказать стоимость")

# Button to make predictions
if submitted:
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
    st.metric(label='Стоимость земли (м²)', value=f"₸ {new_predicted_price:.0f}", delta=None)