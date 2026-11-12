import requests
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import holidays

from school_holidays import school_holidays #this is my .py file, not a package, it has a dictionary with school holidays in Germany


#============================
# Function 1
# Get city coordinates for the city defined by the user
#============================
@st.cache_data(ttl=3600) #this is needed to store the coordinates of a given city for 1 hour, so that we don't have to fetch every time
def get_city_coordinates(city):
    db_password = get_db_password()
    connection = create_connection_string(db_password)

    engine = create_engine(connection)

    # query = text("""
    # SELECT
    #     name,
    #     lon,
    #     lat
    # FROM cities
    # WHERE name = :city 
    #     AND id IN (
    #     SELECT DISTINCT city_id
    #     FROM top3_ger_cities_trips
    #     );
    # """)

    query = text("""
    SELECT
        t.id,
        t.name,
        c.lat,
        c.lon
    FROM top_3_german_cities AS t
    INNER JOIN cities AS c
        ON t.name = c.name
    WHERE t.name = :city
    ;
    """)

    with engine.connect() as connection:
        city_coordinates = pd.read_sql(
            query,
            connection,
            params={"city": city}
        )
    
    engine.dispose()

    if city_coordinates.empty:
        return None

    lat = float(city_coordinates["lat"].iloc[0])
    lon = float(city_coordinates["lon"].iloc[0])

    return lat, lon

def get_db_password():
    db_password = st.secrets["MY_SQL_PASSWORD"] #this uses the toml file with the password, not .env
    return db_password

def create_connection_string(db_password):
    schema = "bikesharing"
    host = "127.0.0.1" #for local instance
    user = "root"
    password = db_password
    port = 3306
    return f'mysql+pymysql://{user}:{password}@{host}:{port}/{schema}'


#==========================================
# Function 2
# Get weather forecast for the date and time defined by user
#==========================================

def get_weather(lat, lon, date_time, city_name):
    api_key_weather = get_api_key()
    weather_json = get_weather_json(lat, lon, api_key_weather)
    weather_df = get_weather_df(weather_json, city_name)
    selected_forecast = select_forecast(date_time, weather_df)
    return selected_forecast

def get_api_key():
    api_key_weather = st.secrets["API_KEY_WEATHER_NEW"] #this uses the toml file with the password, not .env
    return api_key_weather 

def get_weather_json(lat, lon, api_key_weather):
    weather_request = requests.get(f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={api_key_weather}", timeout=15)
    weather_request.raise_for_status()
    weather_json = weather_request.json()
    return weather_json

def get_weather_df(weather_json, city_name):
    # Create an empty list to store observations
    weather_data = []

    # Loop through each forecast
    for forecast in weather_json["list"]:

        row = {
            "city_name": city_name,
            "city_observation_station": weather_json["city"]["name"],
            "dt": forecast["dt"],
            "dt_txt_utc": forecast["dt_txt"],
            "timezone": weather_json["city"]["timezone"],
            "temp": forecast["main"]["temp"],
            "humidity": forecast["main"]["humidity"],
            "wind_speed": forecast["wind"]["speed"],
            "rain_3h": forecast.get("rain", {}).get("3h", 0),
            "snow_3h": forecast.get("snow", {}).get("3h", 0),
            "weather_main": forecast["weather"][0]["main"]
        }

        # Add the observation to the list
        weather_data.append(row)

    # Convert the list of dictionaries into a DataFrame
    weather_df = pd.DataFrame(weather_data)

    #Add the forecast_start column
    weather_df["forecast_start_time_berlin"] = (
    pd.to_datetime(weather_df["dt"], unit="s", utc=True)
    .dt.tz_convert("Europe/Berlin"))

    #Add the forecast_end column
    weather_df["forecast_end_time_berlin"] = (
    weather_df["forecast_start_time_berlin"] + pd.Timedelta(hours=3))

    #Convert weather_main into weather_group
    weather_df["weather_group"] = weather_df["weather_main"].where(
    ~weather_df["weather_main"].str.contains("Thunderstorm", na=False),
    "Rain")

    #Create rain_1h column based on rain_3h column
    weather_df["rain_1h"] = weather_df["rain_3h"]/3

    #Create snow_1h column based on snow_3h column
    weather_df["snow_1h"] = weather_df["snow_3h"]/3

    return weather_df

def select_forecast(date_time, weather_df):
    user_datetime = pd.Timestamp(date_time).tz_localize("Europe/Berlin")
    
    selected_forecast = weather_df[
    (weather_df["forecast_start_time_berlin"] <= user_datetime) &
    (weather_df["forecast_end_time_berlin"] > user_datetime)]

    return selected_forecast


#==============================
# Function 3
# Tranform selected forecast into the predictors for the model
# (add month and hour, school_break, and working_day)
#===============================

def get_predictors(selected_forecast, date_time):

    selected_forecast = selected_forecast.copy()

    #Month and hour 
    user_datetime = pd.Timestamp(date_time).tz_localize("Europe/Berlin")
    selected_forecast["month"] = user_datetime.month
    selected_forecast["hour"] = user_datetime.hour


    #Working_day
    # Map cities to German federal states
    state_mapping = {
        "Berlin": "BE",
        "Köln": "NW",
        "Dresden": "SN"
    }
    # Get the city selected by the user
    city = selected_forecast["city_name"].iloc[0]
    # Get the corresponding federal state
    state = state_mapping[city]
    # Get German public holidays for the selected year and state
    public_holidays = holidays.Germany(
        years=user_datetime.year,
        subdiv=state
    )
    # Check whether the selected date is a weekday and not a holiday
    is_working_day = (
        user_datetime.weekday() < 5 #Mon is 0, Tue is 1 and so on
        and user_datetime.date() not in public_holidays
    )
    # Add the column
    selected_forecast["working_day"] = int(is_working_day)


    #School_break
    # Extract the date selected by the user
    user_date = user_datetime.date()
    # Check whether the date falls within any school holiday period
    is_school_break = any(
        pd.Timestamp(start).date() <= user_date <= pd.Timestamp(end).date()
        for start, end in school_holidays[city]
    )
    # Add the column
    selected_forecast["school_break"] = int(is_school_break)


    #Finally, choose only necessary columns from the resulting df
    model_features = [
        "city_name",
        "month",
        "hour",
        "weather_group",
        "working_day",
        "school_break",
        "temp",
        "humidity",
        "wind_speed",
        "rain_1h",
        "snow_1h"
    ]

    X_new = selected_forecast[model_features]

    return X_new