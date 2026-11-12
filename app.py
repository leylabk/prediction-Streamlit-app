
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import SplineTransformer
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, date, time
import requests
import math

from functions import (
    get_city_coordinates,
    get_weather,
    get_predictors
)

df = pd.read_csv("dataset_final.csv")

# Page configuration
st.set_page_config(
    page_title="CatBoost Prediction",
    page_icon="📊",
    layout="wide"
)

# Upload models and categories
BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "catboost_model_spline.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "catboost_categories_spline.pkl", "rb") as f:
    categories = pickle.load(f)

# Title
st.title("Welcome to the Bike Sharing Demand Prediction App! 🚲")

tab1, tab2, tab3, tab4 = st.tabs(["Home", 
                                  "Dataset Explorer", 
                                  "Prediction Tool", 
                                  "Prediction Tool2"])

with tab1:
    st.write("""
        ### About the Project
        This project aims to analyze and predict hourly bike-sharing demand based on various temporal, weather-related, and calendar features. 
        By exploring historical bike usage data, we investigate how different factors influence bicycle rental patterns across cities.

        ### About the Dataset
        This project combines bike-sharing, weather, and calendar data
        from different sources to analyze and predict hourly bicycle usage.
        This project combines bike-sharing, weather, and calendar data
        from different sources to analyze and predict hourly bicycle usage.

        **Data Sources:**

        - **Bike-Sharing Data:** Historical bike-sharing records obtained
        from the [European Bike Sharing Dataset](https://github.com/tumftm/european-bike-sharing-dataset).
        - **Weather Data:** Weather information collected using the
        [OpenWeather API](https://openweathermap.org/api), including temperature, humidity, rainfall,
        and snowfall.
        - **Calendar Data:** Working day and school holiday information
        generated using Python libraries.

        ## Features Used in the Analysis
        - **Temporal Features:** Year, month, day, and hour.
        - **Weather Features:** Temperature, humidity, rainfall, and snowfall.
        - **Calendar Features:** School breaks and working days.
        - **Location Feature:** City name.
        - **Weather Categories:** Grouped weather conditions.

        **Target Variable:** `total_bike_usage`

        Represents the total number of bicycles used within each hour.
        ### What Can You Explore?

        - **Data Exploration:** Examine the dataset and identify usage patterns.
        - **Data Visualization:** Investigate how weather, time, and calendar
        factors affect bicycle demand.
        - **Model Insights:** Explore the performance and important features
        of the selected CatBoost model.
        - **Demand Prediction:** Estimate hourly bicycle usage based on
        selected conditions.

        ### Explore the App!
        Navigate through the pages to explore bike-sharing patterns, 
        gain insights into the factors influencing bicycle usage, 
        and predict hourly bike-sharing demand using our best-performing CatBoost model.
    """)

    st.caption(
    "Dataset source: Waldner et al. (2025), "
    "Data-Driven Insights into (E-)Bike-Sharing, Transportation. "
    "Licensed under CC BY-NC 4.0."
    )


with tab2:
    st.title("Data Visualization")
    st.markdown(
        "Explore bike-sharing patterns across different "
        "cities, hours, months, and temperature conditions."
    )

    st.divider()

    # Consistent city colors
    city_colors = {
        "Berlin": "#2878B5",
        "Dresden": "#E58B35",
        "Köln": "#329B75"
    }

    # First row
    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("Hourly Usage Patterns")
            st.caption("Average bike usage throughout the day")

            fig, ax = plt.subplots(figsize=(6, 3.5))

            sns.lineplot(
                data=df,
                x="hour",
                y="total_bike_usage",
                hue="city_name",
                palette=city_colors,
                estimator="mean",
                errorbar=None,
                linewidth=2,
                ax=ax
            )

            ax.set_xlabel("Hour of the Day")
            ax.set_ylabel("Average Bike Usage")
            ax.set_xticks(range(0, 24, 2))
            ax.grid(alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(title="City", frameon=False)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with col2:

        with st.container(border=True):
            st.subheader("Monthly Trends")
            st.caption("Average bike usage across the year (August–July)")

            # Define custom month order
            month_order = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7]

            month_labels = [
                "Aug", "Sep", "Oct", "Nov", "Dec",
                "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"
            ]

            month_map = dict(zip(month_order, month_labels))

            monthly_df = df.copy()

            # Convert month numbers into ordered month labels
            monthly_df["month_label"] = pd.Categorical(
                monthly_df["month"].map(month_map),
                categories=month_labels,
                ordered=True
            )

            fig, ax = plt.subplots(figsize=(6, 3.5))

            sns.lineplot(
                data=monthly_df,
                x="month_label",
                y="total_bike_usage",
                hue="city_name",
                palette=city_colors,
                estimator="mean",
                errorbar=None,
                marker="o",
                markersize=5,
                linewidth=2,
                ax=ax
            )

            ax.set_xlabel("Month")
            ax.set_ylabel("Average Bike Usage")

            # No manual xticks needed
            ax.grid(alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(title="City", frameon=False)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

 
    # Second row
    col3, col4 = st.columns(2, gap="large")

    with col3:
        with st.container(border=True):
            st.subheader("City Comparison")
            st.caption("Average hourly bike usage by city")

            city_usage = (
                df.groupby("city_name")["total_bike_usage"]
                .mean()
                .sort_values(ascending=True)
                .reset_index()
            )

            fig, ax = plt.subplots(figsize=(6, 3.5))

            sns.barplot(
                data=city_usage,
                x="total_bike_usage",
                y="city_name",
                hue="city_name",
                palette=city_colors,
                dodge=False,
                legend=False,
                ax=ax
            )

            ax.set_xlabel("Average Bike Usage")
            ax.set_ylabel("")
            ax.grid(axis="x", alpha=0.2)
            ax.grid(axis="y", visible=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with col4:
        with st.container(border=True):
            st.subheader("Temperature and Bike Usage")
            st.caption("Average bike usage across temperature ranges by city")

            df_temp = df.copy()

            df_temp["temp_group"] = pd.cut(
                df_temp["temp"],
                bins=15
            )

            temp_usage = (
                df_temp.groupby(
                    ["temp_group", "city_name"],
                    observed=True
                )["total_bike_usage"]
                .mean()
                .reset_index()
            )

            temp_usage["temp_mid"] = (
                temp_usage["temp_group"].apply(lambda x: x.mid)
            )

            fig, ax = plt.subplots(figsize=(6, 3.5))

            sns.lineplot(
                data=temp_usage,
                x="temp_mid",
                y="total_bike_usage",
                hue="city_name",
                palette=city_colors,
                marker="o",
                markersize=5,
                linewidth=2,
                ax=ax
            )

            ax.set_xlabel("Temperature")
            ax.set_ylabel("Average Bike Usage")
            ax.legend(title="City", frameon=False)

            ax.grid(alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# Third row
    col5, col6 = st.columns(2, gap="large")

    weather_city_usage = (
        df.groupby(["city_name", "weather_group"], observed=True)
        ["total_bike_usage"]
        .mean()
        .reset_index()
    )

    # Weather condition symbols
    weather_emojis = {
        "Clear": "☀   Clear",
        "Clouds": "☁   Clouds",
        "Rain": "☂   Rain",
        "Drizzle": "☔   Drizzle",
        "Snow": "❄   Snow",
        "Thunderstorm": "⚡   Thunderstorm",
        "Mist": "~   Mist",
        "Fog": "≋   Fog"
    }

    # Weather condition colors
    weather_colors = {
        "Clear": "#4CAF50",         # Green
        "Clouds": "#78909C",        # Blue Grey
        "Rain": "#3498DB",          # Blue
        "Drizzle": "#85C1E9",       # Light Blue
        "Snow": "#76D7EA",          # Turquoise
        "Thunderstorm": "#8E44AD", # Purple
        "Mist": "#B0BEC5",          # Light Grey
        "Fog": "#546E7A"            # Dark Blue Grey
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.barplot(
        data=weather_city_usage,
        x="city_name",
        y="total_bike_usage",
        hue="weather_group",
        palette=weather_colors,
        ax=ax
    )

    ax.set_title("Average Bike Usage by City and Weather Condition")
    ax.set_xlabel("")
    ax.set_ylabel("Average Bike Usage")

    # Remove legend and default city labels
    ax.get_legend().remove()
    ax.set_xticks([])

    weather_conditions = weather_city_usage["weather_group"].unique()
    cities = weather_city_usage["city_name"].unique()

    # Add weather condition labels
    for container, weather in zip(ax.containers, weather_conditions):
        for bar in container:
            x = bar.get_x() + bar.get_width() / 2

            ax.text(
                x, -0.10,
                weather_emojis.get(str(weather), str(weather)),
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                rotation=40,
                color=weather_colors.get(str(weather), "black")
            )

    # Add city names
    for i, city in enumerate(cities):
        ax.text(
            i, -0.27,
            city,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=11,
            fontweight="bold"
        )

    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    plt.subplots_adjust(bottom=0.40)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
  



with tab3:
    st.caption(
        "Enter all required input variables manually, "
        "including weather and calendar information, "
        "to predict hourly bike-sharing demand."
    )
# User inputs
    st.subheader("Categorical Features")
    col1, col2 = st.columns(2)

    with col1:
        city_name = st.selectbox(
            "City",
            categories["city_name"]
        )

        weather_group = st.selectbox(
            "Weather Group",
            categories["weather_group"]
        )

        
    with col2:
        working_day = st.selectbox(
                "Working Day",
                categories["working_day"]
            )
        school_break = st.selectbox(
            "School Break",
            categories["school_break"]
        )

        # month = st.selectbox(
        # "Month",
        # categories["month"]
        # )

        # hour = st.selectbox(
        # "Hour",
        # categories["hour"]
        # )

    st.divider()

    st.subheader("Time Features")

    col3, col4 = st.columns(2)

    with col3:
        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=1,
            step=1
        )

    with col4:
        hour = st.number_input(
            "Hour",
            min_value=0,
            max_value=23,
            value=12,
            step=1
        )

    st.divider()

    st.subheader("Weather Measurements")

    col5, col6 = st.columns(2)

    with col5:
        temp = st.number_input("Temperature", value=20.0)
        humidity = st.number_input("Humidity", value=50.0)
        wind_speed = st.number_input("Wind Speed", value=0.0)

    with col6:
        rain_1h = st.number_input("Rain (1h)", value=0.0)
        snow_1h = st.number_input("Snow (1h)", value=0.0)

    st.divider()

    # Prediction
    if st.button("Predict", type="primary", use_container_width=True):

        input_data = pd.DataFrame([{
            "city_name": city_name,
            "weather_group": weather_group,
            "working_day": working_day,
            "school_break": school_break,
            "month": month,
            "hour": hour,
            "temp": temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "rain_1h": rain_1h,
            "snow_1h": snow_1h
        }])

        prediction = model.predict(input_data)

        st.success("Prediction completed!")

        st.metric(
            label="Predicted value for hourly bicycle usage",
            value=f"{prediction[0]:,.2f}"
        )

        with st.expander("View Input Data"):
            st.dataframe(input_data)

with tab4:

    st.header("Automatic Bike Demand Prediction")

    st.caption(
        "Select a city, date, and time. Weather and calendar "
        "information will be retrieved automatically."
    )

    # User selects a city
    city = st.selectbox(
        "Select a city",
        ["Berlin", "Köln", "Dresden"],
        key="automatic_city"
    )

    # User selects a date
    selected_date = st.date_input(
        "Select a date",
        value=date.today(),
        key="automatic_date"
    )

    # User selects a time
    selected_time = st.time_input(
        "Select a time",
        value=time(12, 0),
        key="automatic_time"
    )

    # Combine date and time
    date_time = datetime.combine(
        selected_date,
        selected_time
    )

        # Prediction button
    if st.button(
        "Predict Bike Demand",
        type="primary",
        use_container_width=True,
        key="automatic_predict"
    ):

        # Function 1: Get city coordinates
        coordinates = get_city_coordinates(city)

        if coordinates is None:
            st.error("Coordinates not found for this city.")
            st.stop()

        lat, lon = coordinates

        # Function 2: Get weather forecast
        try:
            selected_forecast = get_weather(
                lat,
                lon,
                date_time,
                city
            )
        except requests.exceptions.RequestException:
            st.error(
                "The weather forecast could not be retrieved. "
                "Please try again later."
            )
            st.stop()

        # Check whether a matching forecast exists
        if selected_forecast.empty:
            st.warning(
                "No weather forecast is available for the "
                "selected date and time. Please choose a "
                "different date or time."
            )
            st.stop()

        # Function 3: Prepare predictors
        X_new = get_predictors(
            selected_forecast,
            date_time
        )

        # Make prediction with the CatBoost model
        prediction = model.predict(X_new)

        # Display prediction
        st.success("Prediction completed!")

        st.metric(
            label="Predicted hourly bike-sharing demand",
            value=f"{math.ceil(max(0, prediction[0])):,} trips"
        )

        # Show model input
        with st.expander("View Input Data"):
            st.dataframe(X_new)