# Importing required libraries
import requests
import pandas as pd
from prophet import Prophet
import datetime

# A complete, reusable function: given a country's coordinates, return its seasonal peak/trough dates
def get_seasonal_pattern(lat, lon, start_date="2019-01-01", end_date="2023-12-31"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    weather_data = response.json()

    df = pd.DataFrame({
        "date": weather_data["daily"]["time"],
        "temperature": weather_data["daily"]["temperature_2m_mean"]
    })
    df["date"] = pd.to_datetime(df["date"])
    season_data = df.rename(columns={"date": "ds", "temperature": "y"})

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(season_data)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    yearly_component = forecast[["ds", "yearly"]].copy()
    yearly_component["day_of_year"] = yearly_component["ds"].dt.dayofyear
    seasonal_curve = yearly_component.groupby("day_of_year")["yearly"].mean()

    peak_day = seasonal_curve.idxmax()
    trough_day = seasonal_curve.idxmin()

    peak_date = datetime.datetime(2024, 1, 1) + datetime.timedelta(days=int(peak_day) - 1)
    trough_date = datetime.datetime(2024, 1, 1) + datetime.timedelta(days=int(trough_day) - 1)

    return {
        "warmest_point": peak_date.strftime("%B %d"),
        "coldest_point": trough_date.strftime("%B %d")
    }