# Importing required libraries
import requests
import pandas as pd
from prophet import Prophet
import datetime

def _day_to_month_range(start_day, end_day):
    base = datetime.datetime(2024, 1, 1)
    start_date = base + datetime.timedelta(days=int(start_day) - 1)
    end_date = base + datetime.timedelta(days=int(end_day) - 1)
    if start_date.strftime("%B") == end_date.strftime("%B"):
        return start_date.strftime("%B")
    return f"{start_date.strftime('%B')} – {end_date.strftime('%B')}"

# Expands outward from a peak/trough day to find where the season range starts or ends
def _expand_range(curve, center_day, threshold, direction, max_days=150):
    day = center_day
    count = 0
    while count < max_days:
        next_day = day + direction
        if next_day > 365:
            next_day = 1
        elif next_day < 1:
            next_day = 365
        if next_day not in curve.index or curve[next_day] < threshold:
            break
        day = next_day
        count += 1
    return day

def get_seasonal_pattern(lat, lon, start_date="2019-01-01", end_date="2023-12-31"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": "temperature_2m_mean", "timezone": "auto"
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

    peak_day = int(seasonal_curve.idxmax())
    trough_day = int(seasonal_curve.idxmin())
    peak_value = seasonal_curve.max()
    trough_value = seasonal_curve.min()
    amplitude = peak_value - trough_value

    warm_threshold = peak_value - 0.3 * amplitude
    cold_threshold = trough_value + 0.3 * amplitude

    summer_end = _expand_range(seasonal_curve, peak_day, warm_threshold, 1)
    summer_start = _expand_range(seasonal_curve, peak_day, warm_threshold, -1)
    winter_end = _expand_range(seasonal_curve, trough_day, cold_threshold, 1)
    winter_start = _expand_range(seasonal_curve, trough_day, cold_threshold, -1)

    base = datetime.datetime(2024, 1, 1)
    peak_date = base + datetime.timedelta(days=peak_day - 1)
    trough_date = base + datetime.timedelta(days=trough_day - 1)

    return {
        "warmest_point": peak_date.strftime("%B %d"),
        "coldest_point": trough_date.strftime("%B %d"),
        "summer_range": _day_to_month_range(summer_start, summer_end),
        "winter_range": _day_to_month_range(winter_start, winter_end),
    }