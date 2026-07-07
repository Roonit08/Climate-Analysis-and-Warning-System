# Importing required libraries
import requests
import datetime

# Mapping Open-Meteo's WMO weather codes to human-readable conditions
weather_code_map = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

def get_weather_description(code):
    return weather_code_map.get(code, "Unknown")

# A single reusable function: give it coordinates, get back clean current weather info
def get_current_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True
    }
    response = requests.get(url, params=params)
    data = response.json()

    current = data["current_weather"]
    return {
        "temperature": current["temperature"],
        "windspeed": current["windspeed"],
        "winddirection": current["winddirection"],
        "condition": get_weather_description(current["weathercode"]),
        "is_day": bool(current["is_day"])
    }

# Getting a 7-day forecast (max/min temp and condition per day)
def get_weekly_forecast(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,weathercode",
        "forecast_days": 7,
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    data = response.json()
    daily = data["daily"]

    forecast = []
    for i in range(len(daily["time"])):
        forecast.append({
            "date": daily["time"][i],
            "max_temp": daily["temperature_2m_max"][i],
            "min_temp": daily["temperature_2m_min"][i],
            "condition": get_weather_description(daily["weathercode"][i]),
        })
    return forecast

# Finds the index in a list of hourly timestamps closest to right now
def _closest_hour_index(time_list):
    now = datetime.datetime.now()
    closest_index = 0
    smallest_diff = None
    for i, t in enumerate(time_list):
        t_parsed = datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M")
        diff = abs((t_parsed - now).total_seconds())
        if smallest_diff is None or diff < smallest_diff:
            smallest_diff = diff
            closest_index = i
    return closest_index

# Getting humidity and rain probability for right now, using hourly data
def get_extended_conditions(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "relativehumidity_2m,precipitation_probability",
        "forecast_days": 1,
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    data = response.json()
    now_index = _closest_hour_index(data["hourly"]["time"])
    return {
        "humidity": data["hourly"]["relativehumidity_2m"][now_index],
        "rain_chance": data["hourly"]["precipitation_probability"][now_index],
    }