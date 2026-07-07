# Importing required libraries
import requests

weather_code_map = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

def get_weather_description(code):
    return weather_code_map.get(code, "Unknown")

def get_current_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current_weather": True}
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