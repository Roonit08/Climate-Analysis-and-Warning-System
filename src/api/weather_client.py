# Importing required libraries
import requests
import json

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
        "condition": get_weather_description(current["weathercode"]),
        "is_day": bool(current["is_day"])
    }