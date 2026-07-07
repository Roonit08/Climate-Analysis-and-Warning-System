# Simple rule-based disaster risk assessment using live weather data
def assess_disaster_risk(weather_info):
    risk_flags = []

    if weather_info["windspeed"] > 60:
        risk_flags.append("High wind risk (possible storm/hurricane conditions)")

    if weather_info["condition"] in ["Heavy rain", "Violent rain showers", "Thunderstorm with heavy hail"]:
        risk_flags.append("Heavy rainfall risk (possible flooding)")

    if not risk_flags:
        risk_flags.append("No significant weather-based risk detected")

    return risk_flags

# Estimating rough risk percentages for different disaster types based on current live weather
# These are simple rule-based estimates, not machine-learning predictions
def estimate_risk_percentages(weather_info):
    windspeed = weather_info.get("windspeed", 0)
    condition = weather_info.get("condition", "").lower()
    temperature = weather_info.get("temperature", 0)

    storm_pct = min(100, round((windspeed / 70) * 100))

    if "heavy rain" in condition or "violent" in condition or "thunderstorm" in condition:
        flood_pct = 80
    elif "moderate rain" in condition or "moderate drizzle" in condition:
        flood_pct = 50
    elif "rain" in condition or "drizzle" in condition:
        flood_pct = 25
    else:
        flood_pct = 5

    if temperature >= 38:
        heat_pct = 85
    elif temperature >= 33:
        heat_pct = 55
    elif temperature >= 28:
        heat_pct = 25
    else:
        heat_pct = 5

    return {"storm_pct": storm_pct, "flood_pct": flood_pct, "heat_pct": heat_pct}