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
