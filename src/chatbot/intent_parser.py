# Simple keyword-based intent detection for the chatbot
def detect_intent(message):
    message = message.lower()

    weather_keywords = ["weather", "temperature now", "current condition", "raining", "sunny", "wind"]
    prediction_keywords = ["predict", "2050", "trend", "future", "forecast"]
    season_keywords = ["summer", "winter", "monsoon", "season", "when does"]
    risk_keywords = ["heatwave", "risk", "danger", "warning", "disaster", "flood", "hurricane"]

    if any(word in message for word in weather_keywords):
        return "weather"
    elif any(word in message for word in prediction_keywords):
        return "prediction"
    elif any(word in message for word in season_keywords):
        return "season"
    elif any(word in message for word in risk_keywords):
        return "risk"
    else:
        return "unknown"

# Detects if any known country name appears in the user's message
def extract_country(message, country_list):
    message_lower = message.lower()
    matches = [c for c in country_list if c.lower() in message_lower]

    if not matches:
        return None

    # Picking the longest match, in case of overlapping names (e.g. "Guinea" vs "Papua New Guinea")
    return max(matches, key=len)