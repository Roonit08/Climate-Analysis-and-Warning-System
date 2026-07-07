# Importing required libraries
import flet as ft
import json
import sys
import math
import datetime
import random
import joblib
import pandas as pd
import tensorflow as tf
from pathlib import Path

def main(page: ft.Page):
    page.title = "Terra Forecast"
    page.window.width = 1150
    page.window.height = 780
    page.window.min_width = 950
    page.window.min_height = 650
    page.padding = 0

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    with open(BASE_DIR / "data" / "country_coordinates.json") as f:
        country_coordinates = json.load(f)

    country_list = sorted(country_coordinates.keys())

    sys.path.append(str(BASE_DIR / "src" / "api"))
    sys.path.append(str(BASE_DIR / "src" / "disaster"))
    from weather_client import get_current_weather, get_weekly_forecast
    from season_timing import get_seasonal_pattern
    from risk_rules import assess_disaster_risk

    selected_country = {"name": "Nepal"}
    theme_state = {"dark": False}
    current_index = {"value": 0}

    lr_model = joblib.load(BASE_DIR / "saved_models" / "linear_regression_model.pkl")
    with open(BASE_DIR / "data" / "country_encoding_map.json") as f:
        country_encoding_map = json.load(f)

    historical_data = pd.read_csv(BASE_DIR / "data" / "model_ready_data.csv")

    heatwave_model = tf.keras.models.load_model(str(BASE_DIR / "saved_models" / "heatwave_nn_model.keras"))
    heatwave_scaler = joblib.load(BASE_DIR / "saved_models" / "heatwave_scaler.pkl")
    with open(BASE_DIR / "saved_models" / "heatwave_threshold.json") as f:
        heatwave_threshold = json.load(f)["threshold"]

    CLIMATE_FACTS = [
        "The last decade was the warmest on record globally.",
        "A rise of just 1.5°C can significantly increase heatwave frequency.",
        "Small island nations face some of the highest climate risks despite contributing the least emissions.",
        "Mountain regions like the Himalayas are warming faster than the global average.",
    ]

    def get_theme():
        if theme_state["dark"]:
            return {"bg": "#0F172A", "card": "#1E293B", "border": "#334155", "text": ft.colors.WHITE,
                    "subtext": ft.colors.GREY_400, "sky": ft.colors.LIGHT_BLUE_300, "sky_light": "#1E3A5F",
                    "green": ft.colors.GREEN_300, "green_light": "#1E3A2F"}
        else:
            return {"bg": ft.colors.GREY_50, "card": ft.colors.WHITE, "border": ft.colors.GREY_200, "text": ft.colors.BLACK,
                    "subtext": ft.colors.GREY_700, "sky": ft.colors.LIGHT_BLUE_600, "sky_light": ft.colors.LIGHT_BLUE_50,
                    "green": ft.colors.GREEN_600, "green_light": ft.colors.GREEN_50}

    def refresh_current_screen():
        screens = [build_dashboard, build_weather, build_predictions, build_risk_warnings, build_settings]
        content_area.content = screens[current_index["value"]]()
        page.update()

    def apply_theme():
        theme = get_theme()
        page.theme_mode = ft.ThemeMode.DARK if theme_state["dark"] else ft.ThemeMode.LIGHT
        nav_rail.bgcolor = theme["card"]
        nav_rail.indicator_color = theme["sky_light"]
        content_area.bgcolor = theme["bg"]
        refresh_current_screen()

    def get_weather_icon(condition):
        c = condition.lower()
        if "clear" in c:
            return ft.icons.WB_SUNNY, ft.colors.ORANGE_400
        elif "overcast" in c or "cloud" in c:
            return ft.icons.CLOUD, ft.colors.GREY_600
        elif "drizzle" in c or "rain" in c:
            return ft.icons.WATER_DROP, ft.colors.LIGHT_BLUE_700
        elif "snow" in c:
            return ft.icons.AC_UNIT, ft.colors.LIGHT_BLUE_300
        elif "thunder" in c:
            return ft.icons.THUNDERSTORM, ft.colors.DEEP_PURPLE_300
        elif "fog" in c:
            return ft.icons.CLOUD, ft.colors.GREY_500
        else:
            return ft.icons.WB_CLOUDY, ft.colors.GREY_500

    def get_condition_gradient(condition, is_day):
        c = condition.lower()
        if "thunder" in c:
            return ["#2C2C54", "#1B1B3A"]
        elif "rain" in c or "drizzle" in c:
            return ["#4A6FA5", "#2C3E50"] if is_day else ["#1F2C3A", "#0F172A"]
        elif "snow" in c:
            return ["#DCEEFB", "#F5FBFF"]
        elif "fog" in c:
            return ["#B0BEC5", "#90A4AE"]
        elif "cloud" in c or "overcast" in c:
            return ["#78909C", "#546E7A"] if is_day else ["#37474F", "#1C262B"]
        elif "clear" in c and is_day:
            return ["#4FC3F7", "#FFD54F"]
        else:
            return ["#1A237E", "#0D1B4C"]

    def get_weather_text_color(condition, is_day):
        c = condition.lower()
        if "snow" in c or "fog" in c:
            return ft.colors.BLACK87
        return ft.colors.WHITE

    def get_outlook_text(temp, condition):
        c = condition.lower()
        if "thunder" in c:
            return "Stormy conditions — best to stay indoors."
        elif "rain" in c or "drizzle" in c:
            return "Grab an umbrella — it's wet out there."
        elif "snow" in c:
            return "Bundle up — snowy conditions expected."
        elif "clear" in c and temp >= 25:
            return "Warm and sunny — a great day to be outside."
        elif "clear" in c:
            return "Clear skies and comfortable conditions."
        elif "cloud" in c or "overcast" in c:
            return "Overcast skies with mild conditions."
        else:
            return "Conditions look calm right now."

    def predict_temperature_trend(country_name):
        country_data = historical_data[historical_data["Country Name"] == country_name].sort_values("year")
        if country_data.empty or country_name not in country_encoding_map:
            return None
        country_code = country_encoding_map[country_name]
        history = country_data["temp_anomaly"].tolist()
        last_year = int(country_data["year"].max())
        results = {"years": [], "values": []}
        for _, row in country_data.iterrows():
            results["years"].append(int(row["year"]))
            results["values"].append(row["temp_anomaly"])
        for year in range(last_year + 1, 2051):
            lag_1 = history[-1]
            rolling_mean_5 = sum(history[-5:]) / len(history[-5:])
            decade = (year // 10) * 10
            input_row = pd.DataFrame([{"year": year, "lag_1": lag_1, "rolling_mean_5": rolling_mean_5,
                                        "decade": decade, "country_encoded": country_code}])
            pred = lr_model.predict(input_row)[0]
            results["years"].append(year)
            results["values"].append(pred)
            history.append(pred)
        return results

    def predict_heatwave_risk(country_name):
        country_data = historical_data[historical_data["Country Name"] == country_name].sort_values("year")
        if country_data.empty or country_name not in country_encoding_map:
            return None
        latest = country_data.iloc[-1]
        country_code = country_encoding_map[country_name]
        input_row = pd.DataFrame([{"year": int(latest["year"]), "lag_1": latest["lag_1"],
                                    "rolling_mean_5": latest["rolling_mean_5"], "decade": int(latest["decade"]),
                                    "country_encoded": country_code}])
        input_scaled = heatwave_scaler.transform(input_row)
        probability = heatwave_model.predict(input_scaled, verbose=0)[0][0]
        return {"probability": float(probability), "high_risk": bool(probability > heatwave_threshold),
                "reference_year": int(latest["year"])}

    def build_dashboard():
        theme = get_theme()
        search_query = {"text": ""}
        results_column = ft.Column(spacing=4)

        def select_country(name):
            def handler(e):
                selected_country["name"] = name
                content_area.content = build_dashboard()
                page.update()
            return handler

        def update_results():
            results_column.controls.clear()
            q = search_query["text"].strip().lower()
            if not q:
                return
            matches = [c for c in country_list if q in c.lower()][:8]
            if not matches:
                results_column.controls.append(ft.Text("No countries found", size=12, color=theme["subtext"]))
                return
            for c in matches:
                is_selected = c == selected_country["name"]
                results_column.controls.append(
                    ft.Container(
                        content=ft.Row(
                            [ft.Icon(ft.icons.CHECK_CIRCLE if is_selected else ft.icons.PLACE_OUTLINED,
                                    color=theme["green"] if is_selected else ft.colors.GREY_400, size=16),
                             ft.Text(c, size=14, weight=ft.FontWeight.BOLD if is_selected else ft.FontWeight.NORMAL, color=theme["text"])],
                            spacing=8),
                        padding=ft.padding.symmetric(horizontal=12, vertical=10), border_radius=8,
                        on_click=select_country(c), ink=True,
                        bgcolor=theme["sky_light"] if is_selected else theme["card"],
                    )
                )

        def search_changed(e):
            search_query["text"] = e.control.value
            update_results()
            page.update()

        search_field = ft.TextField(hint_text="Search for a country...", prefix_icon=ft.icons.SEARCH,
                                     on_change=search_changed, border_radius=10, filled=True,
                                     bgcolor=theme["card"], color=theme["text"])

        hero = ft.Container(
            content=ft.Column([ft.Text("Terra Forecast", size=32, weight=ft.FontWeight.BOLD, color=ft.colors.WHITE),
                                ft.Text("Climate Analysis & Warning System", size=14, color=ft.colors.WHITE)], spacing=4),
            gradient=ft.LinearGradient(begin=ft.alignment.top_left, end=ft.alignment.bottom_right,
                                        colors=[ft.colors.LIGHT_BLUE_600, ft.colors.GREEN_600]),
            padding=22, border_radius=14,
        )

        def stat_card(icon, label, value):
            return ft.Container(
                content=ft.Column([ft.Icon(icon, color=theme["sky"], size=22), ft.Text(value, size=18, weight=ft.FontWeight.BOLD, color=theme["text"]),
                                    ft.Text(label, size=11, color=theme["subtext"])], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=theme["card"], border_radius=10, border=ft.border.all(1, theme["border"]),
                padding=15, expand=True, alignment=ft.alignment.center,
            )

        stats_row = ft.Row([stat_card(ft.icons.PUBLIC, "Countries Tracked", "226"),
                             stat_card(ft.icons.CALENDAR_MONTH, "Years of Data", "51"),
                             stat_card(ft.icons.INSIGHTS, "Forecast Horizon", "2050")], spacing=12)

        country_name = selected_country["name"]
        country_hist = historical_data[historical_data["Country Name"] == country_name].sort_values("year")
        latest_val = country_hist["temp_anomaly"].iloc[-1] if not country_hist.empty else None
        latest_year = int(country_hist["year"].iloc[-1]) if not country_hist.empty else None

        trend_result = predict_temperature_trend(country_name)
        year_2050_val = trend_result["values"][-1] if trend_result else None

        if latest_val is not None and year_2050_val is not None:
            is_rising = year_2050_val > latest_val
            trend_icon = ft.icons.TRENDING_UP if is_rising else ft.icons.TRENDING_DOWN
            trend_color = ft.colors.ORANGE_700 if is_rising else theme["green"]
            trend_label = "Warming"
        else:
            trend_icon, trend_color, trend_label = ft.icons.HELP_OUTLINE, ft.colors.GREY_400, "Unknown"

        def mini_stat(icon, label, value, color):
            children = [ft.Icon(icon, color=color, size=20)]
            if value:
                children.append(ft.Text(value, size=16, weight=ft.FontWeight.BOLD, color=theme["text"]))
            children.append(ft.Text(label, size=11, color=theme["subtext"]))
            return ft.Column(children, spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER)

        selected_card = ft.Container(
            content=ft.Column([
                ft.Row([ft.Icon(ft.icons.PLACE, color=theme["sky"]), ft.Text(country_name, size=20, weight=ft.FontWeight.BOLD, color=theme["text"])], spacing=8),
                ft.Container(height=12),
                ft.Row([mini_stat(ft.icons.THERMOSTAT, f"Latest ({latest_year})" if latest_year else "Latest",
                                   f"{latest_val:.2f}°C" if latest_val is not None else "N/A", theme["sky"]),
                        ft.Container(width=25),
                        mini_stat(ft.icons.INSIGHTS, "Predicted 2050", f"{year_2050_val:.2f}°C" if year_2050_val is not None else "N/A", theme["green"]),
                        ft.Container(width=25),
                        mini_stat(trend_icon, trend_label, "", trend_color)], wrap=True),
            ]),
            bgcolor=theme["card"], border_radius=12, border=ft.border.all(1, theme["border"]), padding=18,
        )

        fact_card = ft.Container(
            content=ft.Row([ft.Icon(ft.icons.LIGHTBULB, color=ft.colors.AMBER_600, size=22),
                             ft.Text(random.choice(CLIMATE_FACTS), size=13, color=theme["text"], expand=True)], spacing=10),
            bgcolor=theme["sky_light"], border_radius=10, padding=14,
        )

        def go_to(index):
            def handler(e):
                current_index["value"] = index
                nav_rail.selected_index = index
                refresh_current_screen()
            return handler

        quick_actions = ft.Row([
            ft.ElevatedButton("View Weather", icon=ft.icons.WB_SUNNY, on_click=go_to(1), bgcolor=theme["sky_light"], color=theme["sky"]),
            ft.ElevatedButton("2050 Forecast", icon=ft.icons.SHOW_CHART, on_click=go_to(2), bgcolor=theme["green_light"], color=theme["green"]),
            ft.ElevatedButton("Check Risk", icon=ft.icons.WARNING_AMBER, on_click=go_to(3),
                               bgcolor=ft.colors.ORANGE_50 if not theme_state["dark"] else "#3D2A1A", color=ft.colors.ORANGE_700),
        ], wrap=True, spacing=10)

        update_results()

        content = ft.Column([
            hero, ft.Container(height=14), stats_row, ft.Container(height=14), fact_card, ft.Container(height=16),
            ft.Text("Select Your Country", size=17, weight=ft.FontWeight.W_600, color=theme["text"]),
            ft.Container(height=6), search_field, results_column, ft.Container(height=10),
            selected_card, ft.Container(height=16), quick_actions,
        ], scroll=ft.ScrollMode.AUTO)

        return ft.Container(content=content, padding=30, expand=True, bgcolor=theme["bg"])

    def build_weather():
        theme = get_theme()
        country_name = selected_country["name"]
        lat, lon = country_coordinates[country_name]

        def refresh_clicked(e):
            content_area.content = build_weather()
            page.update()

        try:
            weather = get_current_weather(lat, lon)
            icon, icon_color = get_weather_icon(weather["condition"])
            outlook = get_outlook_text(weather["temperature"], weather["condition"])
            wind_deg = weather.get("winddirection", 0)
            grad_colors = get_condition_gradient(weather["condition"], weather["is_day"])
            text_color = get_weather_text_color(weather["condition"], weather["is_day"])
            sub_color = ft.colors.with_opacity(0.85, text_color)

            current_section = ft.Container(
                content=ft.Column([
                    ft.Row([ft.Icon(icon, size=60, color=text_color),
                            ft.Column([ft.Text(f"{weather['temperature']}°C", size=44, weight=ft.FontWeight.BOLD, color=text_color),
                                       ft.Text(weather["condition"], size=16, color=sub_color)], spacing=0)], spacing=18),
                    ft.Container(height=8),
                    ft.Text(outlook, size=14, italic=True, color=sub_color),
                    ft.Container(height=18),
                    ft.Row([
                        ft.Column([ft.Row([ft.Container(content=ft.Icon(ft.icons.NAVIGATION, color=text_color, size=20),
                                                         rotate=ft.transform.Rotate(angle=math.radians(wind_deg))),
                                            ft.Text(f"{weather['windspeed']} km/h", size=15, weight=ft.FontWeight.BOLD, color=text_color)], spacing=8),
                                   ft.Text("Wind", size=11, color=sub_color)], spacing=4),
                        ft.Container(width=45),
                        ft.Column([ft.Icon(ft.icons.WB_TWILIGHT if weather["is_day"] else ft.icons.NIGHTLIGHT_ROUND, color=text_color, size=20),
                                   ft.Text("Day" if weather["is_day"] else "Night", size=15, weight=ft.FontWeight.BOLD, color=text_color)], spacing=4),
                    ]),
                ]),
                gradient=ft.LinearGradient(begin=ft.alignment.top_left, end=ft.alignment.bottom_right, colors=grad_colors),
                border_radius=16, padding=26,
            )
        except Exception as ex:
            current_section = ft.Text(f"Could not load weather data: {ex}", color=ft.colors.RED)
            grad_colors = None

        try:
            forecast = get_weekly_forecast(lat, lon)
            day_cards = []
            for day in forecast:
                d = datetime.datetime.strptime(day["date"], "%Y-%m-%d")
                day_icon, day_color = get_weather_icon(day["condition"])
                day_cards.append(ft.Container(
                    content=ft.Column([
                        ft.Text(d.strftime("%a"), size=12, weight=ft.FontWeight.BOLD, color=theme["subtext"]),
                        ft.Icon(day_icon, color=day_color, size=26),
                        ft.Text(f"{day['max_temp']:.0f}°", size=14, weight=ft.FontWeight.BOLD, color=theme["text"]),
                        ft.Text(f"{day['min_temp']:.0f}°", size=12, color=theme["subtext"]),
                    ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    bgcolor=theme["card"], border_radius=10, border=ft.border.all(1, theme["border"]),
                    padding=12, expand=True, alignment=ft.alignment.center,
                ))
            forecast_section = ft.Column([
                ft.Text("7-Day Forecast", size=16, weight=ft.FontWeight.W_600, color=theme["text"]),
                ft.Container(height=10),
                ft.Row(day_cards, spacing=10),
            ])
        except Exception:
            forecast_section = ft.Text("7-day forecast temporarily unavailable", color=theme["subtext"])

        content = ft.Column([
            ft.Row([ft.Text(country_name, size=26, weight=ft.FontWeight.BOLD, color=theme["text"]),
                    ft.IconButton(icon=ft.icons.REFRESH, on_click=refresh_clicked, icon_color=theme["sky"])]),
            ft.Container(height=8), current_section, ft.Container(height=22), forecast_section,
        ], spacing=5, scroll=ft.ScrollMode.AUTO)

        return ft.Container(content=content, padding=30, expand=True, bgcolor=theme["bg"])

    def build_predictions():
        theme = get_theme()
        country_name = selected_country["name"]
        result = predict_temperature_trend(country_name)

        if result is None:
            return ft.Container(content=ft.Text(f"No prediction data available for {country_name}", color=ft.colors.RED), padding=30)

        data_points = [ft.LineChartDataPoint(x, y) for x, y in zip(result["years"], result["values"])]
        min_year, max_year = min(result["years"]), max(result["years"])
        min_val, max_val = min(result["values"]), max(result["values"])

        bottom_labels = []
        start_decade = (min_year // 10) * 10
        for year in range(start_decade, max_year + 1, 10):
            if year >= min_year:
                bottom_labels.append(ft.ChartAxisLabel(value=year, label=ft.Text(str(year), size=11, color=theme["subtext"])))

        y_interval = round((max_val - min_val) / 5, 1) or 0.5

        chart = ft.LineChart(
            data_series=[ft.LineChartData(data_points=data_points, stroke_width=3, color=theme["sky"], curved=True, stroke_cap_round=True)],
            border=ft.border.all(1, theme["border"]),
            horizontal_grid_lines=ft.ChartGridLines(interval=y_interval, color=theme["border"], width=1),
            vertical_grid_lines=ft.ChartGridLines(interval=10, color=theme["border"], width=1),
            left_axis=ft.ChartAxis(title=ft.Text("°C", size=12, color=theme["subtext"]), title_size=20, labels_size=45),
            bottom_axis=ft.ChartAxis(title=ft.Text("Year", size=12, color=theme["subtext"]), title_size=20, labels_size=28, labels=bottom_labels),
            min_y=min_val - 0.3, max_y=max_val + 0.3, min_x=min_year, max_x=max_year,
            tooltip_bgcolor=theme["card"], expand=True,
        )

        year_2050_value = result["values"][-1]
        is_rising = result["values"][-1] > result["values"][0]
        badge_color = ft.colors.ORANGE_700 if is_rising else theme["green"]
        badge_bg = (ft.colors.ORANGE_50 if not theme_state["dark"] else "#3D2A1A") if is_rising else theme["green_light"]
        badge = ft.Container(
            content=ft.Row([ft.Icon(ft.icons.TRENDING_UP if is_rising else ft.icons.TRENDING_DOWN, color=badge_color, size=18),
                             ft.Text("Warming Trend" if is_rising else "Cooling Trend", size=13, weight=ft.FontWeight.BOLD, color=badge_color)], spacing=6),
            bgcolor=badge_bg, border_radius=20, padding=ft.padding.symmetric(horizontal=14, vertical=6),
        )

        # Year search feature
        year_result_text = ft.Text("", size=15, weight=ft.FontWeight.BOLD, color=theme["text"])
        year_input = ft.TextField(label="Search a year (e.g. 2030)", width=220, border_radius=8)

        def check_year(e):
            raw = (year_input.value or "").strip()
            if not raw.isdigit():
                year_result_text.value = "Please enter a valid year."
                page.update()
                return
            year_val = int(raw)
            if year_val < min_year or year_val > 2050:
                year_result_text.value = f"Please enter a year between {min_year} and 2050."
            elif year_val in result["years"]:
                idx = result["years"].index(year_val)
                val = result["values"][idx]
                label = "recorded" if year_val <= 2021 else "predicted"
                year_result_text.value = f"{label.capitalize()} change in {year_val}: {val:.2f}°C"
            else:
                year_result_text.value = "No data available for that year."
            page.update()

        year_search_card = ft.Container(
            content=ft.Column([
                ft.Text("Check a Specific Year", size=16, weight=ft.FontWeight.W_600, color=theme["text"]),
                ft.Container(height=10),
                ft.Row([year_input, ft.ElevatedButton("Check", on_click=check_year, bgcolor=theme["sky_light"], color=theme["sky"])], spacing=10),
                ft.Container(height=8),
                year_result_text,
            ]),
            bgcolor=theme["card"], border_radius=10, border=ft.border.all(1, theme["border"]), padding=20,
        )

        lat, lon = country_coordinates[country_name]
        try:
            seasonal_info = get_seasonal_pattern(lat, lon)
            season_section = ft.Container(
                content=ft.Column([
                    ft.Text("Seasonal Pattern", size=18, weight=ft.FontWeight.W_600, color=theme["text"]),
                    ft.Container(height=10),
                    ft.Row([
                        ft.Column([ft.Text("🔥 Summer", size=12, color=theme["subtext"]),
                                   ft.Text(seasonal_info["summer_range"], size=16, weight=ft.FontWeight.BOLD, color=ft.colors.ORANGE_700),
                                   ft.Text(f"Peak: {seasonal_info['warmest_point']}", size=11, color=theme["subtext"])]),
                        ft.Container(width=50),
                        ft.Column([ft.Text("❄️ Winter", size=12, color=theme["subtext"]),
                                   ft.Text(seasonal_info["winter_range"], size=16, weight=ft.FontWeight.BOLD, color=theme["sky"]),
                                   ft.Text(f"Peak: {seasonal_info['coldest_point']}", size=11, color=theme["subtext"])]),
                    ], wrap=True),
                ]),
                bgcolor=theme["card"], border_radius=10, border=ft.border.all(1, theme["border"]), padding=20,
            )
        except Exception:
            season_section = ft.Text("Seasonal data temporarily unavailable", color=theme["subtext"])

        content = ft.Column([
            ft.Row([ft.Text(f"{country_name} — Temperature Trend to 2050", size=22, weight=ft.FontWeight.W_600, color=theme["text"]), badge], spacing=15, wrap=True),
            ft.Text(f"Predicted change by 2050: {year_2050_value:.2f}°C", size=14, color=theme["subtext"]),
            ft.Container(height=15),
            ft.Container(content=chart, height=320, padding=ft.padding.only(top=15, right=20, bottom=5),
                         bgcolor=theme["card"], border_radius=10, border=ft.border.all(1, theme["border"])),
            ft.Container(height=18), year_search_card, ft.Container(height=18), season_section,
        ], spacing=5, scroll=ft.ScrollMode.AUTO)

        return ft.Container(content=content, padding=30, expand=True, bgcolor=theme["bg"])

    def build_risk_warnings():
        theme = get_theme()
        country_name = selected_country["name"]
        heatwave_result = predict_heatwave_risk(country_name)

        if heatwave_result is None:
            heatwave_card = ft.Text(f"No heatwave data available for {country_name}", color=ft.colors.RED)
        else:
            risk_level_text = "Elevated Risk" if heatwave_result["high_risk"] else "Normal Conditions"
            risk_color = ft.colors.RED_700 if heatwave_result["high_risk"] else theme["green"]
            if heatwave_result["high_risk"]:
                risk_bg = ft.colors.RED_50 if not theme_state["dark"] else "#3D1A1A"
            else:
                risk_bg = theme["green_light"]

            heatwave_card = ft.Container(
                content=ft.Column([
                    ft.Row([ft.Icon(ft.icons.WHATSHOT, color=risk_color, size=28), ft.Text("Heatwave Risk", size=18, weight=ft.FontWeight.W_600, color=theme["text"])]),
                    ft.Text(f"Based on climate trend data through {heatwave_result['reference_year']}", size=11, color=theme["subtext"], italic=True),
                    ft.Container(height=10),
                    ft.Text(risk_level_text, size=22, weight=ft.FontWeight.BOLD, color=risk_color),
                    ft.Container(height=8),
                    ft.ProgressBar(value=heatwave_result["probability"], color=risk_color, bgcolor=theme["border"], height=10, border_radius=5),
                    ft.Container(height=6),
                    ft.Text(f"Model confidence: {heatwave_result['probability']*100:.0f}%", size=13, color=theme["subtext"]),
                ]),
                bgcolor=risk_bg, border_radius=10, border=ft.border.all(1, risk_color), padding=20,
            )

        lat, lon = country_coordinates[country_name]
        try:
            current_weather = get_current_weather(lat, lon)
            disaster_flags = assess_disaster_risk(current_weather)
            disaster_items = []
            for flag in disaster_flags:
                is_safe = flag == "No significant weather-based risk detected"
                disaster_items.append(ft.Row([ft.Icon(ft.icons.CHECK_CIRCLE if is_safe else ft.icons.WARNING,
                                                       color=theme["green"] if is_safe else ft.colors.ORANGE_700, size=18),
                                               ft.Text(flag, size=14, color=theme["text"])]))
            disaster_card = ft.Container(
                content=ft.Column([
                    ft.Text("Live Disaster Risk Monitor", size=18, weight=ft.FontWeight.W_600, color=theme["text"]),
                    ft.Text("Based on real-time current weather conditions", size=11, color=theme["subtext"], italic=True),
                    ft.Container(height=10), *disaster_items,
                ]),
                bgcolor=theme["card"], border_radius=10, border=ft.border.all(1, theme["border"]), padding=20,
            )
        except Exception:
            disaster_card = ft.Text("Live disaster data temporarily unavailable", color=theme["subtext"])

        country_hist = historical_data[historical_data["Country Name"] == country_name]
        if not country_hist.empty:
            max_row = country_hist.loc[country_hist["temp_anomaly"].idxmax()]
            extreme_card = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.HISTORY, color=theme["sky"], size=24),
                    ft.Column([ft.Text("Historical Extreme", size=13, color=theme["subtext"]),
                               ft.Text(f"Highest anomaly recorded: {max_row['temp_anomaly']:.2f}°C in {int(max_row['year'])}",
                                       size=14, weight=ft.FontWeight.BOLD, color=theme["text"])], spacing=2),
                ], spacing=12),
                bgcolor=theme["sky_light"], border_radius=10, padding=16,
            )
        else:
            extreme_card = ft.Container()

        safety_tips = ft.Container(
            content=ft.Column([
                ft.Text("Safety Tips", size=16, weight=ft.FontWeight.W_600, color=theme["text"]),
                ft.Container(height=8),
                ft.Text("• Stay hydrated and avoid outdoor activity during peak heat hours", size=13, color=theme["subtext"]),
                ft.Text("• Keep an emergency kit ready during high-risk weather periods", size=13, color=theme["subtext"]),
                ft.Text("• Monitor local alerts if wind or rainfall risk is flagged", size=13, color=theme["subtext"]),
            ], spacing=6),
            bgcolor=theme["card"], border_radius=10, border=ft.border.all(1, theme["border"]), padding=20,
        )

        content = ft.Column([
            ft.Text(f"{country_name} — Risk Warnings", size=22, weight=ft.FontWeight.W_600, color=theme["text"]),
            ft.Container(height=15), heatwave_card, ft.Container(height=18),
            disaster_card, ft.Container(height=18), extreme_card, ft.Container(height=18), safety_tips,
        ], spacing=5, scroll=ft.ScrollMode.AUTO)

        return ft.Container(content=content, padding=30, expand=True, bgcolor=theme["bg"])

    def build_settings():
        theme = get_theme()

        def theme_toggled(e):
            theme_state["dark"] = e.control.value
            apply_theme()

        content = ft.Column([
            ft.Text("Settings", size=22, weight=ft.FontWeight.W_600, color=theme["text"]),
            ft.Container(height=20),
            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.DARK_MODE, color=theme["sky"]),
                    ft.Column([ft.Text("Dark Mode", size=15, weight=ft.FontWeight.BOLD, color=theme["text"]),
                               ft.Text("Switch between light and dark theme", size=12, color=theme["subtext"])], spacing=2, expand=True),
                    ft.Switch(value=theme_state["dark"], on_change=theme_toggled, active_color=theme["sky"]),
                ], spacing=15),
                bgcolor=theme["card"], border_radius=10, border=ft.border.all(1, theme["border"]), padding=20,
            ),
            ft.Container(height=20),
            ft.Container(
                content=ft.Column([
                    ft.Text("About Terra Forecast", size=15, weight=ft.FontWeight.BOLD, color=theme["text"]),
                    ft.Container(height=8),
                    ft.Text("A climate analysis and warning system covering 226 countries, powered by machine learning and live weather data.", size=13, color=theme["subtext"]),
                    ft.Container(height=10),
                    ft.Text("Built with Python, Flet, scikit-learn, XGBoost, Prophet, and TensorFlow.", size=12, color=theme["subtext"]),
                ]),
                bgcolor=theme["card"], border_radius=10, border=ft.border.all(1, theme["border"]), padding=20,
            ),
        ], scroll=ft.ScrollMode.AUTO)

        return ft.Container(content=content, padding=30, expand=True, bgcolor=theme["bg"])

    content_area = ft.Container(content=None, expand=True, padding=0)

    def nav_changed(e):
        current_index["value"] = e.control.selected_index
        refresh_current_screen()

    nav_rail = ft.NavigationRail(
        selected_index=0, label_type=ft.NavigationRailLabelType.ALL, min_width=100, min_extended_width=200,
        bgcolor=ft.colors.WHITE, indicator_color=ft.colors.LIGHT_BLUE_50,
        destinations=[
            ft.NavigationRailDestination(icon=ft.icons.DASHBOARD_OUTLINED, selected_icon=ft.icons.DASHBOARD, label="Dashboard"),
            ft.NavigationRailDestination(icon=ft.icons.WB_SUNNY_OUTLINED, selected_icon=ft.icons.WB_SUNNY, label="Weather"),
            ft.NavigationRailDestination(icon=ft.icons.SHOW_CHART_OUTLINED, selected_icon=ft.icons.SHOW_CHART, label="Predictions"),
            ft.NavigationRailDestination(icon=ft.icons.WARNING_AMBER_OUTLINED, selected_icon=ft.icons.WARNING_AMBER, label="Risk"),
            ft.NavigationRailDestination(icon=ft.icons.SETTINGS_OUTLINED, selected_icon=ft.icons.SETTINGS, label="Settings"),
        ],
        on_change=nav_changed,
    )

    page.add(ft.Row([nav_rail, ft.VerticalDivider(width=1), content_area], expand=True))
    apply_theme()

ft.app(target=main)