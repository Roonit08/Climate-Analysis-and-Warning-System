# Importing required libraries
import flet as ft
import json
import sys
import joblib
import pandas as pd
import tensorflow as tf
from pathlib import Path

def main(page: ft.Page):
    page.title = "Terra Forecast"
    page.window.width = 1100
    page.window.height = 750
    page.window.min_width = 900
    page.window.min_height = 600
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    page.bgcolor = ft.colors.GREY_50

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    with open(BASE_DIR / "data" / "country_coordinates.json") as f:
        country_coordinates = json.load(f)

    country_list = sorted(country_coordinates.keys())

    sys.path.append(str(BASE_DIR / "src" / "api"))
    sys.path.append(str(BASE_DIR / "src" / "disaster"))
    from weather_client import get_current_weather
    from season_timing import get_seasonal_pattern
    from risk_rules import assess_disaster_risk

    selected_country = {"name": "Nepal"}

    lr_model = joblib.load(BASE_DIR / "saved_models" / "linear_regression_model.pkl")
    with open(BASE_DIR / "data" / "country_encoding_map.json") as f:
        country_encoding_map = json.load(f)

    historical_data = pd.read_csv(BASE_DIR / "data" / "model_ready_data.csv")

    heatwave_model = tf.keras.models.load_model(str(BASE_DIR / "saved_models" / "heatwave_nn_model.keras"))
    heatwave_scaler = joblib.load(BASE_DIR / "saved_models" / "heatwave_scaler.pkl")
    with open(BASE_DIR / "saved_models" / "heatwave_threshold.json") as f:
        heatwave_threshold = json.load(f)["threshold"]

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

            input_row = pd.DataFrame([{
                "year": year,
                "lag_1": lag_1,
                "rolling_mean_5": rolling_mean_5,
                "decade": decade,
                "country_encoded": country_code
            }])

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

        input_row = pd.DataFrame([{
            "year": int(latest["year"]),
            "lag_1": latest["lag_1"],
            "rolling_mean_5": latest["rolling_mean_5"],
            "decade": int(latest["decade"]),
            "country_encoded": country_code
        }])

        input_scaled = heatwave_scaler.transform(input_row)
        probability = heatwave_model.predict(input_scaled, verbose=0)[0][0]
        is_high_risk = probability > heatwave_threshold

        return {"probability": float(probability), "high_risk": bool(is_high_risk)}

    def build_dashboard():
        def country_changed(e):
            selected_country["name"] = e.control.value
            page.update()

        country_dropdown = ft.Dropdown(
            label="Select a Country",
            width=320,
            content_padding=ft.padding.only(left=12, right=12, top=8, bottom=8),
            value=selected_country["name"],
            options=[ft.dropdown.Option(c) for c in country_list],
            on_change=country_changed,
        )

        return ft.Column(
            [
                ft.Text("Dashboard", size=28, weight=ft.FontWeight.BOLD),
                ft.Text("Choose a country to explore its climate data", size=14, color=ft.colors.GREY_700),
                ft.Container(height=20),
                country_dropdown,
            ],
            spacing=10,
        )

    def build_weather():
        country_name = selected_country["name"]
        lat, lon = country_coordinates[country_name]

        def refresh_clicked(e):
            content_area.content = build_weather()
            page.update()

        try:
            weather = get_current_weather(lat, lon)
            weather_display = ft.Column(
                [
                    ft.Row([
                        ft.Text(country_name, size=28, weight=ft.FontWeight.BOLD),
                        ft.IconButton(icon=ft.icons.REFRESH, on_click=refresh_clicked),
                    ]),
                    ft.Container(height=10),
                    ft.Text(f"{weather['temperature']}°C", size=48, weight=ft.FontWeight.BOLD),
                    ft.Text(weather["condition"], size=20, color=ft.colors.GREY_700),
                    ft.Container(height=20),
                    ft.Row(
                        [
                            ft.Column([ft.Text("Wind Speed", size=12, color=ft.colors.GREY_600), ft.Text(f"{weather['windspeed']} km/h", size=16, weight=ft.FontWeight.BOLD)]),
                            ft.Container(width=40),
                            ft.Column([ft.Text("Time of Day", size=12, color=ft.colors.GREY_600), ft.Text("Day" if weather["is_day"] else "Night", size=16, weight=ft.FontWeight.BOLD)]),
                        ]
                    ),
                ],
                spacing=5,
            )
        except Exception as ex:
            weather_display = ft.Text(f"Could not load weather data: {ex}", color=ft.colors.RED)

        return weather_display

    def build_predictions():
        country_name = selected_country["name"]
        result = predict_temperature_trend(country_name)

        if result is None:
            return ft.Text(f"No prediction data available for {country_name}", color=ft.colors.RED)

        data_points = [
            ft.LineChartDataPoint(x, y) for x, y in zip(result["years"], result["values"])
        ]

        min_year = min(result["years"])
        max_year = max(result["years"])
        min_val = min(result["values"])
        max_val = max(result["values"])

        bottom_labels = []
        start_decade = (min_year // 10) * 10
        for year in range(start_decade, max_year + 1, 10):
            if year >= min_year:
                bottom_labels.append(ft.ChartAxisLabel(value=year, label=ft.Text(str(year), size=11, color=ft.colors.GREY_600)))

        value_range = max_val - min_val
        y_interval = round(value_range / 5, 1)
        if y_interval == 0:
            y_interval = 0.5

        chart = ft.LineChart(
            data_series=[
                ft.LineChartData(
                    data_points=data_points,
                    stroke_width=3,
                    color=ft.colors.INDIGO_400,
                    curved=True,
                    stroke_cap_round=True,
                )
            ],
            border=ft.border.all(1, ft.colors.GREY_300),
            horizontal_grid_lines=ft.ChartGridLines(interval=y_interval, color=ft.colors.GREY_200, width=1),
            vertical_grid_lines=ft.ChartGridLines(interval=10, color=ft.colors.GREY_200, width=1),
            left_axis=ft.ChartAxis(
                title=ft.Text("°C", size=12, color=ft.colors.GREY_600),
                title_size=20,
                labels_size=45,
            ),
            bottom_axis=ft.ChartAxis(
                title=ft.Text("Year", size=12, color=ft.colors.GREY_600),
                title_size=20,
                labels_size=28,
                labels=bottom_labels,
            ),
            min_y=min_val - 0.3,
            max_y=max_val + 0.3,
            min_x=min_year,
            max_x=max_year,
            tooltip_bgcolor=ft.colors.WHITE,
            expand=True,
        )

        year_2050_value = result["values"][-1]

        lat, lon = country_coordinates[country_name]
        try:
            seasonal_info = get_seasonal_pattern(lat, lon)
            season_section = ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Seasonal Pattern", size=18, weight=ft.FontWeight.W_600),
                        ft.Container(height=10),
                        ft.Row(
                            [
                                ft.Column(
                                    [
                                        ft.Text("Warmest Point", size=12, color=ft.colors.GREY_600),
                                        ft.Text(seasonal_info["warmest_point"], size=18, weight=ft.FontWeight.BOLD, color=ft.colors.ORANGE_700),
                                    ]
                                ),
                                ft.Container(width=60),
                                ft.Column(
                                    [
                                        ft.Text("Coldest Point", size=12, color=ft.colors.GREY_600),
                                        ft.Text(seasonal_info["coldest_point"], size=18, weight=ft.FontWeight.BOLD, color=ft.colors.BLUE_700),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
                bgcolor=ft.colors.WHITE,
                border_radius=10,
                border=ft.border.all(1, ft.colors.GREY_200),
                padding=20,
            )
        except Exception:
            season_section = ft.Text("Seasonal data temporarily unavailable", color=ft.colors.GREY_500)

        return ft.Column(
            [
                ft.Text(f"{country_name} — Temperature Trend to 2050", size=22, weight=ft.FontWeight.W_600),
                ft.Text(f"Predicted change by 2050: {year_2050_value:.2f}°C", size=14, color=ft.colors.GREY_700),
                ft.Container(height=15),
                ft.Container(
                    content=chart,
                    height=320,
                    padding=ft.padding.only(top=15, right=20, bottom=5),
                    bgcolor=ft.colors.WHITE,
                    border_radius=10,
                    border=ft.border.all(1, ft.colors.GREY_200),
                ),
                ft.Container(height=20),
                season_section,
            ],
            spacing=5,
            scroll=ft.ScrollMode.AUTO,
        )

    def build_risk_warnings():
        country_name = selected_country["name"]

        heatwave_result = predict_heatwave_risk(country_name)

        if heatwave_result is None:
            heatwave_card = ft.Text(f"No heatwave data available for {country_name}", color=ft.colors.RED)
        else:
            risk_level_text = "Elevated Risk" if heatwave_result["high_risk"] else "Normal Conditions"
            risk_color = ft.colors.RED_700 if heatwave_result["high_risk"] else ft.colors.GREEN_700
            risk_bg = ft.colors.RED_50 if heatwave_result["high_risk"] else ft.colors.GREEN_50

            heatwave_card = ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Icon(ft.icons.WHATSHOT, color=risk_color, size=28),
                                ft.Text("Heatwave Risk", size=18, weight=ft.FontWeight.W_600),
                            ]
                        ),
                        ft.Container(height=10),
                        ft.Text(risk_level_text, size=22, weight=ft.FontWeight.BOLD, color=risk_color),
                        ft.Text(
                            f"Model confidence: {heatwave_result['probability']*100:.0f}%",
                            size=13,
                            color=ft.colors.GREY_600,
                        ),
                    ]
                ),
                bgcolor=risk_bg,
                border_radius=10,
                border=ft.border.all(1, risk_color),
                padding=20,
            )

        lat, lon = country_coordinates[country_name]
        try:
            current_weather = get_current_weather(lat, lon)
            disaster_flags = assess_disaster_risk(current_weather)

            disaster_items = []
            for flag in disaster_flags:
                is_safe = flag == "No significant weather-based risk detected"
                disaster_items.append(
                    ft.Row(
                        [
                            ft.Icon(
                                ft.icons.CHECK_CIRCLE if is_safe else ft.icons.WARNING,
                                color=ft.colors.GREEN_700 if is_safe else ft.colors.ORANGE_700,
                                size=18,
                            ),
                            ft.Text(flag, size=14),
                        ]
                    )
                )

            disaster_card = ft.Container(
                content=ft.Column(
                    [
                        ft.Text("Live Disaster Risk Monitor", size=18, weight=ft.FontWeight.W_600),
                        ft.Container(height=10),
                        *disaster_items,
                    ]
                ),
                bgcolor=ft.colors.WHITE,
                border_radius=10,
                border=ft.border.all(1, ft.colors.GREY_200),
                padding=20,
            )
        except Exception:
            disaster_card = ft.Text("Live disaster data temporarily unavailable", color=ft.colors.GREY_500)

        return ft.Column(
            [
                ft.Text(f"{country_name} — Risk Warnings", size=22, weight=ft.FontWeight.W_600),
                ft.Container(height=15),
                heatwave_card,
                ft.Container(height=20),
                disaster_card,
            ],
            spacing=5,
            scroll=ft.ScrollMode.AUTO,
        )

    content_area = ft.Container(
        content=build_dashboard(),
        expand=True,
        padding=30,
    )

    def nav_changed(e):
        selected_index = e.control.selected_index
        if selected_index == 0:
            content_area.content = build_dashboard()
        elif selected_index == 1:
            content_area.content = build_weather()
        elif selected_index == 2:
            content_area.content = build_predictions()
        elif selected_index == 3:
            content_area.content = build_risk_warnings()
        else:
            screens = ["Dashboard", "Weather", "Predictions", "Risk Warnings", "Chatbot"]
            content_area.content = ft.Text(f"{screens[selected_index]} screen coming soon", size=24)
        page.update()

    nav_rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=200,
        bgcolor=ft.colors.WHITE,
        destinations=[
            ft.NavigationRailDestination(icon=ft.icons.DASHBOARD_OUTLINED, selected_icon=ft.icons.DASHBOARD, label="Dashboard"),
            ft.NavigationRailDestination(icon=ft.icons.WB_SUNNY_OUTLINED, selected_icon=ft.icons.WB_SUNNY, label="Weather"),
            ft.NavigationRailDestination(icon=ft.icons.SHOW_CHART_OUTLINED, selected_icon=ft.icons.SHOW_CHART, label="Predictions"),
            ft.NavigationRailDestination(icon=ft.icons.WARNING_AMBER_OUTLINED, selected_icon=ft.icons.WARNING_AMBER, label="Risk"),
            ft.NavigationRailDestination(icon=ft.icons.CHAT_OUTLINED, selected_icon=ft.icons.CHAT, label="Chatbot"),
        ],
        on_change=nav_changed,
    )

    page.add(
        ft.Row(
            [nav_rail, ft.VerticalDivider(width=1), content_area],
            expand=True,
        )
    )

ft.app(target=main)