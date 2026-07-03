CREATE TABLE countries (
    country_id SERIAL PRIMARY KEY,
    country_name VARCHAR(100) UNIQUE NOT NULL,
    latitude FLOAT,
    longitude FLOAT
);

CREATE TABLE temperature_records (
    record_id SERIAL PRIMARY KEY,
    country_id INT REFERENCES countries(country_id),
    year INT NOT NULL,
    month INT,
    temp_anomaly FLOAT
);

CREATE TABLE predictions (
    prediction_id SERIAL PRIMARY KEY,
    country_id INT REFERENCES countries(country_id),
    year INT NOT NULL,
    predicted_temp FLOAT,
    model_used VARCHAR(50)
);

CREATE TABLE heatwave_risk (
    risk_id SERIAL PRIMARY KEY,
    country_id INT REFERENCES countries(country_id),
    year INT NOT NULL,
    risk_score FLOAT,
    risk_level VARCHAR(20)
);

CREATE TABLE live_weather_cache (
    cache_id SERIAL PRIMARY KEY,
    country_id INT REFERENCES countries(country_id),
    fetched_at TIMESTAMP DEFAULT NOW(),
    current_temp FLOAT,
    humidity FLOAT,
    condition VARCHAR(50)
);