import pandas as pd
from connection import get_connection
from pathlib import Path

# Building the path to the feature-engineered dataset, regardless of where this script is run from
BASE_DIR = Path(__file__).resolve().parent.parent.parent
csv_path = BASE_DIR / "data" / "model_ready_data.csv"

# Loading the model-ready dataset
df = pd.read_csv(csv_path)

# Connecting to the database
conn = get_connection()
cur = conn.cursor()

# Inserting each unique country into the countries table
countries = df["Country Name"].unique()
for country in countries:
    cur.execute(
        """
        INSERT INTO countries (country_name)
        VALUES (%s)
        ON CONFLICT (country_name) DO NOTHING
        """,
        (country,)
    )

conn.commit()

# Fetching country_id for each country so temperature records can be linked correctly
cur.execute("SELECT country_id, country_name FROM countries")
country_id_map = {name: cid for cid, name in cur.fetchall()}

# Inserting each year's temperature record, linked to its country
for _, row in df.iterrows():
    country_id = country_id_map[row["Country Name"]]
    cur.execute(
        """
        INSERT INTO temperature_records (country_id, year, temp_anomaly)
        VALUES (%s, %s, %s)
        """,
        (country_id, int(row["year"]), float(row["temp_anomaly"]))
    )

conn.commit()
cur.close()
conn.close()

# Confirming the data load completed successfully
print("Data successfully loaded into PostgreSQL.")
