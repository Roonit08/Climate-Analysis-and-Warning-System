# Loading the saved country coordinates lookup
import json
from pathlib import Path

def load_country_coordinates():
    base_dir = Path(__file__).resolve().parent.parent.parent
    path = base_dir / "data" / "country_coordinates.json"
    with open(path) as f:
        return json.load(f)

def get_coordinates(country_name, country_coordinates):
    return country_coordinates.get(country_name)