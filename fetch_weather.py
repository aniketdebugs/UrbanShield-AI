"""Fetch real historical weather context for Indore from Open-Meteo."""
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "indore_weather.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

params = {
    "latitude": 22.7196,
    "longitude": 75.8577,
    "start_date": "2020-01-01",
    "end_date": "2025-12-31",
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,cloud_cover",
    "timezone": "Asia/Kolkata",
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
}
url = "https://archive-api.open-meteo.com/v1/archive"
print("Downloading historical city-level weather for Indore...")
response = requests.get(url, params=params, timeout=90)
response.raise_for_status()
payload = response.json()
df = pd.DataFrame(payload["hourly"])
df.rename(columns={"time": "datetime"}, inplace=True)
df.to_csv(OUT, index=False)
print(f"Saved {len(df):,} rows: {OUT}")
print(df.head())
