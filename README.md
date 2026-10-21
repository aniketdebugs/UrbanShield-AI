# UrbanShield AI — Indore

A Streamlit Data Science research prototype for urban land-surface temperature exploration,
uncertainty-aware regression, counterfactual scenario comparison, and budget-constrained
intervention planning.

## Quick start (Windows / VS Code)

1. Extract the ZIP and open this folder in VS Code.
2. In the terminal, use Command Prompt if PowerShell blocks activation:
   ```cmd
   .venv\Scripts\activate.bat
   ```
   Create the environment first if needed:
   ```cmd
   python -m venv .venv
   ```
3. Install dependencies:
   ```cmd
   python -m pip install -r requirements.txt
   ```
4. Run the website:
   ```cmd
   python -m streamlit run app.py
   ```
5. Open the local URL Streamlit prints (usually http://localhost:8501).

The app works immediately in DEMO mode with clearly labeled synthetic data so you can
test the website without waiting for satellite exports. Never present demo data or demo
metrics as real Indore observations or research results.

## Optional: current weather
Click "Fetch current Indore weather" in the sidebar. This makes a real request to the
Open-Meteo forecast API. Weather shown is city-level context, not neighborhood-level LST.

## Build the real dataset

### A. Historical weather
```cmd
python src/fetch_weather.py
```
Creates `data/raw/indore_weather.csv` using Open-Meteo historical API.

### B. Landsat data
1. Open https://code.earthengine.google.com/ and sign in.
2. Paste `gee/export_indore_landsat.js` into a new script.
3. Run and inspect errors/scene counts.
4. Export the CSV through the Tasks panel and download it from Google Drive into `data/raw/`.
5. Inspect QA, null values, units, and date coverage before research use.

### C. Important preprocessing work still required
The Landsat export supplies target `lst_c` and satellite features `ndvi`, `ndbi`, coordinates,
date, and cell ID. Before training the research model, build
`data/processed/indore_heat.csv` by:
- joining appropriate weather observations to satellite acquisition dates,
- explicitly documenting weather temporal/spatial resolution limitations,
- deriving validated green/built/water fractions or omitting those features,
- ensuring no synthetic data are mixed into the real dataset,
- checking quality, missing values, units, and physical plausibility,
- defining spatial and chronological holdouts to avoid leakage.

The required real-data schema for `src/train_model.py`:
`cell_id,date,lst_c,air_temp_c,humidity,wind_speed,ndvi,ndbi,green_fraction,built_fraction,water_fraction,latitude,longitude`

Only run real model training after that schema is built and checked:
```cmd
python src/train_model.py
```
The script saves `models/heat_model.joblib` and `models/metadata.json`. The app switches to
REAL mode only when metadata declares `data_mode` as `REAL_DATA`.

## Deploy
Push the project to a GitHub repository and deploy `app.py` using Streamlit Community Cloud.
Ensure all required dependencies and files are available. Keep raw large data out of Git when
possible. For a real-data deployment, include the processed data and trained artifacts or
configure an appropriate data/model hosting solution.

## Data sources
- Landsat 8 Collection 2 Level-2 catalog:
  https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2
- NASA/USGS Landsat surface temperature information:
  https://data.nasa.gov/dataset/landsat-level-2-surface-temperature-collection-2
- Open-Meteo Historical Weather API:
  https://open-meteo.com/en/docs/historical-weather-api
- Open-Meteo Forecast API:
  https://open-meteo.com/en/docs
- OpenStreetMap:
  https://www.openstreetmap.org/

## Research honesty
This is a starter implementation, not a validated official heat warning system. Demo mode is
synthetic. Claims of novelty require a literature review and measured comparison experiments.
