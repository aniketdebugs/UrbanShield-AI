from pathlib import Path
import json
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DEMO_PATH = ROOT / "Indore_demo_data.csv"
REAL_DATA = ROOT / "data" / "processed" / "indore_heat.csv"
MODEL_PATH = ROOT / "models" / "heat_model.joblib"
META_PATH = ROOT / "models" / "metadata.json"

FEATURES = [
    "air_temp_c", "humidity", "wind_speed", "ndvi", "ndbi",
    "green_fraction", "built_fraction", "water_fraction",
    "latitude", "longitude"
]

st.set_page_config(page_title="UrbanShield AI | Indore", page_icon="🌍", layout="wide")

@st.cache_data
def read_dataset(path, mtime):
    return pd.read_csv(path, parse_dates=["date"])

def get_data():
    if REAL_DATA.exists() and META_PATH.exists() and MODEL_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        if meta.get("data_mode") == "REAL_DATA":
            return read_dataset(str(REAL_DATA), REAL_DATA.stat().st_mtime), "REAL", meta
    demo = read_dataset(str(DEMO_PATH), DEMO_PATH.stat().st_mtime)
    return demo, "DEMO", None

def fit_demo_model(df):
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df["lst_c"], test_size=0.25, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    residuals = np.abs(y_test.to_numpy() - pred)
    q = float(np.quantile(residuals, 0.90))
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
        "q": q
    }
    return model, metrics

@st.cache_data(ttl=3600)
def get_current_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 22.7196, "longitude": 75.8577,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "Asia/Kolkata"
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

df, mode, real_meta = get_data()
if mode == "REAL":
    model = joblib.load(MODEL_PATH)
    demo_metrics = None
else:
    model, demo_metrics = fit_demo_model(df)

st.title("🌍 UrbanShield AI")
st.caption("Indore urban heat intelligence • Geospatial Data Science • Decision support")
if mode == "DEMO":
    st.warning(
        "DEMO MODE: the map points and model target values below are synthetic demonstration data, "
        "not measured Indore temperatures. Replace them with validated Landsat observations before "
        "using model results in your academic report."
    )
else:
    st.success("REAL DATA MODE: loaded the processed dataset and model evaluation metadata.")

with st.sidebar:
    st.header("Study configuration")
    st.write("**Study area:** Indore, Madhya Pradesh")
    st.write("**Satellite target:** Land-surface temperature (°C)")
    st.write("**Grid:** approximately 1 km cells")
    st.divider()
    st.write("Use the tabs to explore data, run scenarios, and review model performance.")
    if st.button("Fetch current Indore weather"):
        try:
            weather = get_current_weather()
            st.session_state["weather"] = weather
        except Exception as e:
            st.error(f"Weather API request failed: {e}")

if "weather" in st.session_state:
    current = st.session_state["weather"].get("current", {})
    st.sidebar.subheader("Current city-level weather")
    st.sidebar.metric("Air temperature", f'{current.get("temperature_2m", "N/A")} °C')
    st.sidebar.metric("Relative humidity", f'{current.get("relative_humidity_2m", "N/A")} %')
    st.sidebar.metric("Wind speed", f'{current.get("wind_speed_10m", "N/A")} km/h')
    st.sidebar.caption("City-level API data; not neighborhood-scale or satellite LST.")

tab_map, tab_scenario, tab_budget, tab_model, tab_about = st.tabs([
    "🗺️ Heat map", "🧪 Scenario lab", "💰 Budget planner", "📊 Model evaluation", "ℹ️ Methodology"
])

with tab_map:
    st.subheader("Explore the study grid")

    # Keep the date column as datetime; avoid comparing .dt.date objects.
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    latest_date = df["date"].max()

    # Show the latest available observation for each grid cell.
    # Demo records have different dates, so filtering to the single latest
    # date can leave only one point on the map.
    view = (
        df.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["cell_id"], keep="last")
        .copy()
    )
    latest_rows = df.loc[df["date"].eq(latest_date)]

    st.write("Total rows:", len(df))
    st.write("Rows on latest date:", len(latest_rows))
    st.write(
        "Unique coordinates:",
        view[["latitude", "longitude"]].drop_duplicates().shape[0],
    )
    st.dataframe(
        view[["cell_id", "date", "latitude", "longitude", "lst_c"]].head(10),
        hide_index=True,
    )

    fig = px.scatter(
        view,
        x="longitude",
        y="latitude",
        color="lst_c",
        hover_name="cell_id",
        hover_data=["date", "lst_c", "ndvi", "ndbi"],
        color_continuous_scale="Inferno",
        height=600,
        title=f"Study grid — latest available observation per cell (latest date: {latest_date.date()})",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.85))
    fig.update_layout(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        xaxis=dict(scaleanchor="y", scaleratio=1),
    )

    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grid cells", f'{view["cell_id"].nunique():,}')
    c2.metric("Records", f'{len(df):,}')
    c3.metric("Latest dataset date", str(latest_date.date()))
    c4.metric("Data mode", mode)
    st.caption("Map colour represents the dataset target. In demo mode all target values are synthetic.")


with tab_scenario:
    st.subheader("Counterfactual scenario simulator")
    st.write("Change environmental features and compare model estimates. This is sensitivity analysis, not causal proof.")
    cell_id = st.selectbox("Choose grid cell", sorted(df["cell_id"].astype(str).unique()))
    row = df[df["cell_id"].astype(str) == cell_id].sort_values("date").iloc[-1]
    base = pd.DataFrame([{f: float(row[f]) for f in FEATURES}])
    left,right = st.columns(2)
    with left:
        green = st.slider("Scenario green fraction", 0.0, 1.0, float(row["green_fraction"]), 0.01)
        ndvi = st.slider("Scenario NDVI", -1.0, 1.0, float(row["ndvi"]), 0.01)
    with right:
        built = st.slider("Scenario built-up fraction", 0.0, 1.0, float(row["built_fraction"]), 0.01)
        water = float(row["water_fraction"])
        st.metric("Water fraction held constant", f"{water:.2f}")
    if green + built + water > 1.001:
        st.error("Green + built-up + water fractions exceed 1. Adjust the sliders.")
    else:
        scenario = base.copy()
        scenario.loc[0, "green_fraction"] = green
        scenario.loc[0, "ndvi"] = ndvi
        scenario.loc[0, "built_fraction"] = built
        p0 = float(model.predict(base)[0])
        p1 = float(model.predict(scenario)[0])
        a,b,c = st.columns(3)
        a.metric("Baseline estimate", f"{p0:.2f} °C")
        b.metric("Scenario estimate", f"{p1:.2f} °C", delta=f"{p1-p0:+.2f} °C")
        c.metric("Change", f"{p1-p0:+.2f} °C")
        st.info("In DEMO mode these are model outputs on synthetic data. For real-data mode, ensure the scenario is within the training distribution.")
        q = demo_metrics["q"] if mode == "DEMO" else float(real_meta["conformal_q"])
        st.write(f"Illustrative baseline interval: {p0-q:.2f} to {p0+q:.2f} °C")
        st.write(f"Illustrative scenario interval: {p1-q:.2f} to {p1+q:.2f} °C")
        st.caption("Calibration from the original distribution may not remain valid after changing features.")

with tab_budget:
    st.subheader("Budget-constrained intervention planner")
    st.write("Illustrative planning exercise. Intervention costs and cooling-effect assumptions are user-defined, not verified local quotes or causal estimates.")
    budget = st.number_input("Hypothetical budget (₹)", min_value=10000, max_value=100000000, value=500000, step=10000)
    st.write("Set assumed cost and estimated benefit per candidate intervention.")
    options = [
        ("Tree planting / green cover", 50000, 1.0),
        ("Cool-roof pilot", 80000, 1.3),
        ("Pocket park improvement", 120000, 1.5),
        ("Shaded bus-stop pilot", 40000, 0.6),
        ("Permeable / shaded public space", 90000, 0.9),
    ]
    candidates = []
    for name, default_cost, default_benefit in options:
        with st.expander(name, expanded=False):
            cost = st.number_input(f"Assumed cost — {name} (₹)", 1000, 10000000, default_cost, 1000, key=f"cost_{name}")
            benefit = st.number_input(f"Assumed benefit score — {name}", 0.0, 10.0, default_benefit, 0.1, key=f"benefit_{name}")
            candidates.append({"name":name, "cost":int(cost), "benefit":float(benefit)})
    # Small exact 0/1 knapsack for one unit of each option
    dp = {0: (0.0, [])}
    for item in candidates:
        updated = dict(dp)
        for spent, (score, chosen) in dp.items():
            new_spent = spent + item["cost"]
            if new_spent <= budget:
                new_score = score + item["benefit"]
                if new_spent not in updated or new_score > updated[new_spent][0]:
                    updated[new_spent] = (new_score, chosen + [item])
        dp = updated
    best_spent, (best_score, chosen) = max(dp.items(), key=lambda x: x[1][0])
    x,y,z = st.columns(3)
    x.metric("Budget", f"₹{budget:,.0f}")
    y.metric("Selected spend", f"₹{best_spent:,.0f}")
    z.metric("Total assumed benefit score", f"{best_score:.1f}")
    if chosen:
        st.dataframe(pd.DataFrame(chosen), hide_index=True, use_container_width=True)
    else:
        st.write("No intervention fits the current budget.")
    st.caption("Benefit scores are illustrative. Replace with evidence-based local estimates before drawing policy conclusions.")

with tab_model:
    st.subheader("Model performance")
    if mode == "REAL":
        metrics = {
            "MAE (°C)": real_meta["test_MAE"],
            "RMSE (°C)": real_meta["test_RMSE"],
            "R²": real_meta["test_R2"],
            "Nominal interval coverage": real_meta["nominal_interval_coverage"],
            "Observed test interval coverage": real_meta["test_interval_coverage"],
            "Mean interval width (°C)": real_meta["mean_interval_width"],
        }
        st.write(f"Selected model: **{real_meta['selected_model']}**")
    else:
        metrics = {
            "MAE (°C) — demo only": demo_metrics["MAE"],
            "RMSE (°C) — demo only": demo_metrics["RMSE"],
            "R² — demo only": demo_metrics["R2"],
            "Interval width — demo only": 2*demo_metrics["q"],
        }
        st.write("A Random Forest is fitted at app startup to synthetic demo data. These metrics are only for checking that the interface works.")
    st.dataframe(pd.DataFrame([{"Metric":k,"Value":v} for k,v in metrics.items()]), hide_index=True, use_container_width=True)
    st.warning("Do not report demo-mode scores as experimental results. Use the real-data training pipeline for academic evaluation.")

with tab_about:
    st.subheader("Methodology and limitations")
    st.markdown("""
    **Data sources planned for the research version**
    - NASA/USGS Landsat Collection 2 Level-2 surface temperature and reflectance.
    - Open-Meteo historical weather as city-level contextual information.
    - OpenStreetMap or validated satellite-derived land-cover features.

    **Methods**
    - Data cleaning and feature engineering.
    - Random Forest and Gradient Boosting regression.
    - MAE, RMSE and R² evaluation.
    - Split-conformal prediction intervals.
    - Constrained budget-allocation optimization.

    **Research limitations**
    - Land-surface temperature differs from air temperature and human heat exposure.
    - Weather grid resolution is coarser than neighborhood satellite pixels.
    - Scenario simulation is not causal inference.
    - Spatial and chronological generalization must be evaluated separately.
    - Demo data are synthetic; the research dataset must be real and quality checked.
    """)
    st.download_button(
        "Download currently loaded dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"urbanshield_{mode.lower()}_dataset.csv",
        mime="text/csv"
    )
