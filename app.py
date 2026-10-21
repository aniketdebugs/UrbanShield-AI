from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
# Prefer the processed research dataset when present; otherwise use the bundled demo CSV.
REAL_DATA_PATH = ROOT / "data" / "processed" / "indore_heat.csv"
METADATA_PATH = ROOT / "models" / "metadata.json"
CANDIDATE_FILES = [
    ROOT / "indore_demo_data.csv",
    ROOT / "Indore_demo_data.csv",
    ROOT / "data" / "demo" / "indore_demo_data.csv",
    ROOT / "data" / "demo" / "Indore_demo_data.csv",
]
TARGET_CANDIDATES = ["lst_c", "land_surface_temperature_c", "LST_C", "lst"]

def research_data_is_validated():
    """Only remove demo status when metadata explicitly records validated provenance."""
    try:
        metadata = pd.read_json(METADATA_PATH, typ="series").to_dict()
        return metadata.get("data_status") == "validated_real" and REAL_DATA_PATH.exists()
    except Exception:
        return False

VALIDATED_RESEARCH_DATA = research_data_is_validated()
DATA_PATH = REAL_DATA_PATH if VALIDATED_RESEARCH_DATA else next((p for p in CANDIDATE_FILES if p.exists()), None)
IS_DEMO = not VALIDATED_RESEARCH_DATA

st.set_page_config(page_title="UrbanShield AI | Indore", page_icon="🌍", layout="wide")

st.title("🌍 UrbanShield AI")
st.caption("Indore urban heat intelligence · Geospatial data science · Decision support")
if IS_DEMO:
    st.warning(
        "DEMO MODE: The currently loaded CSV is not marked as validated real-world data. "
        "Its values must be treated as illustrative, not measured Indore temperatures. "
        "To enable research mode, add a quality-checked Landsat-derived dataset at "
        "data/processed/indore_heat.csv and a models/metadata.json file with "
        '\"data_status\": \"validated_real\" plus source, acquisition dates, processing notes, and validation details.'
    )
else:
    st.success("RESEARCH DATA MODE: Dataset provenance is marked as validated in metadata. Review source and validation details before interpreting results.")

def load_data():
    path = DATA_PATH
    if path is None or not path.exists():
        st.error(
            "Could not find a usable dataset. For the demo, place indore_demo_data.csv next to app.py. "
            "For research mode, add data/processed/indore_heat.csv and validated metadata."
        )
        st.stop()
    st.caption(f"Loaded dataset file: `{path.relative_to(ROOT)}`")
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    target = next((c for c in TARGET_CANDIDATES if c.lower() in df.columns), None)
    if target is None:
        st.error("The dataset must include an LST target column, such as 'lst_c'.")
        st.stop()
    if target != "lst_c":
        df = df.rename(columns={target: "lst_c"})
    for col in ["date", "latitude", "longitude", "lst_c"]:
        if col not in df.columns:
            if col == "date":
                df["date"] = pd.NaT
            elif col in ["latitude", "longitude"]:
                df[col] = np.nan
            else:
                st.error(f"Required target column '{col}' is missing.")
                st.stop()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_defaults = {
        "air_temp_c": 32.0, "humidity": 50.0, "wind_speed": 2.0,
        "ndvi": 0.3, "ndbi": 0.1, "green_fraction": 0.3,
        "built_fraction": 0.5, "water_fraction": 0.1,
    }
    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    for col in ["latitude", "longitude", "lst_c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "lst_c"]).copy()
    if "cell_id" not in df.columns:
        df["cell_id"] = [f"CELL_{i:04d}" for i in range(len(df))]
    df["cell_id"] = df["cell_id"].astype(str)
    return df

df = load_data()
FEATURES = [
    "air_temp_c", "humidity", "wind_speed", "ndvi", "ndbi",
    "green_fraction", "built_fraction", "water_fraction", "latitude", "longitude"
]
for c in FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0)

@st.cache_data
def fit_demo_model(data, features):
    X = data[features].astype(float)
    y = data["lst_c"].astype(float)
    if len(data) < 8:
        # A model can still be shown for the demo, but evaluation is explicitly limited.
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "MAE (°C)": mean_absolute_error(y_test, predictions),
        "RMSE (°C)": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R²": r2_score(y_test, predictions) if len(y_test) > 1 else np.nan,
        "Within ±2°C": float(np.mean(np.abs(y_test.to_numpy() - predictions) <= 2) * 100),
    }
    # Refit on all available rows for scenario predictions.
    model.fit(X, y)
    return model, metrics

model, metrics = fit_demo_model(df, FEATURES)
tabs = st.tabs([
    "🗺️ Heat map", "🧪 Scenario lab", "💰 Budget planner",
    "📊 Model evaluation", "✨ Priority explorer", "ℹ️ Methodology"
])

with tabs[0]:
    st.subheader("Explore the study grid")
    latest_date = df["date"].max()
    view = df.sort_values("date").drop_duplicates("cell_id", keep="last")
    if view[["latitude", "longitude"]].drop_duplicates().shape[0] > 1:
        # Real OpenStreetMap street tiles centered on Indore. The basemap is real;
        # the heat values remain demo values unless a validated Landsat dataset is loaded.
        fig = px.scatter_mapbox(
            view, lat="latitude", lon="longitude", color="lst_c",
            hover_name="cell_id", hover_data=["lst_c", "ndvi", "ndbi"],
            color_continuous_scale="Inferno", zoom=11, height=600,
            map_style="open-street-map",
            center={"lat": 22.7196, "lon": 75.8577},
            title="Indore street map with heat-grid overlay"
        )
        fig.update_layout(
            map=dict(center={"lat": 22.7196, "lon": 75.8577}, zoom=11),
            margin={"l": 0, "r": 0, "t": 55, "b": 0}
        )
        st.caption("Basemap: OpenStreetMap street geography around Indore. Heat-coloured points come from the currently loaded dataset; in DEMO MODE, their values are synthetic and are not real measured temperatures.")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Only one valid coordinate was found, so a geographic street-map overlay cannot be drawn. Check the latitude and longitude columns.")
        fig = px.scatter(
            view, x="longitude", y="latitude", color="lst_c",
            hover_name="cell_id", hover_data=["lst_c", "ndvi", "ndbi"],
            color_continuous_scale="Inferno", height=500,
            title="Grid cell location (single coordinate)"
        )
        st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Grid cells", f'{view["cell_id"].nunique():,}')
    c2.metric("Records", f'{len(df):,}')
    c3.metric("Latest dataset date", latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "Not provided")
    st.dataframe(view[["cell_id", "date", "latitude", "longitude", "lst_c", "ndvi", "ndbi"]].head(20),
                 use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Counterfactual scenario simulator")
    st.write("Change environmental features and compare model estimates. This is sensitivity analysis, not causal proof.")
    selected_cell = st.selectbox("Choose grid cell", df["cell_id"].tolist(), key="scenario_cell")
    base = df.loc[df["cell_id"] == selected_cell].iloc[-1]
    col1, col2 = st.columns(2)
    with col1:
        green = st.slider("Scenario green fraction", 0.0, 1.0, float(np.clip(base["green_fraction"], 0, 1)), 0.01)
        ndvi = st.slider("Scenario NDVI", -1.0, 1.0, float(np.clip(base["ndvi"], -1, 1)), 0.01)
    with col2:
        built = st.slider("Scenario built-up fraction", 0.0, 1.0, float(np.clip(base["built_fraction"], 0, 1)), 0.01)
        st.metric("Water fraction held constant", f'{base["water_fraction"]:.2f}')
    if green + built + float(base["water_fraction"]) > 1:
        st.error("Green + built-up + water fractions exceed 1. Adjust the sliders.")
    else:
        baseline = model.predict(pd.DataFrame([{f: base[f] for f in FEATURES}]))[0]
        scenario_row = {f: base[f] for f in FEATURES}
        scenario_row.update({"green_fraction": green, "built_fraction": built, "ndvi": ndvi})
        scenario = model.predict(pd.DataFrame([scenario_row])[FEATURES])[0]
        a, b, c = st.columns(3)
        a.metric("Baseline estimate", f"{baseline:.2f} °C")
        b.metric("Scenario estimate", f"{scenario:.2f} °C", f"{scenario-baseline:+.2f} °C")
        c.metric("Change", f"{scenario-baseline:+.2f} °C")
        st.info("In DEMO mode these are model outputs on synthetic data. They are not measured cooling effects.")
        st.caption("Scenario comparisons are illustrative; do not interpret them as causal evidence.")

with tabs[2]:
    st.subheader("Budget-constrained intervention planner")
    st.caption("Illustrative planning exercise. Costs and benefit scores are user-defined assumptions, not verified local quotes or causal estimates.")
    budget = st.number_input("Hypothetical budget (₹)", min_value=0, value=500000, step=25000)
    defaults = [
        ("Tree planting / green cover", 50000, 1.00),
        ("Cool-roof pilot", 80000, 1.30),
        ("Pocket park improvement", 120000, 1.50),
        ("Shaded bus-stop pilot", 40000, 0.60),
        ("Permeable / shaded public space", 90000, 0.90),
    ]
    costs, benefits = [], []
    for i, (name, cost0, benefit0) in enumerate(defaults):
        with st.expander(name, expanded=False):
            cost = st.number_input(f"Assumed cost — {name} (₹)", min_value=0, value=cost0, step=5000, key=f"cost_{i}")
            benefit = st.number_input(f"Assumed benefit score — {name}", min_value=0.0, value=benefit0, step=0.1, key=f"benefit_{i}")
        costs.append(cost)
        benefits.append(benefit)
    spend = sum(costs)
    total_benefit = sum(benefits[i] for i in range(len(defaults)) if costs[i] > 0)
    a, b, c = st.columns(3)
    a.metric("Budget", f"₹{budget:,.0f}")
    b.metric("Selected spend", f"₹{spend:,.0f}", f"₹{budget-spend:,.0f} remaining" if spend <= budget else f"₹{spend-budget:,.0f} over budget")
    c.metric("Total assumed benefit score", f"{total_benefit:.1f}")
    if spend > budget:
        st.error("The listed interventions exceed the budget. Adjust costs or budget.")
    else:
        st.progress(0 if budget == 0 else min(spend / budget, 1.0))
    summary = pd.DataFrame({"Intervention": [x[0] for x in defaults], "Assumed cost (₹)": costs, "Assumed benefit score": benefits})
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption("Benefit scores are illustrative and should be replaced with evidence-based local estimates before drawing policy conclusions.")

with tabs[3]:
    st.subheader("Model performance")
    if IS_DEMO:
        st.write("A Random Forest model is fitted to demonstration data. These metrics check the interface and workflow; they do not validate real-world accuracy.")
    else:
        st.write("A Random Forest model is fitted to the dataset marked as validated in the project metadata. Review the sampling design and spatial/temporal split before interpreting these metrics.")
    if metrics is None:
        st.warning("Too few records are available for a meaningful held-out evaluation. Add more quality-checked records before interpreting metrics.")
    else:
        st.dataframe(pd.DataFrame({"Metric": list(metrics.keys()), "Value": [
            f'{metrics["MAE (°C)"]:.3f}',
            f'{metrics["RMSE (°C)"]:.3f}',
            f'{metrics["R²"]:.3f}' if pd.notna(metrics["R²"]) else "N/A",
            f'{metrics["Within ±2°C"]:.1f}%'
        ]}), use_container_width=True, hide_index=True)
    if IS_DEMO:
        st.warning("These scores are demo-only because the active data are not marked as validated real-world observations.")
    else:
        st.warning("Metrics are only as reliable as the data and validation design. Prefer spatial and chronological holdouts; a random split can overstate generalization.")

with tabs[4]:
    st.subheader("Priority Explorer")
    st.write("A transparent screening tool to help decide which grid cells to investigate first. It is not a validated heat-risk or health-risk model.")
    if IS_DEMO:
        st.info("DEMO MODE: Priority rankings use unvalidated demonstration values and illustrative weights. They are not official risk ratings.")
    else:
        st.info("Priority rankings are a transparent screening heuristic based on the loaded dataset, not a validated health-risk model or causal estimate.")
    high_temp_weight = st.slider("Weight: relative temperature", 0.0, 1.0, 0.50, 0.05)
    built_weight = st.slider("Weight: built-up fraction", 0.0, 1.0, 0.25, 0.05)
    low_green_weight = st.slider("Weight: low green cover", 0.0, 1.0, 0.25, 0.05)
    weights_total = high_temp_weight + built_weight + low_green_weight
    if weights_total == 0:
        st.warning("Set at least one weight above zero.")
    else:
        ranked = df.copy()
        def norm(s):
            lo, hi = s.min(), s.max()
            return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=s.index)
        ranked["Temperature indicator"] = norm(ranked["lst_c"])
        ranked["Built-up indicator"] = norm(ranked["built_fraction"])
        ranked["Low-green indicator"] = 1 - norm(ranked["green_fraction"])
        ranked["Priority score"] = 100 * (
            high_temp_weight * ranked["Temperature indicator"] +
            built_weight * ranked["Built-up indicator"] +
            low_green_weight * ranked["Low-green indicator"]
        ) / weights_total
        ranked = ranked.sort_values("Priority score", ascending=False)
        ranked["Screening tier"] = pd.cut(
            ranked["Priority score"], bins=[-0.01, 33.33, 66.67, 100.01],
            labels=["Lower illustrative priority", "Medium illustrative priority", "Higher illustrative priority"]
        )
        top_n = st.slider("Number of grid cells to display", 5, min(50, max(5, len(ranked))), min(10, max(5, len(ranked))))
        st.dataframe(ranked[["cell_id", "lst_c", "green_fraction", "built_fraction", "Priority score", "Screening tier"]].head(top_n),
                     use_container_width=True, hide_index=True)
        chart = px.bar(ranked.head(top_n), x="cell_id", y="Priority score", color="Screening tier",
                       title="Highest illustrative screening scores", labels={"Priority score": "Illustrative score (0–100)"})
        st.plotly_chart(chart, use_container_width=True)
        cell = st.selectbox("Explain a grid cell", ranked["cell_id"].head(top_n).tolist(), key="priority_explain")
        item = ranked.loc[ranked["cell_id"] == cell].iloc[0]
        st.markdown(f"**Why this cell ranks here:** its relative temperature indicator is {item['Temperature indicator']:.2f}, built-up indicator is {item['Built-up indicator']:.2f}, and low-green indicator is {item['Low-green indicator']:.2f}.")
        st.caption("The score is a transparent weighted screening heuristic, not a learned probability, validated risk score, or estimate of health outcomes.")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download currently loaded dataset CSV", data=csv, file_name="urbanshield_loaded_dataset.csv", mime="text/csv")

with tabs[5]:
    st.subheader("Methodology and limitations")
    st.markdown("""
**Data sources planned for the research version**
- NASA/USGS Landsat Collection 2 Level-2 surface temperature and reflectance.
- Open-Meteo historical weather as city-level contextual information.
- OpenStreetMap street basemap for geographic context (the current heat overlay uses the loaded CSV values).
- Validated satellite-derived land-cover features for research mode.

**Methods**
- Data cleaning and feature engineering.
- Random Forest and Gradient Boosting regression (the current demo model uses Random Forest).
- MAE, RMSE and R² evaluation.
- Split-conformal prediction intervals are a planned research method, not implemented in this demo version.
- Constrained budget-allocation optimization is a planned extension; the current budget planner compares user-entered assumptions.

**Research limitations**
- Land-surface temperature differs from air temperature and human heat exposure.
- Weather grid resolution is coarser than neighbourhood satellite pixels.
- Scenario simulation is not causal inference.
- Spatial and chronological generalization must be evaluated separately.
- The app remains in DEMO MODE unless a processed dataset and explicit provenance metadata are provided.
- Landsat land-surface temperature is not the same as near-surface air temperature or personal heat exposure.
""")
    st.download_button("Download currently loaded dataset CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="urbanshield_loaded_dataset.csv", mime="text/csv")

st.caption("UrbanShield AI · Geospatial heat-screening prototype · Verify dataset provenance, validation, and intervention assumptions before real-world use.")
