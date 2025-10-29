import io
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit

# --- Streamlit basic configuration -----------------------------------------------------------
st.set_page_config(
    page_title="Análisis de Declinación de Pozos",
    page_icon="📉",
    layout="wide",
)

# --- Modern UI Styles -----------------------------------------------------------------------
# Inyectar CSS personalizado para estilo tipo iPhone
st.markdown(
    """
    <style>
    body, .main, .block-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e3e7ed 100%) !important;
        font-family: 'San Francisco', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
    }
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e3e7ed 100%) !important;
    }
    .stTitle, .stHeader, .stSubheader {
        color: #222;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4f8cff 0%, #38c6ff 100%) !important;
        color: white !important;
        border-radius: 16px !important;
        border: none !important;
        font-size: 1.1em !important;
        padding: 0.5em 1.5em !important;
        box-shadow: 0 2px 8px rgba(80,140,255,0.08);
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #38c6ff 0%, #4f8cff 100%) !important;
    }
    .metric-card {
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 2px 12px rgba(80,140,255,0.08);
        padding: 1.2em 1em;
        margin-bottom: 1em;
        text-align: center;
    }
    .metric-title {
        color: #4f8cff;
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 0.2em;
    }
    .metric-value {
        color: #222;
        font-size: 1.5em;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f4f7fa !important;
        border-radius: 16px 16px 0 0 !important;
        font-weight: 700 !important;
        color: #4f8cff !important;
        font-size: 1.1em !important;
    }
    /* Reglas adicionales para forzar negrita en los labels internos de los tabs */
    .stTabs [data-baseweb="tab"] * {
        font-weight: 700 !important;
    }
    .stTabs button[role="tab"] {
        font-weight: 700 !important;
    }
    .stTabs [aria-selected="true"] {
        background: #fff !important;
        color: #222 !important;
        box-shadow: 0 2px 8px rgba(80,140,255,0.08);
        font-weight: 800 !important;
    }
    .stDataFrame, .stTable {
        background: #fff !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(80,140,255,0.08);
    }
    .stDownloadButton > button {
        background: linear-gradient(90deg, #38c6ff 0%, #4f8cff 100%) !important;
        color: white !important;
        border-radius: 14px !important;
        border: none !important;
        font-size: 1em !important;
        padding: 0.4em 1.2em !important;
        margin-top: 0.5em;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #4f8cff 0%, #38c6ff 100%) !important;
    }
    .stTextInput > div > input {
        border-radius: 12px !important;
        border: 1px solid #c3d0e8 !important;
        background: #f8fafc !important;
        font-size: 1em !important;
    }
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 1px solid #c3d0e8 !important;
        background: #f8fafc !important;
        font-size: 1em !important;
    }
    .stSlider > div {
        background: #e3e7ed !important;
        border-radius: 12px !important;
    }
    .stNumberInput > div > input {
        border-radius: 12px !important;
        border: 1px solid #c3d0e8 !important;
        background: #f8fafc !important;
        font-size: 1em !important;
    }
    .stForm > div {
        background: #fff !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 8px rgba(80,140,255,0.08);
        padding: 1em 1em;
    }
    .stWarning, .stError, .stSuccess {
        border-radius: 12px !important;
        font-size: 1.05em !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Constants -------------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "Datos"
MONTH_DAYS = 30.4375  # Average days per month for time conversion

SESSION_DEFAULTS: Dict[str, object] = {
    "df": None,
    "df_wide": None,
    "df_loaded": None,
    "fits": {},
    "preview": None,
    "fit_start": {},
    "fit_end": {},
    "forecast_months": 12,
    "extrap_months": 60,
    "source_name": None,
    "last_uploaded_file_id": None,
    "update_counter": 0,
    "pending_mapping": None,
    "pending_mapping_preview": None,
}


# --- Custom exceptions -----------------------------------------------------------------------
class ColumnMappingRequired(ValueError):
    """Raised when automatic column detection fails and user input is required."""

    def __init__(self, columns: List[str]):
        message = (
            "No se pudieron identificar automáticamente las columnas de fecha, pozo y producción. "
            "Selecciona las columnas correctas para continuar."
        )
        super().__init__(message)
        self.columns = columns


# --- Helper functions ------------------------------------------------------------------------
def init_session_state() -> None:
    """Ensure all expected session state keys exist."""
    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            # Use copies for mutable defaults
            if isinstance(default_value, dict):
                st.session_state[key] = {}
            else:
                st.session_state[key] = default_value


def list_default_datasets() -> List[Path]:
    """Return CSV files located in the Datos/ directory."""
    if not DATA_DIR.exists():
        return []
    return sorted(p for p in DATA_DIR.glob("*.csv") if p.is_file())


def normalise_column(label: str) -> str:
    """Normalise column names to ease format detection."""
    return "".join(ch for ch in label.lower() if ch.isalnum())


def read_csv_with_fallbacks(file_bytes: bytes) -> pd.DataFrame:
    """Try to read CSV content using multiple encodings."""
    errors = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                io.StringIO(file_bytes.decode(encoding)),
                sep=None,
                engine="python",
            )
        except Exception as exc:  # pragma: no cover - informative feedback
            errors.append(f"{encoding}: {exc}")
    raise ValueError("No fue posible leer el archivo CSV. Errores:\n" + "\n".join(errors))


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to float handling comma and dot decimal separators."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return (
        pd.to_numeric(
            series.astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )
        .astype(float)
    )


def detect_columns(df_raw: pd.DataFrame) -> Tuple[str, str, str]:
    """Identify date, well and production columns."""
    normalized = {col: normalise_column(col) for col in df_raw.columns}

    date_col = None
    well_col = None
    prod_col = None

    for col, norm in normalized.items():
        if date_col is None and (
            "fecha" in norm
            or norm.startswith("date")
            or norm in {"ddmmyyyy", "yyyymmdd", "yyyyddmm"}
        ):
            date_col = col
        if well_col is None and ("pozo" in norm or "well" in norm):
            well_col = col
        if prod_col is None and (
            norm.startswith("qo")
            or "produccion" in norm
            or "production" in norm
            or norm.endswith("m3d")
        ):
            prod_col = col

    if not all([date_col, well_col, prod_col]):
        missing = [
            label
            for label, value in (
                ("fecha", date_col),
                ("pozo", well_col),
                ("produccion", prod_col),
            )
            if value is None
        ]
        raise ValueError(
            "No se pudieron localizar las columnas necesarias: " + ", ".join(missing)
        )

    return date_col, well_col, prod_col


def guess_date_column(df_clean: pd.DataFrame) -> Optional[str]:
    """Infer which column contains dates."""
    normalized = {col: normalise_column(col) for col in df_clean.columns}
    for col, norm in normalized.items():
        if "fecha" in norm or "date" in norm or "ddmmyyyy" in norm:
            return col
    for col in df_clean.columns:
        converted = pd.to_datetime(df_clean[col], dayfirst=True, errors="coerce")
        if converted.notna().mean() > 0.6:
            return col
    return None


def guess_well_column(columns: List[str]) -> Optional[str]:
    """Heuristic to suggest well identifier column."""
    for col in columns:
        norm = normalise_column(col)
        if "pozo" in norm or "well" in norm or norm.endswith("id"):
            return col
    return None


def guess_production_column(columns: List[str]) -> Optional[str]:
    """Heuristic to suggest production column."""
    for col in columns:
        norm = normalise_column(col)
        if norm.startswith("qo") or "produccion" in norm or "production" in norm:
            return col
    return None


def convert_wide_to_long(df_clean: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Handle wide-format datasets by melting into long format."""
    date_col = guess_date_column(df_clean)
    if date_col is None:
        return None

    value_columns = [col for col in df_clean.columns if col != date_col]
    if not value_columns:
        return None

    melted = df_clean.melt(
        id_vars=[date_col],
        value_vars=value_columns,
        var_name="pozo",
        value_name="produccion",
    )

    melted.columns = melted.columns.str.strip()
    melted["pozo"] = melted["pozo"].astype(str).str.strip()
    return melted.rename(columns={date_col: "fecha"})


def preprocess_dataframe(
    df_raw: pd.DataFrame,
    mapping: Optional[Tuple[str, str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Transform raw data into long and wide processed DataFrames."""
    df_clean = df_raw.copy()
    df_clean.columns = [col.strip() for col in df_clean.columns]
    df_clean = df_clean.dropna(how="all")

    if mapping:
        date_col, well_col, prod_col = mapping
        missing = [col for col in mapping if col not in df_clean.columns]
        if missing:
            raise ValueError(
                "Las columnas seleccionadas no existen en el archivo: " + ", ".join(missing)
            )
        df_long = df_clean[[date_col, well_col, prod_col]].rename(
            columns={date_col: "fecha", well_col: "pozo", prod_col: "produccion"}
        )
    else:
        try:
            date_col, well_col, prod_col = detect_columns(df_clean)
        except ValueError as base_exc:
            df_long = convert_wide_to_long(df_clean)
            if df_long is None:
                raise ColumnMappingRequired(list(df_clean.columns)) from base_exc
        else:
            df_long = df_clean[[date_col, well_col, prod_col]].rename(
                columns={date_col: "fecha", well_col: "pozo", prod_col: "produccion"}
            )

    df_long["fecha"] = pd.to_datetime(df_long["fecha"], dayfirst=True, errors="coerce")
    df_long["pozo"] = df_long["pozo"].astype(str).str.strip()
    df_long["produccion"] = coerce_numeric(df_long["produccion"])

    df_long = df_long.dropna(subset=["fecha"])
    df_long = df_long.dropna(subset=["produccion"])
    df_long = df_long[df_long["pozo"] != ""]
    df_long = df_long[df_long["pozo"].str.lower() != "nan"]
    df_long = df_long.sort_values(["pozo", "fecha"]).reset_index(drop=True)

    df_wide = (
        df_long.pivot_table(
            index="fecha",
            columns="pozo",
            values="produccion",
            aggfunc="mean",
        )
        .sort_index()
        .copy()
    )

    return df_long, df_wide


@st.cache_data(show_spinner=False)
def load_processed_data(
    file_bytes: bytes,
    signature: str,
    mapping: Optional[Tuple[str, str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and preprocess CSV content. Cached by signature."""
    df_raw = read_csv_with_fallbacks(file_bytes)
    df_long, df_wide = preprocess_dataframe(df_raw, mapping)
    return df_raw, df_long, df_wide


def reset_state_with_data(
    df_raw: pd.DataFrame,
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    source_name: str,
    file_id: Optional[str],
) -> None:
    """Update session state after loading new data."""
    st.session_state.df_loaded = df_raw
    st.session_state.df = df_long
    st.session_state.df_wide = df_wide
    st.session_state.source_name = source_name
    st.session_state.last_uploaded_file_id = file_id

    # Clear analysis state
    st.session_state.fits = {}
    st.session_state.preview = None
    st.session_state.fit_start = {}
    st.session_state.fit_end = {}
    st.session_state.forecast_months = SESSION_DEFAULTS["forecast_months"]
    st.session_state.extrap_months = SESSION_DEFAULTS["extrap_months"]
    st.session_state.pending_mapping = None
    st.session_state.pending_mapping_preview = None
    st.session_state.update_counter += 1


def arps_decline(t: np.ndarray, qo: float, decline: float, b: float) -> np.ndarray:
    """Arps hyperbolic decline model."""
    if np.isclose(b, 0.0):
        return qo * np.exp(-decline * t)
    return qo / np.power(1.0 + b * decline * t, 1.0 / b)


def months_between(start: pd.Timestamp, end: Iterable[pd.Timestamp]) -> np.ndarray:
    """Return the elapsed months between start and each timestamp."""
    delta = pd.to_datetime(end) - start
    return np.array([d.days / MONTH_DAYS for d in delta])


def fit_decline_curve(well_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Fit Arps decline parameters for a single well subset."""
    # Asegurar que no hay valores menores a 0.1
    well_df = well_df[well_df["produccion"] >= 0.1].copy()
    
    if len(well_df) < 3:
        return None

    base_date = well_df["fecha"].min()
    t = months_between(base_date, well_df["fecha"])
    q = well_df["produccion"].to_numpy()

    # Validación mínima de datos
    if not all(np.isfinite(q)):
        return None

    # Intentar diferentes estimaciones iniciales
    guesses = [
        (max(q[0], 1.0), 0.05, 0.5),  # Original
        (np.mean(q), 0.01, 0.8),       # Media con decline suave
        (max(q), 0.1, 0.3),            # Máximo con decline fuerte
    ]
    
    best_fit = None
    min_mse = float('inf')
    
    for qo_guess, di_guess, b_guess in guesses:
        try:
            popt, _ = curve_fit(
                arps_decline,
                t,
                q,
                p0=[qo_guess, di_guess, b_guess],
                bounds=([0.0, 0.0, 0.0], [np.inf, 1.0, 5.0]),
                maxfev=20000,
            )
            
            qo, decline, b = popt
            if all(np.isfinite([qo, decline, b])):
                q_hat = arps_decline(t, qo, decline, b)
                mse = float(np.mean((q - q_hat) ** 2))
                
                if mse < min_mse:
                    min_mse = mse
                    # R² calculado pero solo informativo, no para filtrar
                    ss_res = np.sum((q - q_hat) ** 2)
                    ss_tot = np.sum((q - np.mean(q)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    best_fit = {
                        "qo": float(qo),
                        "D": float(decline),
                        "b": float(b),
                        "mse": mse,
                        "r2": float(r_squared),
                        "n_points": len(well_df),
                        "fit_start": well_df["fecha"].min().isoformat(),
                        "fit_end": well_df["fecha"].max().isoformat(),
                        "data_start": base_date.isoformat(),
                        "data_end": well_df["fecha"].max().isoformat(),
                    }
        except Exception:
            continue
    
    return best_fit


def get_well_dataset(well: str) -> pd.DataFrame:
    """Return long-format data for a specific well."""
    df_long: pd.DataFrame = st.session_state.df
    return df_long[df_long["pozo"] == well].copy()


def build_period_options(well_df: pd.DataFrame) -> List[str]:
    """Return available month periods for slider selection."""
    periods = (
        well_df["fecha"]
        .dt.to_period("M")
        .drop_duplicates()
        .sort_values()
    )
    return [str(period) for period in periods]


def filter_by_period(well_df: pd.DataFrame, start_period: str, end_period: str) -> pd.DataFrame:
    """Filter well data between inclusive period bounds provided as YYYY-MM strings."""
    start_ts = pd.Period(start_period).to_timestamp(how="start")
    end_ts = pd.Period(end_period).to_timestamp(how="end")
    # Filtrar valores menores a 0.1 además del periodo
    return well_df[
        (well_df["fecha"] >= start_ts) & 
        (well_df["fecha"] <= end_ts) & 
        (well_df["produccion"] >= 0.1)  # Cambio aquí: filtrar < 0.1
    ].copy()


def build_decline_plot(
    well_df: pd.DataFrame,
    fit_subset: pd.DataFrame,
    saved_fit: Optional[Dict[str, float]],
    preview_fit: Optional[Dict[str, float]],
    extrap_months: int,
) -> go.Figure:
    """Compose Plotly figure with historical data, fit and preview results."""
    fig = go.Figure()
    base_date = well_df["fecha"].min()

    fig.add_trace(
        go.Scatter(
            x=well_df["fecha"],
            y=well_df["produccion"],
            mode="markers",
            name="Histórico",
            marker=dict(color="#1f77b4", size=8, symbol="circle"),
        )
    )

    if not fit_subset.empty:
        fig.add_trace(
            go.Scatter(
                x=fit_subset["fecha"],
                y=fit_subset["produccion"],
                mode="markers",
                name="Periodo de ajuste",
                marker=dict(color="#d62728", size=10, symbol="diamond"),
        )
    )

    def add_curve(fit: Dict[str, float], label: str, style: Dict[str, object]) -> None:
        start = pd.to_datetime(fit["data_start"])
        last_hist = well_df["fecha"].max()
        future_months = np.arange(
            0,
            months_between(start, [last_hist])[0] + extrap_months + 1,
            1.0,
        )
        dates = [start + pd.DateOffset(months=int(m)) for m in future_months]
        values = arps_decline(future_months, fit["qo"], fit["D"], fit["b"])

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="lines",
                name=label,
                line=style,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f} m³/d<extra>" + label + "</extra>",
            )
        )

    if saved_fit:
        add_curve(
            saved_fit,
            "Curva guardada",
            dict(color="#2ca02c", width=3),
        )

    if preview_fit:
        add_curve(
            preview_fit,
            "Vista previa",
            dict(color="#ff7f0e", width=3, dash="dash"),
        )

    fig.update_layout(
        height=500,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Fecha",
        yaxis_title="Producción (m³/d)",
        yaxis=dict(range=[0, None]),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(family="San Francisco, Segoe UI, Helvetica Neue, Arial, sans-serif", size=15),
        hoverlabel=dict(bgcolor="#e3e7ed", font_size=14, font_family="San Francisco, Segoe UI"),
        legend_title_text="<b>Series</b>",
        legend_bgcolor="#f8fafc",
        legend_bordercolor="#c3d0e8",
        legend_borderwidth=2,
    )
    return fig


def compute_forecast_series(
    fit: Dict[str, float],
    start_date: pd.Timestamp,
    months: int,
    well_data: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Generate forecast DataFrame for a single well."""
    forecast_dates = pd.date_range(start=start_date, periods=months, freq="MS")
    base_date = pd.to_datetime(fit["data_start"])
    t_values = months_between(base_date, forecast_dates)
    forecast = arps_decline(t_values, fit["qo"], fit["D"], fit["b"])

    if method == "ajustada":
        last_actual = well_data["fecha"].max()
        last_actual_value = well_data.loc[well_data["fecha"] == last_actual, "produccion"].iloc[-1]
        t_last = months_between(base_date, [last_actual])[0]
        fitted_last = arps_decline(np.array([t_last]), fit["qo"], fit["D"], fit["b"])[0]
        if fitted_last > 0:
            factor = last_actual_value / fitted_last
            forecast = forecast * factor

    return pd.DataFrame(
        {
            "fecha": forecast_dates,
            "pozo": fit.get("pozo"),
            "pronostico": forecast,
        }
    )


def batch_fit_all_wells(wells: List[str]) -> Tuple[List[str], List[str]]:
    """Fit decline curves for all wells using full historical range."""
    successes: List[str] = []
    failures: List[str] = []

    for well in wells:
        well_df = get_well_dataset(well)
        if len(well_df) < 3:
            failures.append(well)
            continue

        fit_result = fit_decline_curve(well_df)
        if fit_result is None:
            failures.append(well)
            continue

        fit_result["pozo"] = well
        st.session_state.fits[well] = fit_result

        periods = build_period_options(well_df)
        if periods:
            st.session_state.fit_start[well] = periods[0]
            st.session_state.fit_end[well] = periods[-1]

        successes.append(well)

    st.session_state.preview = None
    return successes, failures


def render_column_mapping_prompt() -> bool:
    """Display UI to let the user map CSV columns when detection fails."""
    pending = st.session_state.get("pending_mapping")
    if not pending:
        return False

    st.warning("Selecciona las columnas que corresponden a fecha, pozo y producción.")
    st.markdown("<div class='metric-card' style='background:#fffbe6;'><b>Mapeo manual de columnas</b></div>", unsafe_allow_html=True)

    columns: List[str] = pending.get("columns", [])
    preview: Optional[pd.DataFrame] = st.session_state.get("pending_mapping_preview")

    if not columns:
        st.error("El archivo no contiene columnas disponibles para mapear.")
        return True

    default_date = (
        guess_date_column(preview)
        if isinstance(preview, pd.DataFrame) and not preview.empty
        else None
    )
    default_well = guess_well_column(columns)
    default_prod = guess_production_column(columns)

    def option_index(option: Optional[str]) -> int:
        if option and option in columns:
            return columns.index(option)
        return 0 if columns else -1

    with st.form("column_mapping_form"):
        st.markdown(f"<div style='font-size:1.1em; color:#4f8cff;'><b>Archivo:</b> {pending.get('source_name', 'Archivo cargado')}</div>", unsafe_allow_html=True)
        if isinstance(preview, pd.DataFrame) and not preview.empty:
            st.dataframe(preview.head(20), use_container_width=True)
        date_col = st.selectbox(
            "Columna de fecha",
            options=columns,
            index=option_index(default_date),
            key="mapping_date_col",
        )
        well_col = st.selectbox(
            "Columna de pozo",
            options=columns,
            index=option_index(default_well),
            key="mapping_well_col",
        )
        prod_col = st.selectbox(
            "Columna de producción",
            options=columns,
            index=option_index(default_prod),
            key="mapping_prod_col",
        )
        submitted = st.form_submit_button("Confirmar columnas")

    if submitted:
        if len({date_col, well_col, prod_col}) < 3:
            st.error("Cada columna debe ser diferente.")
            return True

        mapping_tuple = (date_col, well_col, prod_col)
        try:
            df_raw, df_long, df_wide = load_processed_data(
                pending["file_bytes"],
                pending["file_id"],
                mapping_tuple,
            )
        except Exception as exc:
            st.error(f"No se pudo procesar el archivo con la selección indicada: {exc}")
            return True

        reset_state_with_data(
            df_raw,
            df_long,
            df_wide,
            pending.get("source_name", "Archivo cargado"),
            pending["file_id"],
        )
        st.session_state.pending_mapping = None
        st.session_state.pending_mapping_preview = None
        st.success("Columnas asignadas correctamente.")

        return False

    return True


def render_dataset_summary() -> None:
    """Display basic info about the loaded dataset."""
    df_long: Optional[pd.DataFrame] = st.session_state.df
    if df_long is None or df_long.empty:
        st.caption("No hay datos cargados.")
        return

    wells = df_long["pozo"].nunique()
    start_date = df_long["fecha"].min()
    end_date = df_long["fecha"].max()
    records = len(df_long)

    # Modern metric cards
    cols = st.columns(4)
    with cols[0]:
        st.markdown("<div class='metric-card'><div class='metric-title'>Pozos</div><div class='metric-value'>%d</div></div>" % wells, unsafe_allow_html=True)
    with cols[1]:
        st.markdown("<div class='metric-card'><div class='metric-title'>Registros</div><div class='metric-value'>%d</div></div>" % records, unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Inicio</div><div class='metric-value'>{start_date.date()}</div></div>", unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Fin</div><div class='metric-value'>{end_date.date()}</div></div>", unsafe_allow_html=True)
    st.markdown("<br>")


def handle_file_upload(uploaded_file) -> None:
    """Process a file uploaded via Streamlit uploader."""
    if uploaded_file is None:
        return

    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.last_uploaded_file_id == file_id:
        st.info("El archivo ya ha sido cargado previamente.")
        return

    file_bytes = uploaded_file.getvalue()
    try:
        df_raw, df_long, df_wide = load_processed_data(file_bytes, file_id, None)
    except ColumnMappingRequired as exc:
        try:
            df_preview = read_csv_with_fallbacks(file_bytes).head(20)
        except Exception:
            df_preview = None
        st.session_state.pending_mapping = {
            "columns": exc.columns,
            "file_bytes": file_bytes,
            "file_id": file_id,
            "source_name": uploaded_file.name,
        }
        st.session_state.pending_mapping_preview = df_preview
        st.warning(
            "No se identificaron las columnas del archivo. Selecciónalas en la sección principal para continuar."
        )
        return
    except Exception as exc:
        st.error(str(exc))
        return

    reset_state_with_data(df_raw, df_long, df_wide, uploaded_file.name, file_id)
    st.success(f"Datos cargados desde {uploaded_file.name}.")


def handle_default_dataset(selection: Optional[str]) -> None:
    """Load a dataset from the Datos directory."""
    if not selection:
        return

    path = DATA_DIR / selection
    if not path.exists():
        st.error("No se encontró el archivo seleccionado.")
        return

    file_bytes = path.read_bytes()
    file_id = f"default-{selection}-{path.stat().st_mtime}"
    try:
        df_raw, df_long, df_wide = load_processed_data(file_bytes, file_id, None)
    except ColumnMappingRequired as exc:
        try:
            df_preview = read_csv_with_fallbacks(file_bytes).head(20)
        except Exception:
            df_preview = None
        st.session_state.pending_mapping = {
            "columns": exc.columns,
            "file_bytes": file_bytes,
            "file_id": file_id,
            "source_name": selection,
        }
        st.session_state.pending_mapping_preview = df_preview
        st.warning(
            "El dataset seleccionado necesita que asignes manualmente las columnas. Hazlo en la sección principal."
        )
        return
    reset_state_with_data(df_raw, df_long, df_wide, selection, file_id)
    st.success(f"Datos de ejemplo '{selection}' cargados.")


def render_decline_analysis() -> None:
    """UI logic for decline curve fitting and visualization."""
    df_long: Optional[pd.DataFrame] = st.session_state.df
    if df_long is None or df_long.empty:
        st.info("Carga datos de producción para iniciar el análisis.")
        return

    wells = sorted(df_long["pozo"].unique())

    if st.button(
        "Ajustar todos los pozos",
        help="Calcula y guarda automáticamente el ajuste de declinación para cada pozo utilizando todo su historial.",
    ):
        successes, failures = batch_fit_all_wells(wells)
        if successes:
            st.success(
                f"Ajustes generados para {len(successes)} pozos."
            )
        if failures:
            st.warning(
                f"No se pudo ajustar a {len(failures)} pozos (insuficientes datos o error): {', '.join(failures)}."
            )

    selected_well = st.selectbox("Selecciona un pozo", wells)
    well_df = get_well_dataset(selected_well)

    periods = build_period_options(well_df)
    if len(periods) < 1:
        st.warning("No hay fechas válidas para el pozo seleccionado.")
        return

    defaults = (
        st.session_state.fit_start.get(selected_well, periods[0]),
        st.session_state.fit_end.get(selected_well, periods[-1]),
    )

    start_value, end_value = st.select_slider(
        "Periodo de ajuste (mensual)",
        options=periods,
        value=defaults,
        format_func=lambda x: pd.Period(x).strftime("%Y-%m"),
    )

    if pd.Period(start_value) > pd.Period(end_value):
        st.warning("La fecha inicial debe ser anterior a la final.")
        return

    st.session_state.fit_start[selected_well] = start_value
    st.session_state.fit_end[selected_well] = end_value

    fit_subset = filter_by_period(well_df, start_value, end_value)
    if len(fit_subset) < 3:
        st.warning("Se requieren al menos 3 puntos válidos dentro del periodo seleccionado.")
        return

    saved_fit = st.session_state.fits.get(selected_well)
    preview_fit = st.session_state.preview if st.session_state.preview and st.session_state.preview.get("pozo") == selected_well else None

    cols = st.columns(3)
    with cols[0]:
        extrap_months = st.number_input(
            "Meses de extrapolación",
            min_value=6,
            max_value=120,
            value=st.session_state.extrap_months,
            step=6,
        )
        st.session_state.extrap_months = extrap_months

    with cols[1]:
        if st.button(
            "Vista previa del ajuste",
            type="secondary",
            help="Calcula temporalmente la curva para revisar resultados antes de guardar.",
        ):
            fit_result = fit_decline_curve(fit_subset)
            if fit_result is None:
                st.error("No se pudo ajustar la curva para el periodo seleccionado.")
            else:
                fit_result["pozo"] = selected_well
                st.session_state.preview = fit_result
                st.success("Vista previa calculada.")

    with cols[2]:
        if st.button(
            "Guardar ajuste",
            type="primary",
            help="Almacena los parámetros de la curva ajustada para usarlos en pronósticos y controles.",
        ):
            fit_result = fit_decline_curve(fit_subset)
            if fit_result is None:
                st.error("No se pudo ajustar la curva para el periodo seleccionado.")
            else:
                fit_result["pozo"] = selected_well
                st.session_state.fits[selected_well] = fit_result
                st.session_state.preview = None
                st.success(f"Ajuste guardado para {selected_well}.")

    saved_fit = st.session_state.fits.get(selected_well)
    preview_fit = st.session_state.preview if st.session_state.preview and st.session_state.preview.get("pozo") == selected_well else None

    fig = build_decline_plot(
        well_df,
        fit_subset,
        saved_fit,
        preview_fit,
        st.session_state.extrap_months,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detalle individual del ajuste del pozo seleccionado ocultado por preferencia del usuario.
    # Si quieres volver a mostrarlo, restaurar el bloque que generaba las tarjetas de métricas para `saved_fit`.

    if st.session_state.fits:
        fits_df = pd.DataFrame.from_records(
            [
                {
                    "Pozo": well,
                    "qo (m³/d)": fit_data["qo"],
                    "D (1/mes)": fit_data["D"],
                    "b": fit_data["b"],
                    "MSE": fit_data["mse"],
                    "Puntos válidos": fit_data.get("n_points", "N/A"),
                    "Inicio ajuste": fit_data["fit_start"],
                    "Fin ajuste": fit_data["fit_end"],
                }
                for well, fit_data in sorted(st.session_state.fits.items())
            ]
        )

        # Mostrar 'qo' con 2 decimales en la tabla de parámetros
        if "qo (m³/d)" in fits_df.columns:
            try:
                fits_df["qo (m³/d)"] = fits_df["qo (m³/d)"].astype(float).round(2)
            except Exception:
                # Si no es posible convertir/round, dejar como está
                pass

        st.subheader("Parámetros de ajustes guardados")
        st.dataframe(fits_df, use_container_width=True)

        csv_bytes = fits_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar parámetros (CSV)",
            csv_bytes,
            file_name="well_parameters.csv",
            mime="text/csv",
        )


def render_forecast_tab() -> None:
    """Forecasting workflow."""
    df_long: Optional[pd.DataFrame] = st.session_state.df
    if not st.session_state.fits:
        st.info("Guarda al menos un ajuste de declinación para generar pronósticos.")
        return
    if df_long is None or df_long.empty:
        st.warning("No hay datos para procesar.")
        return

    wells = sorted(st.session_state.fits.keys())
    option_all = "Todos los pozos"
    options = [option_all] + wells

    selected_options = st.multiselect(
        "Selecciona pozos",
        options=options,
        default=options,
    )

    if not selected_options:
        st.info("Selecciona al menos un pozo.")
        return

    selected_wells = wells if option_all in selected_options else selected_options

    st.info(f"Se han seleccionado {len(selected_wells)} pozos para el pronóstico.")

    max_date = df_long["fecha"].max()
    next_month = (max_date + pd.offsets.MonthBegin(1)).to_pydatetime().date()
    start_date_input = st.date_input(
        "Fecha de inicio del pronóstico",
        value=next_month,
        min_value=max_date.date(),
    )
    forecast_months = st.slider(
        "Meses a pronosticar",
        min_value=1,
        max_value=60,
        value=st.session_state.forecast_months,
    )
    st.session_state.forecast_months = forecast_months

    method = st.radio(
        "Método de proyección",
        options=("continuo", "ajustada"),
        format_func=lambda x: "Continuar curva" if x == "continuo" else "Desde último valor",
        horizontal=True,
    )
    st.session_state["forecast_method"] = method

    start_timestamp = pd.Timestamp(start_date_input)
    forecast_frames: List[pd.DataFrame] = []
    missing_fits: List[str] = []

    for well in selected_wells:
        fit = st.session_state.fits.get(well)
        if not fit:
            missing_fits.append(well)
            continue

        well_data = get_well_dataset(well)
        fit_with_name = fit.copy()
        fit_with_name["pozo"] = well
        forecast_df = compute_forecast_series(
            fit_with_name,
            start_timestamp,
            forecast_months,
            well_data,
            method,
        )
        forecast_frames.append(forecast_df)

    if missing_fits:
        st.warning("No hay ajustes guardados para: " + ", ".join(missing_fits))

    if not forecast_frames:
        st.info("No se generaron pronósticos.")
        return

    forecast_df = pd.concat(forecast_frames, ignore_index=True)

    history_window_start = (start_timestamp - pd.DateOffset(months=24)).normalize()
    history_df = (
        df_long[df_long["pozo"].isin(selected_wells)]
        .copy()
    )
    history_df = history_df[history_df["fecha"] >= history_window_start]

    agg_history = (
        history_df.groupby("fecha")["produccion"]
        .sum()
        .reset_index()
        .rename(columns={"produccion": "historico"})
    )

    agg_forecast = (
        forecast_df.groupby("fecha")["pronostico"]
        .sum()
        .reset_index()
    )

    chart = go.Figure()
    if not agg_history.empty:
        chart.add_trace(
            go.Scatter(
                x=agg_history["fecha"],
                y=agg_history["historico"],
                mode="lines+markers",
                name="Histórico",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8, symbol="circle"),
            )
        )

    chart.add_trace(
        go.Scatter(
            x=agg_forecast["fecha"],
            y=agg_forecast["pronostico"],
            mode="lines+markers",
            name="Pronóstico",
            line=dict(color="#ff7f0e", dash="dash", width=3),
            marker=dict(size=8, symbol="diamond"),
        )
    )

    chart.update_layout(
        height=450,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Fecha",
        yaxis_title="Producción total (m³/d)",
        yaxis=dict(range=[0, None]),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(family="San Francisco, Segoe UI, Helvetica Neue, Arial, sans-serif", size=15),
        hoverlabel=dict(bgcolor="#e3e7ed", font_size=14, font_family="San Francisco, Segoe UI"),
        legend_title_text="<b>Series</b>",
        legend_bgcolor="#f8fafc",
        legend_bordercolor="#c3d0e8",
        legend_borderwidth=2,
    )

    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Pronóstico detallado")
    st.dataframe(forecast_df, use_container_width=True)
    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar pronóstico (CSV)",
        csv_bytes,
        file_name="forecast.csv",
        mime="text/csv",
    )


def render_control_tab() -> None:
    """Control comparison tab."""
    if not st.session_state.fits:
        st.info("Guarda ajustes de declinación antes de comparar con controles.")
        return

    control_file = st.file_uploader(
        "Archivo de control (CSV)",
        type="csv",
        key="control_uploader",
    )
    threshold = st.slider(
        "Umbral de alerta (%)",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        step=0.01,
        format="%.2f",
    )

    if control_file is None:
        st.info("Carga un archivo con controles para continuar.")
        return

    try:
        control_raw = read_csv_with_fallbacks(control_file.getvalue())
    except Exception as exc:
        st.error(str(exc))
        return

    try:
        date_col, well_col, value_col = detect_columns(control_raw)
    except ValueError as err:
        st.error(str(err))
        return

    control_df = control_raw[[date_col, well_col, value_col]].rename(
        columns={date_col: "fecha", well_col: "pozo", value_col: "control"}
    )
    control_df["fecha"] = pd.to_datetime(control_df["fecha"], dayfirst=True, errors="coerce")
    control_df["control"] = coerce_numeric(control_df["control"]).round(2)
    control_df = control_df.dropna(subset=["fecha", "control"])
    control_df["pozo"] = control_df["pozo"].astype(str).str.strip()

    method = st.session_state.get("forecast_method", "continuo")

    results: List[Dict[str, object]] = []
    for _, row in control_df.iterrows():
        well = row["pozo"]
        fit = st.session_state.fits.get(well)
        if not fit:
            results.append(
                {
                    "Pozo": well,
                    "Fecha": row["fecha"].date(),
                    "Control": row["control"],
                    "Pronóstico": np.nan,
                    "Delta": np.nan,
                    "Delta %": np.nan,
                    "Alerta": "Sin ajuste",
                }
            )
            continue

        base_date = pd.to_datetime(fit["data_start"])
        t_val = months_between(base_date, [row["fecha"]])[0]
        forecast_val = np.round(arps_decline(np.array([t_val]), fit["qo"], fit["D"], fit["b"])[0], 2)

        if method == "ajustada":
            well_data = get_well_dataset(well)
            last_actual = well_data["fecha"].max()
            last_actual_value = well_data.loc[well_data["fecha"] == last_actual, "produccion"].iloc[-1]
            t_last = months_between(base_date, [last_actual])[0]
            fitted_last = arps_decline(np.array([t_last]), fit["qo"], fit["D"], fit["b"])[0]
            if fitted_last > 0:
                factor = last_actual_value / fitted_last
                forecast_val *= factor

        delta = row["control"] - forecast_val
        delta_pct = (delta / forecast_val) * 100 if forecast_val else np.nan
        alert = abs(delta_pct) > threshold if not np.isnan(delta_pct) else False

        results.append(
            {
                "Pozo": well,
                "Fecha": row["fecha"].date(),
                "Control": row["control"],
                "Pronóstico": forecast_val,
                "Delta": delta,
                "Delta %": delta_pct,
                "Alerta": "ALERTA" if alert else "",
            }
        )

    if not results:
        st.warning("No se encontraron registros válidos en el archivo de control.")
        return

    result_df = pd.DataFrame(results)

    # Formatear columnas numéricas a 2 decimales para presentación
    numeric_cols = [col for col in ("Control", "Pronóstico", "Delta", "Delta %") if col in result_df.columns]
    if not result_df.empty and numeric_cols:
        result_df[numeric_cols] = result_df[numeric_cols].round(2)

    def highlight_alerts(row: pd.Series) -> List[str]:
        color = "background-color: #ffe6e6" if row.get("Alerta") else ""
        return [color] * len(row)

    st.dataframe(
        result_df.style.apply(highlight_alerts, axis=1),
        use_container_width=True,
    )

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar comparación (CSV)",
        csv_bytes,
        file_name="control_comparison.csv",
        mime="text/csv",
    )


# --- Main application ------------------------------------------------------------------------
def main() -> None:
    init_session_state()

    with st.sidebar:
        st.header("Datos de producción")
        uploader = st.file_uploader("Subir archivo CSV", type="csv")
        handle_file_upload(uploader)

        default_files = list_default_datasets()
        if default_files:
            selected_default = st.selectbox(
                "Dataset de ejemplo",
                options=[""] + [p.name for p in default_files],
                format_func=lambda x: "Selecciona..." if x == "" else x,
            )
            if st.button(
                "Cargar dataset de ejemplo",
                help="Reemplaza los datos actuales por el archivo seleccionado en la carpeta Datos.",
            ) and selected_default:
                handle_default_dataset(selected_default)

        if st.button(
            "Reiniciar datos",
            help="Limpia el dataset cargado y todos los ajustes guardados en la sesión actual.",
        ):
            for key in SESSION_DEFAULTS:
                st.session_state[key] = SESSION_DEFAULTS[key] if not isinstance(SESSION_DEFAULTS[key], dict) else {}
            st.success("Estado reiniciado.")

    st.title("📉 Análisis de Declinación y Pronóstico de Producción")
    st.markdown("<hr style='border: none; height: 2px; background: linear-gradient(90deg,#4f8cff,#38c6ff); margin-bottom: 1em;'>", unsafe_allow_html=True)

    if render_column_mapping_prompt():
        return

    render_dataset_summary()

    # Eliminado <br> en encabezados y tabs
    tabs = st.tabs([
        "📉 Declinación",
        "🔮 Pronóstico",
        "📋 Control"
    ])
    st.markdown("<hr style='border: none; height: 2px; background: linear-gradient(90deg,#4f8cff,#38c6ff); margin-bottom: 1em;'>", unsafe_allow_html=True)
    with tabs[0]:
        st.markdown("<div style='text-align:center; font-size:1.3em; color:#4f8cff; font-weight:600;'>📉 Análisis de Declinación</div>", unsafe_allow_html=True)
        render_decline_analysis()
    with tabs[1]:
        st.markdown("<div style='text-align:center; font-size:1.3em; color:#38c6ff; font-weight:600;'>🔮 Pronóstico de Producción</div>", unsafe_allow_html=True)
        render_forecast_tab()
    with tabs[2]:
        st.markdown("<div style='text-align:center; font-size:1.3em; color:#4f8cff; font-weight:600;'>📋 Comparación de Control</div>", unsafe_allow_html=True)
        render_control_tab()


if __name__ == "__main__":
    main()
