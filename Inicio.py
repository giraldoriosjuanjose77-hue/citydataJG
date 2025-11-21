"""
streamlit_app_datetime_fallback.py

App Streamlit completa y segura respecto a versiones de Streamlit:
- Lee CSV exportados desde Grafana/Influx (detecta encoding y delimiter).
- Detecta y parsea la columna de tiempo (soporta DD/MM/YYYY HH:MM).
- Convierte timestamps a datetime.datetime "naive" y proporciona inputs de rango de tiempo
  usando st.datetime_input si existe; sino usa st.date_input + st.time_input como fallback.
- Normaliza columnas temperatura/humedad y muestra gráfica / estadísticas.
- Muestra mensajes claros de error/adv.
"""
import io
import csv
import gzip
import re
from datetime import datetime, time as dtime, date as ddate
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="TempHum — DateInput Fallback", layout="wide", page_icon="📈")

# -------------------------
# Helpers: decoding, delimiter, header
# -------------------------
def try_decode_bytes(raw: bytes) -> Tuple[Optional[str], Optional[str]]:
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1252", "latin1"]
    for enc in encodings:
        try:
            return raw.decode(enc), enc
        except Exception:
            continue
    try:
        return raw.decode("latin1", errors="replace"), "latin1-replace"
    except Exception:
        return None, None

def detect_delimiter(sample: str) -> str:
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample[:8192])
        return dialect.delimiter
    except Exception:
        commas = sample.count(",")
        semis = sample.count(";")
        return ";" if semis > commas else ","

def find_header_line(lines):
    pattern = re.compile(r'\b(time|timestamp|_time|fecha|date|temperature|temp|humidity|humedad|device|location|lat|lon)\b', flags=re.IGNORECASE)
    for i, line in enumerate(lines[:200]):
        parts = [p.strip().strip('"') for p in re.split(r'[;,]', line)]
        if any(pattern.search(p) for p in parts) and any(re.search(r'[A-Za-z]', p) for p in parts):
            return i
    return None

# -------------------------
# Robust CSV reader
# -------------------------
def read_csv_robust(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if not raw:
        return pd.DataFrame()
    # gzip detection
    if isinstance(raw, (bytes, bytearray)) and len(raw) >= 2 and raw[0] == 0x1f and raw[1] == 0x8b:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                raw = gz.read()
            st.info("Archivo gzip detectado y descomprimido.")
        except Exception:
            st.error("Falló descompresión gzip.")
            return pd.DataFrame()
    if isinstance(raw, bytes):
        text, enc = try_decode_bytes(raw)
        if text is None:
            st.error("No se pudo decodificar el archivo con encodings comunes.")
            return pd.DataFrame()
        st.info(f"Encoding detectado: {enc}")
    else:
        text = str(raw)
        st.info("Archivo leído como texto.")
    lines = text.splitlines()
    if not lines:
        st.error("Archivo vacío.")
        return pd.DataFrame()
    sample = "\n".join(lines[:20])
    delim = detect_delimiter(sample)
    header_idx = find_header_line(lines)
    if header_idx is None:
        header_idx = 0
    try:
        df = pd.read_csv(io.StringIO(text), header=header_idx, sep=delim)
    except Exception as e:
        st.error(f"Error leyendo CSV con pandas: {e}")
        return pd.DataFrame()
    # drop metadata columns from Influx/Grafana if present
    cols_to_drop = [c for c in df.columns if isinstance(c, str) and (c.strip().startswith('#') or c.lower().startswith('result') or c.lower().startswith('table') or 'unnamed' in c.lower())]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')
    df.columns = [str(c).strip() for c in df.columns]
    return df

# -------------------------
# Convert to naive python datetime
# -------------------------
def to_naive_datetime(ts) -> Optional[datetime]:
    try:
        ts_pd = pd.to_datetime(ts)
        py = ts_pd.to_pydatetime()
        if getattr(py, "tzinfo", None) is not None:
            py = py.replace(tzinfo=None)
        return py
    except Exception:
        return None

# -------------------------
# Fallback datetime input (use date_input+time_input if datetime_input missing)
# -------------------------
def datetime_range_sidebar(label_from: str, default_start: datetime, default_end: datetime, min_dt: datetime, max_dt: datetime) -> Tuple[datetime, datetime]:
    """
    Returns (start_datetime, end_datetime). Uses st.sidebar.datetime_input if available,
    otherwise uses date_input + time_input to construct datetimes (compatible with older Streamlit).
    """
    # ensure all args are naive datetimes
    def ensure_naive(dt):
        if dt is None:
            return None
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.replace(tzinfo=None)
        return dt

    default_start = ensure_naive(default_start)
    default_end = ensure_naive(default_end)
    min_dt = ensure_naive(min_dt)
    max_dt = ensure_naive(max_dt)

    # If Streamlit has datetime_input, prefer it (newer versions)
    if hasattr(st.sidebar, "datetime_input"):
        try:
            s = st.sidebar.datetime_input(label_from, value=default_start, min_value=min_dt, max_value=max_dt)
            e = st.sidebar.datetime_input("Hasta", value=default_end, min_value=min_dt, max_value=max_dt)
            # ensure python datetimes (they may be date objects if user picks only date)
            if isinstance(s, ddate) and isinstance(e, ddate):
                s = datetime.combine(s, dtime.min)
                e = datetime.combine(e, dtime.max)
            return s, e
        except Exception as exc:
            st.warning("datetime_input no disponible; usando fallback de fecha+hora.")
    # Fallback: date + time inputs
    # Date inputs
    ds = st.sidebar.date_input("Desde (fecha)", value=default_start.date() if default_start else min_dt.date())
    ts = st.sidebar.time_input("Desde (hora)", value=default_start.time() if default_start else dtime(0, 0))
    de = st.sidebar.date_input("Hasta (fecha)", value=default_end.date() if default_end else max_dt.date())
    te = st.sidebar.time_input("Hasta (hora)", value=default_end.time() if default_end else dtime(23, 59))
    start_dt = datetime.combine(ds, ts)
    end_dt = datetime.combine(de, te)
    # clamp to min/max
    if start_dt < min_dt:
        start_dt = min_dt
    if end_dt > max_dt:
        end_dt = max_dt
    return start_dt, end_dt

# -------------------------
# Main app flow
# -------------------------
st.title("TempHum — Soporte para versiones de Streamlit sin datetime_input")

uploaded_file = st.file_uploader("Sube el CSV (joinbyfield o export de Grafana/Influx)", type=["csv", "txt", "gz"])
if uploaded_file is None:
    st.info("Sube el CSV para comenzar.")
    st.stop()

df_raw = read_csv_robust(uploaded_file)
if df_raw.empty:
    st.stop()

# Normalize column names (lowercase for detection)
cols_map = {c: c.lower() for c in df_raw.columns}
df = df_raw.rename(columns=cols_map)
st.write("Columnas detectadas:", list(df.columns))

# Detect and parse time column (support dayfirst for DD/MM/YYYY)
time_candidates = [c for c in df.columns if c in ('time', '_time', 'timestamp', 'date', 'fecha')]
chosen_time_col = time_candidates[0] if time_candidates else None
if not chosen_time_col:
    # infer by content
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(10).tolist()
        parsed = 0
        for s in sample:
            try:
                pd.to_datetime(s, dayfirst=True)
                parsed += 1
            except Exception:
                pass
        if parsed >= max(1, len(sample)//2):
            chosen_time_col = c
            break

if not chosen_time_col:
    st.error("No se encontró columna de tiempo. Asegúrate de que el CSV tenga 'time' o similar.")
    st.stop()

df[chosen_time_col] = pd.to_datetime(df[chosen_time_col].astype(str), errors='coerce', dayfirst=True)
na_count = int(df[chosen_time_col].isna().sum())
if na_count > 0:
    st.warning(f"{na_count} filas con tiempo inválido serán descartadas (se muestran algunos ejemplos).")
    st.write(df[df[chosen_time_col].isna()].head(5))

df = df[df[chosen_time_col].notna()].copy()
df = df.rename(columns={chosen_time_col: "Time"})
df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
df = df.set_index('Time')

# Determine min/max and default_start (as naive datetimes)
min_ts = df.index.min()
max_ts = df.index.max()
min_dt = to_naive_datetime(min_ts)
max_dt = to_naive_datetime(max_ts)
if min_dt is None or max_dt is None:
    st.error("No se pudo determinar rango de tiempo válido.")
    st.stop()

default_start_pd = max_ts - pd.Timedelta(hours=6) if (max_ts - min_ts) > pd.Timedelta(hours=1) else min_ts
default_start = to_naive_datetime(default_start_pd) or min_dt
default_end = max_dt

# Use robust datetime input (fallback when needed)
start_dt, end_dt = datetime_range_sidebar("Desde", default_start, default_end, min_dt, max_dt)
if start_dt is None or end_dt is None:
    st.error("Rango de tiempo inválido.")
    st.stop()
if start_dt > end_dt:
    st.error("La fecha 'Desde' es posterior a 'Hasta'. Corrige el rango.")
    st.stop()

# Filter dataframe
df_filtered = df[(df.index >= pd.to_datetime(start_dt)) & (df.index <= pd.to_datetime(end_dt))].copy()
if df_filtered.empty:
    st.error("No hay datos en el rango seleccionado.")
    st.stop()

# Detect temp/hum columns
temp_col = None
hum_col = None
for c in df_filtered.columns:
    lc = c.lower()
    if temp_col is None and ('temp' in lc):
        temp_col = c
    if hum_col is None and ('hum' in lc):
        hum_col = c

# Heuristic: numeric ranges
if temp_col is None or hum_col is None:
    numeric_cols = [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])]
    for c in numeric_cols:
        vals = pd.to_numeric(df_filtered[c].astype(str).str.replace(',', '.'), errors='coerce')
        med = vals.median(skipna=True)
        if pd.notna(med):
            if -50 < med < 60 and temp_col is None:
                temp_col = c
            elif 0 <= med <= 100 and hum_col is None:
                hum_col = c

# Normalize names
rename_map = {}
if temp_col:
    rename_map[temp_col] = 'temperatura'
if hum_col:
    rename_map[hum_col] = 'humedad'
if rename_map:
    df_filtered = df_filtered.rename(columns=rename_map)
    st.success(f"Columnas renombradas: {rename_map}")

# Convert numeric strings
for col in ['temperatura', 'humedad']:
    if col in df_filtered.columns:
        df_filtered[col] = pd.to_numeric(df_filtered[col].astype(str).str.replace(',', '.'), errors='coerce')

# UI: tabs
tab1, tab2 = st.tabs(["📈 Gráfica", "📊 Estadísticas"])

with tab1:
    st.subheader("Gráfica")
    plot_cols = [c for c in ['temperatura', 'humedad'] if c in df_filtered.columns]
    if plot_cols:
        st.line_chart(df_filtered[plot_cols])
    else:
        st.write("No se detectaron columnas temperatura/humedad para graficar.")

with tab2:
    st.subheader("Estadísticas")
    for col in ['temperatura', 'humedad']:
        if col in df_filtered.columns:
            s = df_filtered[col].dropna()
            if not s.empty:
                st.metric(f"{col.capitalize()} Máximo", f"{s.max():.2f}")
                st.metric(f"{col.capitalize()} Mínimo", f"{s.min():.2f}")
                st.metric(f"{col.capitalize()} Media", f"{s.mean():.2f}")
                st.metric("Desviación estándar", f"{s.std(ddof=0):.2f}")
            else:
                st.write(f"No hay datos válidos en {col}.")

st.markdown("---")
st.download_button("Descargar CSV procesado (UTF-8)", data=df_filtered.reset_index().to_csv(index=False), file_name="temphum_procesado.csv", mime="text/csv")
