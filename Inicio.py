"""
streamlit_app_complete.py

App Streamlit completa y lista para usar:
- Lee CSV exportados desde Grafana/Influx (detecta encoding y delimitador).
- Limpia metadatos (#group, #datatype, result, table).
- Detecta la fila de encabezado real.
- Parsea timestamps (soporta DD/MM/YYYY HH:MM con dayfirst=True) y convierte a datetime.datetime "naive"
  antes de pasarlos a st.datetime_input para evitar StreamlitAPIException.
- Detecta columnas de temperatura/humedad y las normaliza a 'temperatura' y 'humedad'.
- Detecta ubicación del sensor por prioridad: lat/lon en CSV -> columna 'location' geocodificada -> geolocalización por IP -> entrada manual.
- Muestra mapa con la ubicación detectada, gráficas, estadísticas (max, min, mean, std) y permite descargar CSV procesado.
- No requiere tokens; la geocodificación usa Nominatim (respeta sus límites). Hay cache local .geocode_cache.json.

Uso:
    streamlit run streamlit_app_complete.py

No compartas tokens ni credenciales. Si tu CSV tiene formatos especiales, copia los ejemplos problemáticos que muestre la app y te doy la adaptación.
"""
import io
import gzip
import json
import re
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import requests
import streamlit as st

# ---------- Config ----------
st.set_page_config(page_title="TempHum — App completa", layout="wide", page_icon="📍")
GEOCODE_CACHE_FILE = ".geocode_cache.json"
USER_AGENT = "TempHumApp/1.0 (contact: example@example.com)"  # cámbialo si quieres identificar peticiones

# ---------- Helpers: encoding, delimiter, header detection ----------
def try_decode_bytes(raw: bytes) -> Tuple[Optional[str], Optional[str]]:
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1252", "latin1"]
    for enc in encodings:
        try:
            text = raw.decode(enc)
            return text, enc
        except Exception:
            continue
    try:
        text = raw.decode("latin1", errors="replace")
        return text, "latin1-replace"
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

def find_header_line(lines, pattern=re.compile(r'\b(time|timestamp|_time|fecha|date|temperature|temp|humidity|humedad|device|location|lat|lon)\b', flags=re.IGNORECASE)):
    for i, line in enumerate(lines[:200]):
        parts = [p.strip().strip('"') for p in re.split(r'[;,]', line)]
        if any(pattern.search(p) for p in parts) and any(re.search(r'[A-Za-z]', p) for p in parts):
            return i
    for i, line in enumerate(lines):
        parts = [p.strip().strip('"') for p in re.split(r'[;,]', line)]
        if any(pattern.search(p) for p in parts) and any(re.search(r'[A-Za-z]', p) for p in parts):
            return i
    return None

# ---------- Helpers: reading and cleaning CSV ----------
def read_csv_robust(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if not raw:
        return pd.DataFrame()
    # gzip?
    if isinstance(raw, (bytes, bytearray)) and len(raw) >= 2 and raw[0] == 0x1f and raw[1] == 0x8b:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                raw = gz.read()
            st.info("Archivo gzip detectado y descomprimido.")
        except Exception as e:
            st.error(f"Error descomprimiendo gzip: {e}")
            return pd.DataFrame()
    if isinstance(raw, bytes):
        text, enc = try_decode_bytes(raw)
        if text is None:
            st.error("No se pudo decodificar el archivo con los encodings habituales.")
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
    # drop metadata columns typical from Influx/Grafana
    cols_to_drop = [c for c in df.columns if isinstance(c, str) and (c.strip().startswith('#') or c.lower().startswith('result') or c.lower().startswith('table') or 'unnamed' in c.lower())]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ---------- Helpers: datetime conversion for Streamlit inputs ----------
def _to_naive_pydatetime(ts) -> Optional[datetime]:
    try:
        ts_pd = pd.to_datetime(ts)
        py = ts_pd.to_pydatetime()
        if getattr(py, "tzinfo", None) is not None:
            py = py.replace(tzinfo=None)
        return py
    except Exception:
        return None

# ---------- Geocoding helpers (Nominatim) ----------
def load_geocode_cache():
    try:
        p = Path(GEOCODE_CACHE_FILE)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_geocode_cache(cache):
    try:
        Path(GEOCODE_CACHE_FILE).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def geocode_name(name: str, cache: dict) -> Optional[Tuple[float, float, str]]:
    if not name or not name.strip():
        return None
    key = name.strip().lower()
    if key in cache and cache[key].get("lat") and cache[key].get("lon"):
        e = cache[key]
        return e["lat"], e["lon"], e.get("display_name", key)
    try:
        params = {"q": name, "format": "json", "limit": 1}
        headers = {"User-Agent": USER_AGENT}
        r = requests.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers, timeout=8)
        if r.status_code == 200:
            arr = r.json()
            if isinstance(arr, list) and arr:
                it = arr[0]
                lat = float(it.get("lat"))
                lon = float(it.get("lon"))
                disp = it.get("display_name")
                cache[key] = {"lat": lat, "lon": lon, "display_name": disp}
                save_geocode_cache(cache)
                return lat, lon, disp
    except Exception:
        pass
    cache[key] = {}
    save_geocode_cache(cache)
    return None

def ip_geolocation() -> Optional[Tuple[float, float, str]]:
    try:
        r = requests.get("https://ipapi.co/json/", timeout=6)
        if r.status_code == 200:
            j = r.json()
            lat = j.get("latitude") or j.get("lat")
            lon = j.get("longitude") or j.get("lon")
            city = j.get("city")
            country = j.get("country_name")
            if lat and lon:
                return float(lat), float(lon), f"{city or ''} {country or ''}".strip()
    except Exception:
        pass
    return None

# ---------- UI ----------
st.title("TempHum — App completa (CSV → Gráfica, Estadísticas y Ubicación)")

uploaded_file = st.file_uploader("Sube CSV (joinbyfield o export de Grafana/Influx)", type=["csv", "txt", "gz"])
if uploaded_file is None:
    st.info("Sube el CSV para comenzar.")
    st.stop()

df_raw = read_csv_robust(uploaded_file)
if df_raw.empty:
    st.stop()

st.sidebar.header("Controles")

# normalize lowercase map for detection but preserve original names for display
orig_cols = list(df_raw.columns)
lower_map = {c: c.lower() for c in orig_cols}
df = df_raw.rename(columns=lower_map)

st.write("Columnas detectadas:")
st.write(list(df.columns))

# ---------- detect time column ----------
time_candidates = [c for c in df.columns if c in ('time', '_time', 'timestamp', 'date', 'fecha')]
chosen_time_col = time_candidates[0] if time_candidates else None
if not chosen_time_col:
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
            st.write(f"Columna de tiempo inferida: '{c}'")
            break

if not chosen_time_col:
    st.error("No se encontró columna de tiempo. Asegúrate de que tu CSV tenga una columna 'time' o similar.")
    st.stop()

# parse time (dayfirst=True to support DD/MM/YYYY)
df[chosen_time_col] = pd.to_datetime(df[chosen_time_col].astype(str), errors='coerce', dayfirst=True)
na_time = int(df[chosen_time_col].isna().sum())
if na_time > 0:
    st.warning(f"{na_time} filas con tiempo inválido serán descartadas. Ejemplos:")
    st.write(df[df[chosen_time_col].isna()].head(5))

df = df[df[chosen_time_col].notna()].copy()
df = df.rename(columns={chosen_time_col: "Time"})
df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
df = df.set_index('Time')

# ---------- datetime_input: convert to naive datetimes ----------
min_time_pd = df.index.min()
max_time_pd = df.index.max()
min_time = _to_naive_pydatetime(min_time_pd)
max_time = _to_naive_pydatetime(max_time_pd)
if min_time is None or max_time is None:
    st.error("No se pudo determinar rango de tiempo válido desde los datos.")
    st.stop()
default_start_pd = max_time_pd - pd.Timedelta(hours=6) if (max_time_pd - min_time_pd) > pd.Timedelta(hours=1) else min_time_pd
default_start = _to_naive_pydatetime(default_start_pd) or min_time

start = st.sidebar.datetime_input("Desde", value=default_start, min_value=min_time, max_value=max_time)
end = st.sidebar.datetime_input("Hasta", value=max_time, min_value=min_time, max_value=max_time)
if start is None or end is None:
    st.error("Rango de tiempo inválido en inputs.")
    st.stop()

# apply time filter
df_filtered = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))].copy()
if df_filtered.empty:
    st.error("No hay datos en el rango seleccionado.")
    st.stop()

# ---------- detect temp/hum columns ----------
temp_col = None
hum_col = None
for c in df_filtered.columns:
    lc = c.lower()
    if temp_col is None and ('temp' in lc):
        temp_col = c
    if hum_col is None and ('hum' in lc):
        hum_col = c

# If not found, heuristics on numeric columns
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

rename_map = {}
if temp_col:
    rename_map[temp_col] = 'temperatura'
if hum_col:
    rename_map[hum_col] = 'humedad'
if rename_map:
    df_filtered = df_filtered.rename(columns=rename_map)
    st.success(f"Columnas renombradas: {rename_map}")

# convert numeric with comma support
for col in ['temperatura', 'humedad']:
    if col in df_filtered.columns:
        df_filtered[col] = pd.to_numeric(df_filtered[col].astype(str).str.replace(',', '.'), errors='coerce')

# ---------- Location detection ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Ubicación del sensor")

lat_col = None
lon_col = None
for c in df.columns:
    if c in ('lat', 'latitude'):
        lat_col = c
    if c in ('lon', 'lng', 'longitude'):
        lon_col = c

location_col = None
for c in df.columns:
    if 'location' in c or 'place' in c:
        location_col = c
        break

detected_latlon = None
detected_label = None
map_df = None

geocode_cache = load_geocode_cache()

# priority 1: lat/lon in CSV
if lat_col and lon_col:
    lat_s = pd.to_numeric(df[lat_col], errors='coerce')
    lon_s = pd.to_numeric(df[lon_col], errors='coerce')
    valid = lat_s.notna() & lon_s.notna() & lat_s.between(-90, 90) & lon_s.between(-180, 180)
    if valid.any():
        coords = pd.DataFrame({'latitude': lat_s[valid].astype(float), 'longitude': lon_s[valid].astype(float)}).drop_duplicates().reset_index(drop=True)
        detected_latlon = (coords['latitude'].median(), coords['longitude'].median())
        detected_label = "Coordenadas desde CSV"
        map_df = coords

# priority 2: geocode location name column
if detected_latlon is None and location_col:
    loc_series = df[location_col].dropna().astype(str).str.strip()
    if not loc_series.empty:
        most = loc_series.mode().iloc[0] if not loc_series.mode().empty else loc_series.iloc[0]
        ge = geocode_name(most, geocode_cache)
        if ge:
            lat, lon, disp = ge
            detected_latlon = (lat, lon)
            detected_label = f"Geocodificado desde '{most}'"
            map_df = pd.DataFrame([{'latitude': lat, 'longitude': lon}])

# user options: use IP approx or manual
if detected_latlon:
    st.success(f"Ubicación detectada: {detected_label} ({detected_latlon[0]:.6f}, {detected_latlon[1]:.6f})")
else:
    if st.sidebar.button("Usar geolocalización por IP (aprox.)"):
        ipg = ip_geolocation()
        if ipg:
            lat, lon, city = ipg
            detected_latlon = (lat, lon)
            detected_label = f"IP: {city}"
            map_df = pd.DataFrame([{'latitude': lat, 'longitude': lon}])
            st.success(f"Ubicación aproximada por IP: {city} ({lat:.6f}, {lon:.6f})")
        else:
            st.sidebar.error("No fue posible obtener ubicación por IP.")

    st.sidebar.markdown("O ingresa coordenadas manualmente:")
    manual_lat = st.sidebar.text_input("Latitud", "")
    manual_lon = st.sidebar.text_input("Longitud", "")
    if st.sidebar.button("Usar ubicación manual"):
        try:
            latm = float(manual_lat)
            lonm = float(manual_lon)
            if -90 <= latm <= 90 and -180 <= lonm <= 180:
                detected_latlon = (latm, lonm)
                detected_label = "Ubicación manual"
                map_df = pd.DataFrame([{'latitude': latm, 'longitude': lonm}])
                st.success(f"Ubicación manual usada: ({latm:.6f}, {lonm:.6f})")
            else:
                st.sidebar.error("Coordenadas fuera de rango.")
        except Exception:
            st.sidebar.error("Latitud/longitud inválidas.")

# Show map or warning
st.subheader("Ubicación del sensor")
if detected_latlon:
    lat, lon = detected_latlon
    if map_df is None:
        map_df = pd.DataFrame([{'latitude': lat, 'longitude': lon}])
    st.map(map_df)
    st.write(f"**Ubicación:** {detected_label}")
    st.write(f"Lat: {lat:.6f}, Lon: {lon:.6f}")
else:
    st.warning("No se pudo detectar ubicación automáticamente. Usa la opción de IP o ingresa coordenadas manuales o añade lat/lon en el CSV.")

# ---------- Visualizaciones y estadísticas ----------
tab1, tab2, tab3 = st.tabs(["📈 Gráfica", "📊 Estadísticas", "📄 Datos / Descargar"])

with tab1:
    st.subheader("Gráfica")
    plot_cols = [c for c in ['temperatura', 'humedad'] if c in df_filtered.columns]
    if plot_cols:
        st.line_chart(df_filtered[plot_cols])
    else:
        st.write("No hay columnas detectadas para graficar.")

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

with tab3:
    st.subheader("Datos y descarga")
    st.dataframe(df_filtered.reset_index().head(500), height=300)
    csv_out = df_filtered.reset_index().to_csv(index=False)
    st.download_button("Descargar CSV procesado (UTF-8)", data=csv_out, file_name="temphum_procesado.csv", mime="text/csv")

st.markdown("---")
st.markdown("Consejos: si quieres que la ubicación se detecte automáticamente en el futuro, añade campos latitude y longitude en el payload que envía tu sensor al escribir en Influx (o añade tags `location` consistentes que podamos geocodificar).")
