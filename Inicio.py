"""
streamlit_app_with_location.py

Streamlit app completa (reemplaza a la versión anterior). Añade detección y visualización
dinámica de la ubicación del sensor siguiendo este orden de prioridad:

1) Si el CSV contiene columnas 'lat' y 'lon' (o 'latitude'/'longitude'), usa esas coordenadas.
2) Si el CSV contiene una columna 'location' con nombres (ej. "lab"), intenta geocodificarla
   usando Nominatim (OpenStreetMap). Resultados se cachean en memoria y en archivo
   .geocode_cache.json para evitar consultas repetidas.
3) Si no hay lat/lon ni location geocodificable, ofrece:
   - "Usar geolocalización por IP (aprox.)" mediante ipapi.co (sin token, limitada).
   - Entrada manual de latitud/longitud por el usuario.

Además:
- Detecta delimitador y encoding del CSV.
- Normaliza columnas y parsea la columna de tiempo en formato DD/MM/YYYY HH:MM.
- Mantiene la interfaz con pestañas, gráficas, estadísticas y descarga CSV procesado.
- Muestra la ubicación (marcadores) en un mapa y el nombre de la ubicación detectada.

Requisitos:
- streamlit, pandas, requests
  pip install streamlit pandas requests

Uso:
    streamlit run streamlit_app_with_location.py
Sube el CSV exportado (joinbyfield) o el CSV que tengas. La app intentará sacar la ubicación
del sensor automáticamente. Si lo prefieres, introduce lat/lon manualmente en la barra lateral.
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

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="TempHum — Sensor y Ubicación", layout="wide", page_icon="📍")
CACHE_FILE = ".geocode_cache.json"
USER_AGENT = "TempHumApp/1.0 (contact: your_email@example.com)"  # cambia por tu contacto si quieres

# -------------------------
# Helpers: encoding, delimiter, CSV read (robusto)
# -------------------------
def try_decode(raw_bytes: bytes) -> Tuple[str, str]:
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be", "cp1252", "latin1"]
    for enc in encodings:
        try:
            text = raw_bytes.decode(enc)
            return text, enc
        except Exception:
            continue
    # fallback
    text = raw_bytes.decode("latin1", errors="replace")
    return text, "latin1-replace"

def detect_delimiter(sample_text: str) -> str:
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text[:4096])
        return dialect.delimiter
    except Exception:
        commas = sample_text.count(",")
        semis = sample_text.count(";")
        return ";" if semis > commas else ","

def find_header_line(lines: list, pattern=re.compile(r'\b(time|timestamp|_time|fecha|date|temperature|temp|humidity|humedad|device|location|lat|lon)\b', flags=re.IGNORECASE)) -> Optional[int]:
    for i, line in enumerate(lines[:200]):
        parts = [p.strip().strip('"') for p in re.split(r'[;,]', line)]
        if any(pattern.search(p) for p in parts) and any(re.search(r'[A-Za-z]', p) for p in parts):
            return i
    for i, line in enumerate(lines):
        parts = [p.strip().strip('"') for p in re.split(r'[;,]', line)]
        if any(pattern.search(p) for p in parts) and any(re.search(r'[A-Za-z]', p) for p in parts):
            return i
    return None

def read_csv_robust(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if isinstance(raw, (bytes, bytearray)) and len(raw) >= 2 and raw[0] == 0x1f and raw[1] == 0x8b:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                raw = gz.read()
            st.info("Archivo gzip detectado y descomprimido.")
        except Exception as e:
            st.error(f"Error descomprimiendo gzip: {e}")
            return pd.DataFrame()
    if isinstance(raw, bytes):
        text, enc = try_decode(raw)
        st.info(f"Encoding detectado: {enc}")
    else:
        text = str(raw)
        st.info("Archivo leído como texto.")
    lines = text.splitlines()
    if not lines:
        st.error("El archivo está vacío.")
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
    # eliminar columnas metadata típicas
    cols_to_drop = [c for c in df.columns if isinstance(c, str) and (c.strip().startswith('#') or c.lower().startswith('result') or c.lower().startswith('table') or 'unnamed' in c.lower())]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')
    # normalizar nombres
    df.columns = [str(c).strip() for c in df.columns]
    return df

# -------------------------
# Geocoding helpers (Nominatim + cache)
# -------------------------
def load_geocode_cache():
    try:
        path = Path(CACHE_FILE)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_geocode_cache(cache: dict):
    try:
        Path(CACHE_FILE).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def geocode_location_name(name: str, cache: dict) -> Optional[Tuple[float, float, str]]:
    """
    Geocode a location name using Nominatim (OpenStreetMap).
    Returns (lat, lon, display_name) or None.
    Caches results in provided cache dict (by name key).
    """
    if not name or not name.strip():
        return None
    key = name.strip().lower()
    if key in cache:
        entry = cache[key]
        return entry.get("lat"), entry.get("lon"), entry.get("display_name")
    # call Nominatim
    try:
        params = {"q": name, "format": "json", "limit": 1}
        headers = {"User-Agent": USER_AGENT}
        r = requests.get("https://nominatim.openstreetmap.org/search", params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                item = data[0]
                lat = float(item.get("lat"))
                lon = float(item.get("lon"))
                display = item.get("display_name")
                cache[key] = {"lat": lat, "lon": lon, "display_name": display}
                save_geocode_cache(cache)
                return lat, lon, display
    except Exception:
        pass
    cache[key] = {}
    save_geocode_cache(cache)
    return None

def get_ip_geolocation() -> Optional[Tuple[float, float, str]]:
    """
    Obtains approximate location based on IP using ipapi.co (no token required but limited).
    Returns (lat, lon, city,country) or None.
    """
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

# -------------------------
# UI: upload and CSV processing
# -------------------------
st.title("TempHum — Visualizar ubicación del sensor")

uploaded_file = st.file_uploader("Sube el CSV del sensor (joinbyfield o CSV con lat/lon/location)", type=["csv", "txt", "gz"])
if uploaded_file is None:
    st.info("Sube el CSV que descargaste desde Grafana/Influx o el CSV del sensor.")
    st.stop()

df_raw = read_csv_robust(uploaded_file)
if df_raw.empty:
    st.stop()

st.sidebar.header("Opciones de ubicación")

# Normalize column names to lowercase for detection
orig_columns = list(df_raw.columns)
lower_map = {c: c.lower() for c in orig_columns}
df = df_raw.rename(columns=lower_map)

st.write("Columnas detectadas en el CSV:")
st.write(list(df.columns))

# -------------------------
# Detect lat/lon columns
# -------------------------
lat_col = None
lon_col = None
for c in df.columns:
    lc = c.lower()
    if lat_col is None and lc in ("lat", "latitude"):
        lat_col = c
    if lon_col is None and lc in ("lon", "lng", "longitude"):
        lon_col = c

# If not exact names, try heuristics (columns that parse to floats and look like coords)
if lat_col is None or lon_col is None:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # find pairs with plausible ranges
    for c1 in numeric_cols:
        for c2 in numeric_cols:
            if c1 == c2:
                continue
            lat_sample = pd.to_numeric(df[c1], errors="coerce")
            lon_sample = pd.to_numeric(df[c2], errors="coerce")
            if lat_sample.between(-90, 90).mean() > 0.5 and lon_sample.between(-180, 180).mean() > 0.5:
                lat_col = lat_col or c1
                lon_col = lon_col or c2

# Detect textual location column
location_col = None
for c in df.columns:
    if "location" in c.lower() or "place" in c.lower():
        location_col = c
        break

# Detect time column and parse (use dayfirst since CSVs showed DD/MM/YYYY)
time_col = None
for c in df.columns:
    if c in ("time", "_time", "timestamp", "date", "fecha"):
        time_col = c
        break
if not time_col:
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
            time_col = c
            break

if not time_col:
    st.error("No se encontró columna de tiempo en el CSV. Asegúrate de subir un CSV con columna 'time' o similar.")
    st.stop()

# Parse time and set index
df[time_col] = pd.to_datetime(df[time_col].astype(str), errors="coerce", dayfirst=True)
na_times = int(df[time_col].isna().sum())
if na_times > 0:
    st.warning(f"{na_times} filas no pudieron parsearse en la columna de tiempo; se eliminarán.")
df = df[df[time_col].notna()].copy()
df = df.rename(columns={time_col: "Time"})
df["Time"] = pd.to_datetime(df["Time"], dayfirst=True)
df = df.set_index("Time")

# -------------------------
# Location resolution logic
# -------------------------
st.sidebar.subheader("Resolver ubicación (prioridad automática)")

# load geocode cache
geocode_cache = load_geocode_cache()

detected_latlon = None
detected_label = None
map_df = None

# 1) prefer lat/lon columns
if lat_col and lon_col:
    try:
        lat_series = pd.to_numeric(df[lat_col], errors="coerce")
        lon_series = pd.to_numeric(df[lon_col], errors="coerce")
        valid = lat_series.notna() & lon_series.notna() & lat_series.between(-90, 90) & lon_series.between(-180, 180)
        if valid.any():
            coords = pd.DataFrame({
                "latitude": lat_series[valid].astype(float),
                "longitude": lon_series[valid].astype(float)
            })
            # reduce duplicates
            coords_unique = coords.drop_duplicates().reset_index(drop=True)
            detected_latlon = (coords_unique["latitude"].median(), coords_unique["longitude"].median())
            detected_label = "Coordenadas desde CSV (lat/lon)"
            map_df = coords_unique
    except Exception:
        pass

# 2) if no lat/lon, try 'location' column geocoding
if detected_latlon is None and location_col:
    # pick most common non-null location string
    loc_series = df[location_col].dropna().astype(str).str.strip()
    if not loc_series.empty:
        most_common = loc_series.mode().iloc[0] if not loc_series.mode().empty else loc_series.iloc[0]
        geocoded = geocode_location_name(most_common, geocode_cache)
        if geocoded:
            lat, lon, display = geocoded
            detected_latlon = (lat, lon)
            detected_label = f"Geocodificado desde '{most_common}': {display}"
            map_df = pd.DataFrame([{"latitude": lat, "longitude": lon}])

# 3) provide IP-based approx if user requests
use_ip = st.sidebar.button("Usar geolocalización por IP (aprox.)")
if use_ip and detected_latlon is None:
    ipgeo = get_ip_geolocation()
    if ipgeo:
        lat, lon, city = ipgeo
        detected_latlon = (lat, lon)
        detected_label = f"Ubicación aproximada por IP: {city}"
        map_df = pd.DataFrame([{"latitude": lat, "longitude": lon}])
    else:
        st.sidebar.warning("No fue posible obtener ubicación por IP.")

# 4) manual input fallback (always available)
st.sidebar.markdown("**Ubicación manual**")
manual_lat = st.sidebar.text_input("Latitud (ej. 6.2006)", value="")
manual_lon = st.sidebar.text_input("Longitud (ej. -75.5783)", value="")
use_manual = st.sidebar.button("Usar ubicación manual")
if use_manual:
    try:
        latm = float(manual_lat)
        lonm = float(manual_lon)
        if -90 <= latm <= 90 and -180 <= lonm <= 180:
            detected_latlon = (latm, lonm)
            detected_label = "Ubicación manual"
            map_df = pd.DataFrame([{"latitude": latm, "longitude": lonm}])
        else:
            st.sidebar.error("Coordenadas fuera de rango.")
    except Exception:
        st.sidebar.error("Introduce latitud y longitud válidas (números).")

# Show resolved location (or fallback message)
st.subheader("Ubicación del sensor")
if detected_latlon:
    lat, lon = detected_latlon
    st.markdown(f"**Ubicación detectada:** {detected_label}")
    st.write(f"Lat: {lat:.6f} — Lon: {lon:.6f}")
    # show map
    if map_df is None:
        map_df = pd.DataFrame([{"latitude": lat, "longitude": lon}])
    st.map(map_df)
else:
    st.warning("No se pudo determinar la ubicación automáticamente. Usa la entrada manual o sube CSV con columnas 'lat' y 'lon' o una columna 'location' con nombre reconocible.")
    # Offer geocode input
    st.info("Si tienes un nombre de lugar (ej. 'lab' o 'Universidad X'), escríbelo aquí y presiona 'Geocodificar'.")
    geoname = st.text_input("Nombre de la ubicación para geocodificar (opcional)", value="")
    if st.button("Geocodificar nombre"):
        if geoname.strip():
            res = geocode_location_name(geoname.strip(), geocode_cache)
            if res:
                lat, lon, display = res
                detected_latlon = (lat, lon)
                detected_label = f"Geocodificado desde '{geoname}': {display}"
                map_df = pd.DataFrame([{"latitude": lat, "longitude": lon}])
                st.success(f"Geocodificado: {display} ({lat:.6f}, {lon:.6f})")
                st.map(map_df)
            else:
                st.error("No se pudo geocodificar ese nombre (intenta con una cadena más específica).")
        else:
            st.error("Escribe un nombre para geocodificar.")

# -------------------------
# Rest of the app: visualizations and stats using df (filtered by time)
# -------------------------
# Time range filter (global)
min_time = df.index.min()
max_time = df.index.max()
st.sidebar.markdown("---")
st.sidebar.write(f"Rango de datos en CSV: {min_time} → {max_time}")
default_start = max_time - pd.Timedelta(hours=6) if (max_time - min_time) > pd.Timedelta(hours=1) else min_time
start = st.sidebar.datetime_input("Desde", value=default_start)
end = st.sidebar.datetime_input("Hasta", value=max_time)
if start is None:
    start = min_time
if end is None:
    end = max_time

df_filtered = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))].copy()
if df_filtered.empty:
    st.error("No hay datos en el rango seleccionado.")
    st.stop()

# Normalize/temp/hum detection
temp_col = None
hum_col = None
for c in df_filtered.columns:
    lc = c.lower()
    if temp_col is None and ('temp' in lc):
        temp_col = c
    if hum_col is None and ('hum' in lc):
        hum_col = c

# convert numeric if present
for col in (temp_col, hum_col):
    if col and col in df_filtered.columns:
        df_filtered[col] = pd.to_numeric(df_filtered[col].astype(str).str.replace(',', '.'), errors='coerce')

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Gráfica", "📊 Estadísticas", "📄 Datos / Descargar"])

with tab1:
    st.subheader("Gráfica de temperatura y humedad")
    if temp_col and hum_col and temp_col in df_filtered.columns and hum_col in df_filtered.columns:
        st.line_chart(df_filtered[[temp_col, hum_col]])
    elif temp_col and temp_col in df_filtered.columns:
        st.line_chart(df_filtered[temp_col])
    else:
        st.write("No se detectaron columnas de temperatura/humedad para graficar.")

with tab2:
    st.subheader("Estadísticas")
    for col in [temp_col, hum_col]:
        if col and col in df_filtered.columns:
            s = df_filtered[col].dropna()
            if not s.empty:
                st.markdown(f"### {col}")
                st.metric("Máximo", f"{s.max():.2f}")
                st.metric("Mínimo", f"{s.min():.2f}")
                st.metric("Media", f"{s.mean():.2f}")
                st.metric("Desviación estándar", f"{s.std(ddof=0):.2f}")
            else:
                st.write(f"No hay valores válidos en {col}.")

with tab3:
    st.subheader("Datos y descarga")
    st.dataframe(df_filtered.reset_index().head(500), height=300)
    csv_out = df_filtered.reset_index().to_csv(index=False)
    st.download_button("Descargar CSV procesado (UTF-8)", data=csv_out, file_name="temphum_procesado.csv", mime="text/csv")

# Footer: tips about storing location in the sensor payload
st.markdown("---")
st.markdown("Consejos para que la ubicación se detecte automáticamente:")
st.markdown("""
- Lo ideal es que tu sensor envíe las coordenadas (lat y lon) como campos al escribir en InfluxDB.  
  Ejemplo en Line Protocol: cultivo,device=esp32_1 location=lab latitude=6.2006,longitude=-75.5783 temperature=25.1 163...
- Si no puedes enviar lat/lon, añade un tag `location="lab"` consistente y la app intentará geocodificar ese nombre.
- Evita nombres ambiguos; usa `lab_eafit_main` o `sensor_casa_juan` o añade coordenadas.
""")
