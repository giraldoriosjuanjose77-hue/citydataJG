"""
Inicio.py

Versión completa y corregida de la app Streamlit usada en tu proyecto.
Objetivo principal: resolver el StreamlitAPIException lanzado por st.sidebar.datetime_input
(causado por pasar objetos que Streamlit no acepta, p.ej. pandas.Timestamp con tzinfo o numpy types).

Qué incluye este archivo:
- Lectura robusta de CSV exportados desde Grafana/Influx (detecta encoding, delimitador, gzip).
- Detección automática de la fila de encabezado real y limpieza de metadatos (#group, #datatype, result, table).
- Parsing de la columna de tiempo (soporta DD/MM/YYYY HH:MM con dayfirst=True).
- Conversión explícita de pandas.Timestamp / numpy.datetime64 a datetime.datetime "naive"
  (sin tzinfo) antes de pasarlo a st.sidebar.datetime_input.
- Detección/normalización de columnas temperatura/humedad (renombra a 'temperatura' y 'humedad').
- Detección de ubicación por prioridad: lat/lon en CSV -> columna 'location' geocodificada -> IP -> manual.
- Visualizaciones (mapa, gráficas), estadísticas (max/min/mean/std) y descarga CSV procesado.
- Mensajes claros de error/advertencia para depuración.

Instrucciones:
1) Reemplaza tu archivo Inicio.py por este archivo.
2) Ejecuta: streamlit run Inicio.py
3) Sube tu CSV exportado desde Grafana/Influx (o el joinbyfield que te funciona para tus compañeros).
4) La app manejará el datetime_input sin provocar StreamlitAPIException.

No se incluyen tokens ni credenciales.
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
# Configuración y constantes
# -------------------------
st.set_page_config(page_title="Análisis Sensores - Inicio", layout="wide", page_icon="📊")

GEOCODE_CACHE_FILE = ".geocode_cache.json"
USER_AGENT = "TempHumApp/1.0 (contact: example@example.com)"  # cámbialo si quieres identificar peticiones

# -------------------------
# Helpers: encoding/delimiter/header detection
# -------------------------
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

def find_header_line(lines, pattern=re.compile(
    r'\b(time|timestamp|_time|fecha|date|temperature|temp|humidity|humedad|device|location|lat|lon)\b',
    flags=re.IGNORECASE)):
    for i, line in enumerate(lines[:200]):
        parts = [p.strip().strip('"') for p in re.split(r'[;,]', line)]
        if any(pattern.search(p) for p in parts) and any(re.search(r'[A-Za-z]', p) for p in parts):
            return i
    for i, line in enumerate(lines):
        parts = [p.strip().strip('"') for p in re.split(r'[;,]', line)]
        if any(pattern.search(p) for p in parts) and any(re.search(r'[A-Za-z]', p) for p in parts):
            return i
    return None

# -------------------------
# CSV reading and cleaning
# -------------------------
def read_csv_robust(uploaded_file) -> pd.DataFrame:
    """
    Lee un archivo subido y devuelve un DataFrame limpio.
    Soporta gzip, distintos encodings y detecta delimitador y header.
    Elimina columnas metadata típicas de Influx/Grafana (#group, #datatype, result, table, Unnamed).
    """
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    if not raw:
        st.error("Archivo vacío.")
        return pd.DataFrame()

    # Detect gzip magic bytes
    if isinstance(raw, (bytes, bytearray)) and len(raw) >= 2 and raw[0] == 0x1f and raw[1] == 0x8b:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                raw = gz.read()
            st.info("Archivo comprimido (gzip) detectado y descomprimido.")
        except Exception as e:
            st.error(f"Error al descomprimir gzip: {e}")
            return pd.DataFrame()

    # Decode bytes -> text
    if isinstance(raw, bytes):
        text, enc = try_decode_bytes(raw)
        if text is None:
            st.error("No se pudo decodificar el archivo. Prueba guardarlo como UTF-8 o UTF-8 sin BOM.")
            return pd.DataFrame()
        st.info(f"Encoding detectado: {enc}")
    else:
        text = str(raw)
        st.info("Contenido leído como texto.")

    lines = text.splitlines()
    if not lines:
        st.error("El archivo no contiene líneas.")
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

    # Eliminar columnas metadata comunes
    cols_to_drop = [c for c in df.columns if isinstance(c, str) and (
        c.strip().startswith('#') or c.lower().startswith('result') or c.lower().startswith('table') or 'unnamed' in c.lower()
    )]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')

    df.columns = [str(c).strip() for c in df.columns]
    return df

# -------------------------
# Helpers: datetime handling for Streamlit
# -------------------------
def _to_naive_pydatetime(ts) -> Optional[datetime]:
    """
    Convierte pandas.Timestamp / numpy.datetime64 / str a datetime.datetime "naive" (sin tzinfo).
    Devuelve None si no se puede convertir.
    """
    try:
        ts_pd = pd.to_datetime(ts)
        py = ts_pd.to_pydatetime()
        if getattr(py, "tzinfo", None) is not None:
            py = py.replace(tzinfo=None)
        return py
    except Exception:
        return None

# -------------------------
# Geocoding helpers (Nominatim) and cache
# -------------------------
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

# -------------------------
# UI: carga y procesamiento
# -------------------------
st.title("Análisis de Sensores — Inicio")

uploaded_file = st.file_uploader("Sube el CSV (exportado desde Grafana/Influx o join-by-field)", type=["csv", "txt", "gz"])
if uploaded_file is None:
    st.info("Por favor sube un archivo CSV para comenzar.")
    st.stop()

df_raw = read_csv_robust(uploaded_file)
if df_raw.empty:
    st.stop()

# Preserve original column names but also have a lowercase mapping for detection
orig_cols = list(df_raw.columns)
lower_map = {c: c.lower() for c in orig_cols}
df = df_raw.rename(columns=lower_map)

st.write("Columnas detectadas en el CSV:")
st.write(list(df.columns))

# -------------------------
# Detect time column robustly and parse
# -------------------------
time_candidates = [c for c in df.columns if c in ('time', '_time', 'timestamp', 'date', 'fecha')]
chosen_time_col = time_candidates[0] if time_candidates else None

if not chosen_time_col:
    # heurística por contenido
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(10).tolist()
        parsed = 0
        for s in sample:
            try:
                pd.to_datetime(s, dayfirst=True)
                parsed += 1
            except Exception:
                pass
        if parsed >= max(1, len(sample) // 2):
            chosen_time_col = c
            st.write(f"Columna de tiempo inferida por contenido: '{c}'")
            break

if not chosen_time_col:
    st.error("No se encontró columna de tiempo. Asegúrate de que el CSV tenga 'time' o campo similar.")
    st.stop()

# Parse times (use dayfirst=True because CSVs suelen venir DD/MM/YYYY)
df[chosen_time_col] = pd.to_datetime(df[chosen_time_col].astype(str), errors='coerce', dayfirst=True)
na_time = int(df[chosen_time_col].isna().sum())
if na_time > 0:
    st.warning(f"{na_time} filas no pudieron parsearse en la columna de tiempo '{chosen_time_col}'. Se eliminarán.")
    st.write(df[df[chosen_time_col].isna()].head(5))

# Keep rows with valid time
df = df[df[chosen_time_col].notna()].copy()
# Rename and set index
df = df.rename(columns={chosen_time_col: "Time"})
df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
df = df.set_index('Time')

# -------------------------
# Ensure the datetimes passed to Streamlit are "naive" datetimes
# -------------------------
min_time_pd = df.index.min()
max_time_pd = df.index.max()

min_time = _to_naive_pydatetime(min_time_pd)
max_time = _to_naive_pydatetime(max_time_pd)

if min_time is None or max_time is None:
    st.error("No se pudo determinar un rango de tiempo válido a partir de los datos.")
    st.stop()

# default start: 6 hours before max if range allows, else min_time
default_start_pd = max_time_pd - pd.Timedelta(hours=6) if (max_time_pd - min_time_pd) > pd.Timedelta(hours=1) else min_time_pd
default_start = _to_naive_pydatetime(default_start_pd) or min_time

# IMPORTANT: pass only Python datetime.datetime (naive) to datetime_input
try:
    start = st.sidebar.datetime_input("Desde", value=default_start, min_value=min_time, max_value=max_time)
    end = st.sidebar.datetime_input("Hasta", value=max_time, min_value=min_time, max_value=max_time)
except Exception as e:
    st.error(f"Error al crear los datetime_input: {e}")
    st.stop()

if start is None or end is None:
    st.error("Rango de tiempo inválido en los inputs.")
    st.stop()

# Apply time filter using pandas (convert start/end to pandas timestamps for comparison)
df_filtered = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))].copy()
if df_filtered.empty:
    st.error("No hay datos en el rango seleccionado.")
    st.stop()

# -------------------------
# Detect temp/humidity columns and normalize
# -------------------------
temp_col = None
hum_col = None
for c in df_filtered.columns:
    lc = c.lower()
    if temp_col is None and ('temp' in lc):
        temp_col = c
    if hum_col is None and ('hum' in lc):
        hum_col = c

# Heuristics by numeric ranges if not found
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
    st.success(f"Columnas renombradas automáticamente: {rename_map}")

# Convert numeric strings with comma decimals
for col in ['temperatura', 'humedad']:
    if col in df_filtered.columns:
        df_filtered[col] = pd.to_numeric(df_filtered[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

# Drop rows without numeric data
if 'temperatura' in df_filtered.columns and 'humedad' in df_filtered.columns:
    df_filtered = df_filtered[~(df_filtered['temperatura'].isna() & df_filtered['humedad'].isna())]
else:
    if 'temperatura' in df_filtered.columns:
        df_filtered = df_filtered[~df_filtered['temperatura'].isna()]
    if 'humedad' in df_filtered.columns:
        df_filtered = df_filtered[~df_filtered['humedad'].isna()]

if df_filtered.empty:
    st.error("Después del filtrado no quedan filas con datos válidos.")
    st.stop()

# -------------------------
# Location detection (lat/lon, location geocoding, ip, manual)
# -------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Ubicación del sensor")

# find lat/lon columns
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

# 1) prefer lat/lon columns
if lat_col and lon_col:
    lat_s = pd.to_numeric(df[lat_col], errors='coerce')
    lon_s = pd.to_numeric(df[lon_col], errors='coerce')
    valid_mask = lat_s.notna() & lon_s.notna() & lat_s.between(-90, 90) & lon_s.between(-180, 180)
    if valid_mask.any():
        coords = pd.DataFrame({'latitude': lat_s[valid_mask].astype(float), 'longitude': lon_s[valid_mask].astype(float)}).drop_duplicates().reset_index(drop=True)
        detected_latlon = (coords['latitude'].median(), coords['longitude'].median())
        detected_label = "Coordenadas desde CSV"
        map_df = coords

# 2) geocode location name
if detected_latlon is None and location_col:
    loc_series = df[location_col].dropna().astype(str).str.strip()
    if not loc_series.empty:
        most_common = loc_series.mode().iloc[0] if not loc_series.mode().empty else loc_series.iloc[0]
        ge = geocode_name(most_common, geocode_cache)
        if ge:
            lat, lon, disp = ge
            detected_latlon = (lat, lon)
            detected_label = f"Geocodificado desde '{most_common}'"
            map_df = pd.DataFrame([{'latitude': lat, 'longitude': lon}])

# 3) options for user: use IP approx or manual input
if detected_latlon:
    st.sidebar.success(f"Ubicación detectada: {detected_label} ({detected_latlon[0]:.6f}, {detected_latlon[1]:.6f})")
else:
    if st.sidebar.button("Usar geolocalización por IP (aprox.)"):
        ipg = ip_geolocation()
        if ipg:
            lat, lon, city = ipg
            detected_latlon = (lat, lon)
            detected_label = f"IP: {city}"
            map_df = pd.DataFrame([{'latitude': lat, 'longitude': lon}])
            st.sidebar.success(f"Ubicación aproximada por IP: {city} ({lat:.6f}, {lon:.6f})")
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
                st.sidebar.success(f"Ubicación manual usada: ({latm:.6f}, {lonm:.6f})")
            else:
                st.sidebar.error("Coordenadas fuera de rango.")
        except Exception:
            st.sidebar.error("Introduce latitud y longitud válidas (números).")

# Show location or message
st.subheader("Ubicación del sensor")
if detected_latlon:
    lat, lon = detected_latlon
    if map_df is None:
        map_df = pd.DataFrame([{'latitude': lat, 'longitude': lon}])
    st.map(map_df)
    st.write(f"**Ubicación:** {detected_label}")
    st.write(f"Lat: {lat:.6f}, Lon: {lon:.6f}")
else:
    st.warning("No se pudo determinar la ubicación automáticamente. Usa IP o ingresa lat/lon manualmente o añade lat/lon en el CSV.")

# -------------------------
# Visualización y estadísticas
# -------------------------
tab1, tab2, tab3 = st.tabs(["📈 Gráfica", "📊 Estadísticas", "📄 Datos / Descargar"])

with tab1:
    st.subheader("Gráfica")
    plot_cols = [c for c in ['temperatura', 'humedad'] if c in df_filtered.columns]
    if plot_cols:
        st.line_chart(df_filtered[plot_cols])
    else:
        st.write("No se detectaron columnas para graficar (temperatura/humedad).")

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

# -------------------------
# Footer: consejos
# -------------------------
st.markdown("---")
st.markdown("""
Consejos para que la ubicación se detecte automáticamente:
- Envía `latitude` y `longitude` desde el sensor al escribir en InfluxDB (Line Protocol) si puedes.
- Si usas tags `location`, úsalos de forma consistente para permitir geocodificado automático.
""")
