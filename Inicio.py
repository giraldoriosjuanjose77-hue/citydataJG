"""
streamlit_app_fixed_time.py

Versión completa y corregida de tu app que soluciona el error "'Time'"
Alcance:
- Detecta delimitador (coma o punto y coma).
- Normaliza nombres de columnas (quita espacios y pasa a minúsculas).
- Busca la columna de tiempo de forma robusta (time, Time, _time, timestamp, date, fecha).
- Parsea fechas en formato DD/MM/YYYY HH:MM (dayfirst=True) y UTF-8/latin1.
- Renombra/normaliza temperature/humidity y preserva la UI (mapa, pestañas, gráficos, estadísticas, filtros).
Instrucciones:
- Reemplaza tu archivo actual por este contenido y ejecuta:
    streamlit run streamlit_app_fixed_time.py
- Sube el CSV "joinbyfield" (ej. el que pegaste) y la app debería procesarlo sin KeyError 'Time'.
"""

import io
import csv
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

# Página
st.set_page_config(page_title="Análisis de Sensores - Fix Time", page_icon="📊", layout="wide")

st.title("Análisis de Sensores — CSV (fix 'Time')")
st.markdown("Sube tu CSV exportado (por ejemplo `time;temperature;humidity`) y la app procesará la columna de tiempo correctamente.")

# Mapa estático (tu contenido original)
eafit_location = pd.DataFrame({'lat': [6.2006], 'lon': [-75.5783], 'location': ['Universidad EAFIT']})
st.subheader("📍 Ubicación de los Sensores - Universidad EAFIT")
st.map(eafit_location, zoom=15)

# Helper: detectar delimitador
def detect_delimiter(sample: str) -> str:
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample[:4096])
        return dialect.delimiter
    except Exception:
        # heurística simple
        commas = sample.count(',')
        semis = sample.count(';')
        return ';' if semis > commas else ','

# Subir archivo
uploaded_file = st.file_uploader('Seleccione archivo CSV', type=['csv', 'txt'], accept_multiple_files=False)
if uploaded_file is None:
    st.info('Sube el CSV (ej. time;temperature;humidity).')
    st.stop()

# Leer bytes y decodificar con fallback
raw = uploaded_file.read()
text = None
for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
    try:
        text = raw.decode(enc)
        detected_enc = enc
        break
    except Exception:
        continue
if text is None:
    # último recurso
    try:
        text = raw.decode('latin1', errors='replace')
        detected_enc = 'latin1-replace'
    except Exception as e:
        st.error(f"No se pudo decodificar el archivo: {e}")
        st.stop()

st.info(f"Encoding detectado: {detected_enc}")

lines = text.splitlines()
if not lines:
    st.error("Archivo vacío.")
    st.stop()

# Detectar delimitador y leer con pandas
delimiter = detect_delimiter("\n".join(lines[:20]))
st.info(f"Delimitador detectado: '{delimiter}'")

# Buscar línea de encabezado (primera que contenga nombres reconocibles)
header_candidates = ('time', '_time', 'timestamp', 'date', 'fecha', 'temperature', 'temp', 'humidity', 'humedad')
header_idx = None
for i, line in enumerate(lines[:50]):
    parts = [p.strip().strip('"') for p in line.split(delimiter)]
    # hay al menos una etiqueta reconocible y al menos un token alfabético
    if any(any(h in p.lower() for h in header_candidates) for p in parts) and any(any(c.isalpha() for c in p) for p in parts):
        header_idx = i
        break
if header_idx is None:
    # usar primera línea por defecto
    header_idx = 0

# Leer CSV usando header detectado
try:
    df = pd.read_csv(io.StringIO("\n".join(lines)), header=header_idx, sep=delimiter)
except Exception as e:
    st.error(f"Error leyendo CSV con pandas: {e}")
    st.stop()

# Eliminar columnas de metadatos típicas si existen
drop_meta = [c for c in df.columns if isinstance(c, str) and (c.strip().startswith('#') or c.lower().startswith('result') or c.lower().startswith('table') or 'unnamed' in c.lower())]
if drop_meta:
    df = df.drop(columns=drop_meta, errors='ignore')
    st.info(f"Se eliminaron columnas metadata: {drop_meta}")

# Normalizar nombres: strip y lowercase keys for detection
df.columns = [str(c).strip() for c in df.columns]
col_map_lower = {c: c.lower() for c in df.columns}
df.rename(columns=col_map_lower, inplace=True)

st.info(f"Columnas detectadas: {list(df.columns)}")

# Encontrar columna de tiempo
time_candidates = [c for c in df.columns if c in ('time', '_time', 'timestamp', 'date', 'fecha')]
chosen_time_col = time_candidates[0] if time_candidates else None

if not chosen_time_col:
    # heurística por contenido: la que más parsea como fecha con dayfirst=True
    for c in df.columns:
        sample_vals = df[c].dropna().astype(str).head(10).tolist()
        parsed = 0
        for s in sample_vals:
            try:
                pd.to_datetime(s, dayfirst=True)
                parsed += 1
            except Exception:
                pass
        if parsed >= max(1, len(sample_vals) // 2):
            chosen_time_col = c
            st.write(f"Columna de tiempo inferida por contenido: '{c}'")
            break

if not chosen_time_col:
    st.error("No se encontró columna de tiempo. Asegúrate de que exista una columna como 'time' o 'timestamp'.")
    st.stop()

st.success(f"Usando columna de tiempo: '{chosen_time_col}'")

# Parsear fechas (dayfirst=True para DD/MM/YYYY)
df[chosen_time_col] = pd.to_datetime(df[chosen_time_col].astype(str), errors='coerce', dayfirst=True)
na_time = int(df[chosen_time_col].isna().sum())
if na_time > 0:
    st.warning(f"{na_time} filas no pudieron parsearse en la columna de tiempo '{chosen_time_col}'. Se eliminarán esas filas.")
    st.write("Algunos ejemplos no parseables:")
    st.write(df[df[chosen_time_col].isna()].head(5))

# Filtrar filas con tiempo válido
df = df[df[chosen_time_col].notna()].copy()

# Renombrar y set index a 'Time'
df = df.rename(columns={chosen_time_col: 'Time'})
df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
df = df.set_index('Time')

# Detectar columnas temperatura y humedad
temp_col = None
hum_col = None
for c in df.columns:
    lc = c.lower()
    if temp_col is None and ('temp' in lc or 'temperature' in lc or 'temperatura' in lc):
        temp_col = c
    if hum_col is None and ('hum' in lc or 'humidity' in lc or 'humedad' in lc):
        hum_col = c

# Si el CSV ya tiene 'temperature' y 'humidity' en minúsculas, lo detectará
rename_map = {}
if temp_col:
    rename_map[temp_col] = 'temperatura'
if hum_col:
    rename_map[hum_col] = 'humedad'
if rename_map:
    df = df.rename(columns=rename_map)
    st.success(f"Columnas renombradas automáticamente: {rename_map}")

# Si no se detectaron, informa
if 'temperatura' not in df.columns and 'humedad' not in df.columns:
    st.warning("No se detectaron columnas llamadas 'temperatura' ni 'humedad'. Columnas actuales:")
    st.write(list(df.columns))

# Convertir a numérico (soporta coma decimal)
for col in ['temperatura', 'humedad']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

# Eliminar filas sin datos numéricos
if 'temperatura' in df.columns and 'humedad' in df.columns:
    df = df[~(df['temperatura'].isna() & df['humedad'].isna())]
else:
    if 'temperatura' in df.columns:
        df = df[~df['temperatura'].isna()]
    if 'humedad' in df.columns:
        df = df[~df['humedad'].isna()]

if df.empty:
    st.error("Después del filtrado no quedan filas válidas.")
    st.stop()

# Interfaz: pestañas como en tu app original
tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualización", "📊 Estadísticas", "🔍 Filtros", "🗺️ Información del Sitio"])

with tab1:
    st.subheader('Visualización de Datos')
    vars_available = [c for c in ['temperatura', 'humedad'] if c in df.columns]
    if not vars_available:
        st.error("No hay columnas de 'temperatura' ni 'humedad' detectadas para graficar.")
    else:
        options = vars_available.copy()
        if 'temperatura' in vars_available and 'humedad' in vars_available:
            options = ['temperatura', 'humedad', 'Ambas variables']
        variable = st.selectbox("Seleccione variable a visualizar", options)
        chart_type = st.selectbox("Seleccione tipo de gráfico", ["Línea", "Área", "Barra"])
        def plot_series(series, title, chart_type):
            st.write(f"### {title}")
            if chart_type == "Línea":
                st.line_chart(series)
            elif chart_type == "Área":
                st.area_chart(series)
            else:
                st.bar_chart(series)
        if variable == "Ambas variables":
            if 'temperatura' in df.columns:
                plot_series(df['temperatura'], "Temperatura", chart_type)
            if 'humedad' in df.columns:
                plot_series(df['humedad'], "Humedad", chart_type)
        else:
            plot_series(df[variable], variable.capitalize(), chart_type)
        if st.checkbox('Mostrar datos crudos'):
            st.dataframe(df.head(200))

with tab2:
    st.subheader('Análisis Estadístico')
    stat_cols = [c for c in ['temperatura', 'humedad'] if c in df.columns]
    if not stat_cols:
        st.write("No hay columnas para calcular estadísticas.")
    else:
        for col in stat_cols:
            s = df[col].dropna()
            if s.empty:
                st.write(f"No hay datos válidos en {col}.")
                continue
            st.metric(f"{col.capitalize()} Máximo", f"{s.max():.2f}")
            st.metric(f"{col.capitalize()} Mínimo", f"{s.min():.2f}")
            st.metric(f"{col.capitalize()} Media", f"{s.mean():.2f}")
            st.write(f"Desviación estándar: {s.std(ddof=0):.2f}")

with tab3:
    st.subheader('Filtros de Datos')
    filter_variable = st.selectbox("Seleccione variable para filtrar", [c for c in ['temperatura', 'humedad'] if c in df.columns])
    col1, col2 = st.columns(2)
    with col1:
        min_val = st.slider(f'Valor mínimo de {filter_variable}', float(df[filter_variable].min()), float(df[filter_variable].max()), float(df[filter_variable].mean()), key="min_val")
        filtrado_df_min = df[df[filter_variable] > min_val]
        st.write(f"Registros con {filter_variable} superior a {min_val}:")
        st.dataframe(filtrado_df_min.head(200))
    with col2:
        max_val = st.slider(f'Valor máximo de {filter_variable}', float(df[filter_variable].min()), float(df[filter_variable].max()), float(df[filter_variable].mean()), key="max_val")
        filtrado_df_max = df[df[filter_variable] < max_val]
        st.write(f"Registros con {filter_variable} inferior a {max_val}:")
        st.dataframe(filtrado_df_max.head(200))
    if st.button('Descargar datos filtrados'):
        if not filtrado_df_min.empty:
            csv_bytes = filtrado_df_min.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button("Descargar CSV", data=csv_bytes, file_name="datos_filtrados.csv", mime="text/csv")
        else:
            st.info("No hay datos para descargar con ese filtro.")

with tab4:
    st.subheader("Información del Sitio de Medición")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Ubicación del Sensor")
        st.write("**Universidad EAFIT**")
        st.write("- Latitud: 6.2006")
        st.write("- Longitud: -75.5783")
    with col2:
        st.write("### Detalles del Sensor")
        st.write("- Tipo: ESP32")
        st.write("- Variables medidas: Temperatura (°C), Humedad (%)")

# Vista previa y descarga final
st.markdown("---")
st.subheader("Datos procesados - vista previa")
st.dataframe(df.reset_index().head(500), height=300)
csv_all = df.reset_index().to_csv(index=False)
st.download_button("Descargar CSV procesado (UTF-8)", data=csv_all, file_name="temphum_procesado.csv", mime="text/csv")
