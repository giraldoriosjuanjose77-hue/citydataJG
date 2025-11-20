import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Análisis de Sensores - Mi Ciudad",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title('📊 Análisis de datos de Sensores en Mi Ciudad')
st.markdown("""
    Esta aplicación permite analizar datos de temperatura y humedad
    recolectados por sensores de temperatura y humedad en diferentes puntos de la ciudad.
""")

# Create map data for EAFIT
eafit_location = pd.DataFrame({
    'lat': [6.2006],
    'lon': [-75.5783],
    'location': ['Universidad EAFIT']
})

# Display map
st.subheader("📍 Ubicación de los Sensores - Universidad EAFIT")
st.map(eafit_location, zoom=15)

# File uploader
uploaded_file = st.file_uploader('Seleccione archivo CSV', type=['csv'])

if uploaded_file is not None:
    try:
        # Load and process data
        df1 = pd.read_csv(uploaded_file)

        # Mostrar columnas detectadas para debug (útil para ver por qué fallaba)
        st.info("Columnas detectadas en el CSV:")
        st.write(list(df1.columns))

        # Normalizar nombres de columnas: quitar espacios al inicio/fin
        df1.columns = [c.strip() for c in df1.columns]

        # RENOMBRAR TEMPERATURA/HUMEDAD si vienen con nombres largos (opcional)
        # Detectamos columnas que contengan 'temp' o 'hum' (case-insensitive)
        col_map = {}
        for c in df1.columns:
            lc = c.lower()
            if 'temp' in lc and 'temper' not in col_map.values():
                col_map[c] = 'temperatura'
            if ('hum' in lc or 'hume' in lc) and 'humedad' not in col_map.values():
                col_map[c] = 'humedad'
        # aplicamos mapping si encontramos
        if col_map:
            df1 = df1.rename(columns=col_map)
            st.success(f"Se renombraron columnas detectadas: {col_map}")

        # BUSCAR columna de tiempo (varias posibilidades comunes)
        possible_time_cols = [c for c in df1.columns if c.lower() in ('time','timestamp','_time','fecha','date')]
        # Si no hay nombres exactos, intentar detectar una columna con tipo datetime o que parezca fecha
        if not possible_time_cols:
            # heurística: columna cuyo nombre contiene 'time' o 'fecha' o contiene 'date'
            possible_time_cols = [c for c in df1.columns if any(k in c.lower() for k in ('time','fecha','date','timestamp','_time'))]

        chosen_time_col = None
        if possible_time_cols:
            # preferimos la primera candidata
            chosen_time_col = possible_time_cols[0]
            st.write(f"Columna de tiempo seleccionada (detectada): '{chosen_time_col}'")
        else:
            # intentar detectar columna que tenga formato de fecha en las primeras filas
            for c in df1.columns:
                sample = df1[c].astype(str).dropna().head(10).tolist()
                parsed = 0
                for s in sample:
                    try:
                        pd.to_datetime(s)
                        parsed += 1
                    except Exception:
                        pass
                if parsed >= max(1, len(sample)//2):
                    chosen_time_col = c
                    st.write(f"Columna de tiempo inferida por contenido: '{chosen_time_col}'")
                    break

        if not chosen_time_col:
            st.error("No se encontró ninguna columna de tiempo. Por favor renombra la columna de tiempo a 'Time', 'time' o 'timestamp' o revisa el CSV.")
            st.stop()

        # Parsear la columna de tiempo con coerción; mostrar filas con parsing fallido si los hay
        df1[chosen_time_col] = pd.to_datetime(df1[chosen_time_col], errors='coerce', dayfirst=False)

        # Si todas las filas quedaron NaT => intentar con dayfirst=True (formatos DD/MM/YYYY)
        if df1[chosen_time_col].isna().all():
            df1[chosen_time_col] = pd.to_datetime(df1[chosen_time_col].astype(str), errors='coerce', dayfirst=True)

        # Revisar si hay muchos NaT
        na_count = df1[chosen_time_col].isna().sum()
        if na_count > 0:
            st.warning(f"{na_count} filas no pudieron parsearse en la columna '{chosen_time_col}'. Se mostrarán sólo las filas con tiempo válido.")
            # mostrar ejemplos problemáticos para que puedas corregir el CSV
            bad_examples = df1[df1[chosen_time_col].isna()].head(5)
            st.write("Ejemplos de valores no parseables en la columna de tiempo:")
            st.write(bad_examples[[c for c in df1.columns if c != chosen_time_col]].head(5))
            # continuar pero filtrando filas inválidas

        # Filtrar filas con tiempo válido
        df1 = df1[df1[chosen_time_col].notna()].copy()
        df1 = df1.rename(columns={chosen_time_col: 'Time'})
        df1['Time'] = pd.to_datetime(df1['Time'])
        df1 = df1.set_index('Time')

        # A partir de aquí asumimos que las columnas 'temperatura' y 'humedad' existen (o las renombramos arriba)
        if 'temperatura' not in df1.columns or 'humedad' not in df1.columns:
            st.warning("No se detectaron columnas llamadas exactamente 'temperatura' y 'humedad'. Columnas actuales:")
            st.write(list(df1.columns))
            st.info("Si tus columnas tienen otros nombres, renómbralas a 'temperatura' y 'humedad' o asegúrate que contienen 'temp'/'hum' en su nombre para que el renombrado automático funcione.")

        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualización", "📊 Estadísticas", "🔍 Filtros", "🗺️ Información del Sitio"])

        with tab1:
            st.subheader('Visualización de Datos')

            # Variable selector: limitar a columnas detectadas
            vars_available = [c for c in ['temperatura', 'humedad'] if c in df1.columns]
            if not vars_available:
                st.error("No hay columnas de 'temperatura' ni 'humedad' detectadas para graficar.")
            else:
                # ofrecer opción 'Ambas variables' sólo si ambas existen
                options = vars_available.copy()
                if 'temperatura' in vars_available and 'humedad' in vars_available:
                    options = ['temperatura', 'humedad', 'Ambas variables']
                variable = st.selectbox("Seleccione variable a visualizar", options)

                # Chart type selector
                chart_type = st.selectbox(
                    "Seleccione tipo de gráfico",
                    ["Línea", "Área", "Barra"]
                )

                # Create plot based on selection
                def plot_series(series, title, chart_type):
                    st.write(f"### {title}")
                    if chart_type == "Línea":
                        st.line_chart(series)
                    elif chart_type == "Área":
                        st.area_chart(series)
                    else:
                        st.bar_chart(series)

                if variable == "Ambas variables":
                    plot_series(df1["temperatura"], "Temperatura", chart_type)
                    plot_series(df1["humedad"], "Humedad", chart_type)
                else:
                    plot_series(df1[variable], variable.capitalize(), chart_type)

                # Raw data display with toggle
                if st.checkbox('Mostrar datos crudos'):
                    st.write(df1.head(200))

        with tab2:
            st.subheader('Análisis Estadístico')

            # Variable selector for statistics
            stat_variable = st.radio(
                "Seleccione variable para estadísticas",
                [c for c in ['temperatura', 'humedad'] if c in df1.columns]
            )

            # Statistical summary
            stats_df = df1[stat_variable].describe()

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(stats_df)

            with col2:
                # Additional statistics
                st.metric(f"{stat_variable.capitalize()} Promedio", f"{stats_df['mean']:.2f}{'°C' if stat_variable=='temperatura' else '%'}")
                st.metric(f"{stat_variable.capitalize()} Máxima", f"{stats_df['max']:.2f}{'°C' if stat_variable=='temperatura' else '%'}")
                st.metric(f"{stat_variable.capitalize()} Mínima", f"{stats_df['min']:.2f}{'°C' if stat_variable=='temperatura' else '%'}")
                st.write(f"Desviación estándar: {stats_df['std']:.2f}")

        with tab3:
            st.subheader('Filtros de Datos')

            # Variable selector for filtering
            filter_variable = st.selectbox(
                "Seleccione variable para filtrar",
                [c for c in ['temperatura', 'humedad'] if c in df1.columns]
            )

            col1, col2 = st.columns(2)

            with col1:
                # Minimum value filter
                min_val = st.slider(
                    f'Valor mínimo de {filter_variable}',
                    float(df1[filter_variable].min()),
                    float(df1[filter_variable].max()),
                    float(df1[filter_variable].mean()),
                    key="min_val"
                )

                filtrado_df_min = df1[df1[filter_variable] > min_val]
                st.write(f"Registros con {filter_variable} superior a",
                        f"{min_val}{'°C' if filter_variable == 'temperatura' else '%'}:")
                st.dataframe(filtrado_df_min)

            with col2:
                # Maximum value filter
                max_val = st.slider(
                    f'Valor máximo de {filter_variable}',
                    float(df1[filter_variable].min()),
                    float(df1[filter_variable].max()),
                    float(df1[filter_variable].mean()),
                    key="max_val"
                )

                filtrado_df_max = df1[df1[filter_variable] < max_val]
                st.write(f"Registros con {filter_variable} inferior a",
                        f"{max_val}{'°C' if filter_variable == 'temperatura' else '%'}:")
                st.dataframe(filtrado_df_max)

            # Download filtered data
            if st.button('Descargar datos filtrados'):
                csv = filtrado_df_min.to_csv().encode('utf-8')
                st.download_button(
                    label="Descargar CSV",
                    data=csv,
                    file_name='datos_filtrados.csv',
                    mime='text/csv',
                )

        with tab4:
            st.subheader("Información del Sitio de Medición")

            col1, col2 = st.columns(2)

            with col1:
                st.write("### Ubicación del Sensor")
                st.write("**Universidad EAFIT**")
                st.write("- Latitud: 6.2006")
                st.write("- Longitud: -75.5783")
                st.write("- Altitud: ~1,495 metros sobre el nivel del mar")

            with col2:
                st.write("### Detalles del Sensor")
                st.write("- Tipo: ESP32")
                st.write("- Variables medidas:")
                st.write("  * Temperatura (°C)")
                st.write("  * Humedad (%)")
                st.write("- Frecuencia de medición: Según configuración")
                st.write("- Ubicación: Campus universitario")

    except Exception as e:
        st.error(f'Error al procesar el archivo: {str(e)}')
else:
    st.warning('Por favor, cargue un archivo CSV para comenzar el análisis.')

# Footer
st.markdown("""
    ---
    Desarrollado para el análisis de datos de sensores urbanos.
    Ubicación: Universidad EAFIT, Medellín, Colombia
""")
