# -*- coding: utf-8 -*-
"""App Streamlit: anticipacion de movimientos en masa por municipio (Santander).

Ejecutar:  .venv\\Scripts\\python.exe -m streamlit run reto/app.py

NOTA ANTI-FUGA: panel_prueba_2025.csv esta SELLADO. Ni este script ni
modelo.py lo abren: las metricas son validacion temporal interna
(2015-2022 -> 2023-2024) y las predicciones de enero 2025 usan solo
datos reales de dic-2024. La comparacion contra la realidad de 2025 se
hace EXTERNAMENTE con predicciones_enero_2025.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import precision_score, recall_score

from modelo import ejecutar

st.set_page_config(page_title="Movimientos en masa - Santander", layout="wide")

DATOS = Path(__file__).parent / "datos"


@st.cache_data(show_spinner="Ajustando modelos (solo 2015-2024; 2025 sellado)...")
def correr():
    return ejecutar(verbose=False)


@st.cache_data
def cargar_geo():
    return json.loads((DATOS / "santander_municipios.geojson").read_text(encoding="utf-8"))


res = correr()
tabla, umbrales = res["tabla"], res["umbrales"]
probas_val, y_val = res["probas_val"], res["y_val"]
enero, top10, mejor = res["enero"], res["top10"], res["mejor"]

st.title("Anticipacion de movimientos en masa por municipio - Santander")
st.caption("Prueba de seleccion UTSmart IA Challenge 2026 - Categoria media. Datos simulados.")

# ---------------- 0. Archivo de prueba sellado ----------------
st.error(
    "**`panel_prueba_2025.csv` está sellado**: ni el entrenamiento, ni la eleccion "
    "de hiperparametros, ni el umbral, ni la app lo abren jamas. Las metricas de "
    "abajo provienen de una **validacion temporal interna** (2015-2022 → 2023-2024) "
    "y la comparacion contra lo que realmente paso en 2025 se hace **externamente** "
    "con el archivo exportado `predicciones_enero_2025.csv` (87 municipios).",
    icon="🔒")

# ---------------- 1. La trampa / fuga de datos ----------------
with st.expander("La trampa: fuga de datos (fuga al futuro)", expanded=True):
    st.warning(
        "La columna **`lluvia_mm` es la lluvia del MISMO mes**: no se conoce al inicio "
        "del mes y usarla como entrada seria informacion del futuro. Por eso **nunca es "
        "una feature**: solo se usa para construir agregados de meses ANTERIORES "
        "(acumulados y desviacion climatica), que si son conocidos al inicio del mes.",
        icon="⚠️")
    st.markdown("""
**Otras reglas respetadas:**
- **No mezclamos años**: la evaluacion es un corte temporal (entrenar 2015-2022, validar 2023-2024); el tuning usa `TimeSeriesSplit` dentro de 2015-2022.
- **No reportamos exactitud**: con ~92% de meses sin evento, predecir siempre "no" daria ~91% sin detectar nada.
- **Features** (todas conocidas al inicio del mes): `lluvia_mm_mes_anterior`, acumulados de lluvia 3m/6m, desviacion vs. climatologia, `eventos_12m`, `eventos_3m`, `meses_sin_evento`, `altitud_m`, lluvia×altitud, estacionalidad del mes.
- El **modelo final** se reentrena con TODOS los años 2015-2024 (maximo aprendizaje) y predice enero 2025.
""")

# ---------------- 2. Comparativa con la linea base ----------------
st.header("1. Modelo vs. linea base (validacion 2023-2024)")
st.dataframe(
    tabla.style.format({"Sensibilidad (recall)": "{:.3f}", "Precisión": "{:.3f}",
                        "F2": "{:.3f}", "Umbral": "{:.2f}"}),
    width='stretch')

st.success(
    f"**Priorizamos la sensibilidad (recall).** Un falso negativo es un municipio que "
    f"sufre un movimiento en masa **sin alerta previa**: vidas y viviendas. Un falso "
    f"positivo solo cuesta una alerta preventiva. En validacion, la linea base detecta "
    f"solo el {tabla.iloc[0]['Sensibilidad (recall)']:.0%} de los eventos; todos los "
    f"modelos la superan por mucho. Mejor balance (F2): **{mejor}**.")

# ---------------- 3. Umbral ----------------
st.header("2. Si bajamos el umbral, ¿que ganamos y que perdemos?")
nombres = list(probas_val.keys())
modelo_nom = st.selectbox("Modelo", nombres, index=nombres.index(mejor))
probs = probas_val[modelo_nom]
umbral = st.slider("Umbral de decision", 0.05, 0.95,
                   float(umbrales[modelo_nom]), 0.01)
pred = (probs >= umbral).astype(int)
c1, c2 = st.columns(2)
c1.metric("Sensibilidad (recall)", f"{recall_score(y_val, pred):.3f}",
          help="Fraccion de eventos reales detectados. Umbral mas bajo -> sube.")
c2.metric("Precision", f"{precision_score(y_val, pred, zero_division=0):.3f}",
          help="Fraccion de alertas reales. Umbral mas bajo -> baja (mas falsas alarmas).")
st.info("Bajar el umbral **gana sensibilidad** (detectamos mas deslizamientos) y "
        "**pierde precision** (mas alertas falsas). Subirlo, lo contrario.")

# ---------------- 4. Top 10 enero 2025 ----------------
st.header(f"3. Diez municipios con mayor probabilidad - enero 2025 ({mejor})")
st.caption("Entradas 100% conocidas al 1° de enero 2025: lluvia real de diciembre 2024, "
           "acumulados de 2024, eventos de 2024, historial y altitud.")
top10s = top10.copy()
top10s.index = top10s.index + 1

tabcol, barracol = st.columns([1, 2])
tabcol.dataframe(top10s.style.format({"probabilidad": "{:.1%}"}), width='stretch')
fig_bar = px.bar(top10.sort_values("probabilidad"), x="probabilidad", y="municipio",
                 orientation="h", labels={"probabilidad": "Probabilidad", "municipio": ""},
                 color="probabilidad", color_continuous_scale="orrd")
fig_bar.update_layout(coloraxis_showscale=False, height=420, margin=dict(l=0, t=20))
barracol.plotly_chart(fig_bar, width='stretch')

with open(res["pred_path"], "rb") as f:
    st.download_button("Descargar predicciones de los 87 municipios (CSV, para la "
                       "comparacion externa contra 2025)", f,
                       file_name="predicciones_enero_2025.csv")

# ---------------- 5. Mapa ----------------
st.header("4. Mapa de probabilidad - enero 2025")
geo = cargar_geo()
map_df = enero[["codigo_dane", "municipio", mejor]].rename(columns={mejor: "probabilidad"})
map_df["codigo_dane"] = map_df["codigo_dane"].astype(str)
fig_map = px.choropleth_map(
    map_df, geojson=geo, locations="codigo_dane",
    featureidkey="properties.codigo_dane", color="probabilidad",
    hover_name="municipio", color_continuous_scale="orrd",
    center={"lat": 6.65, "lon": -73.2}, zoom=6.5,
    map_style="carto-positron", labels={"probabilidad": "Probabilidad"})
fig_map.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig_map, width='stretch')

with st.expander("Como se hizo (defensa ante el jurado)"):
    st.markdown("""
- **¿Usaron `lluvia_mm`?** No como feature. Es la lluvia del mismo mes y no se conoce al inicio: solo alimenta agregados de meses anteriores (`lluvia_mm_mes_anterior`, acumulados, desviacion climatica).
- **¿El archivo 2025?** Sellado. Ni entrenamiento, ni tuning, ni umbral, ni entrada de prediccion. Las metricas son validacion temporal 2023-2024; la comparacion final contra 2025 es externa con `predicciones_enero_2025.csv`.
- **¿Que significa un falso negativo aqui?** Un deslizamiento sin alerta en un municipio: perdida de vidas y viviendas. Por eso priorizamos sensibilidad.
- **Linea base**: repetir `hubo_mm` del mismo mes del año anterior (dato del periodo de entrenamiento, legal).
- **Modelos**: regresion logistica, bosque aleatorio y gradient boosting con `class_weight='balanced'` (desbalance ~8%), tuning con TimeSeriesSplit (scoring F2), umbral por F2 en la validacion.
- **Evaluacion**: corte temporal puro, sin mezclar años.
""")
