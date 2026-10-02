# -*- coding: utf-8 -*-
"""App Streamlit: anticipacion de movimientos en masa por municipio (Santander).

Ejecutar:  .venv\\Scripts\\python.exe -m streamlit run reto/app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import precision_score, recall_score

from modelo import (FEATURES, cargar_prueba, cargar_entrenamiento,
                    construir_modelos, elegir_umbral)

st.set_page_config(page_title="Movimientos en masa - Santander", layout="wide")

DATOS = Path(__file__).parent / "datos"


@st.cache_data(show_spinner="Entrenando modelos (solo con 2015-2024)...")
def entrenar():
    """Entrena con 2015-2024 y devuelve test 2025, probabilidades y tabla base."""
    train = cargar_entrenamiento()
    test = cargar_prueba()

    base_2024 = train[train.anio == 2024][["codigo_dane", "mes", "hubo_mm"]].rename(
        columns={"hubo_mm": "base"})
    test = test.merge(base_2024, on=["codigo_dane", "mes"], how="left").fillna({"base": 0})

    salida = {}
    for nombre, modelo in construir_modelos().items():
        umbral = elegir_umbral(nombre, modelo, train)
        modelo.fit(train[FEATURES], train["hubo_mm"])
        salida[nombre] = {
            "probs": modelo.predict_proba(test[FEATURES])[:, 1],
            "umbral": umbral,
        }
    return test, salida


@st.cache_data
def cargar_geo():
    return json.loads((DATOS / "santander_municipios.geojson").read_text(encoding="utf-8"))


test, resultados = entrenar()
y = test["hubo_mm"].to_numpy()

st.title("Anticipacion de movimientos en masa por municipio - Santander")
st.caption("Prueba de seleccion UTSmart IA Challenge 2026 - Categoria media. Datos simulados.")

# ---------------- 1. La trampa / fuga de datos ----------------
with st.expander("La trampa: fuga de datos (fuga al futuro)", expanded=True):
    st.warning(
        "La columna **`lluvia_mm` es la lluvia del MISMO mes**: no se conoce al inicio "
        "del mes y usarla seria informacion del futuro. Por eso se **descarto**. "
        "En su lugar usamos **`lluvia_mm_mes_anterior`**, que si esta disponible al "
        "inicio del mes.", icon="⚠️")
    st.markdown("""
**Otras reglas respetadas:**
- El modelo se entrena **solo con 2015-2024**; el archivo de 2025 se usa **una unica vez, al final**, para evaluar. No mezclamos años.
- El **umbral** se eligio con validacion temporal interna (entrenar 2015-2022, validar 2023-2024), sin tocar 2025.
- **No reportamos exactitud** como metrica principal: con ~92% de meses sin evento, predecir siempre "no" daria ~91% de exactitud sin detectar nada.
- Features: `lluvia_mm_mes_anterior`, `eventos_12m`, `altitud_m`, `mes` (estacionalidad seno/coseno).
""")

# ---------------- 2. Comparativa con la linea base ----------------
st.header("1. Modelo vs. linea base (evaluacion en 2025)")

filas = [{"Modelo": "Linea base (mismo mes 2024)",
          "Sensibilidad (recall)": recall_score(y, test["base"]),
          "Precision": precision_score(y, test["base"], zero_division=0)}]
for nombre, r in resultados.items():
    pred = (r["probs"] >= r["umbral"]).astype(int)
    filas.append({"Modelo": f"{nombre} (umbral {r['umbral']:.2f})",
                  "Sensibilidad (recall)": recall_score(y, pred),
                  "Precision": precision_score(y, pred, zero_division=0)})
tabla = pd.DataFrame(filas)
st.dataframe(tabla.style.format({"Sensibilidad (recall)": "{:.3f}",
                                 "Precision": "{:.3f}"}), width='stretch')

mejor = "Bosque aleatorio"
st.success(
    f"**Priorizamos la sensibilidad (recall).** Un falso negativo es un municipio "
    f"que sufre un movimiento en masa **sin alerta previa**: vidas y viviendas perdidas. "
    f"Un falso positivo solo cuesta una alerta preventiva. "
    f"La linea base (repetir 2024) solo detecta el {recall_score(y, test['base']):.0%} de los eventos; "
    f"**{mejor}** detecta el {recall_score(y, (resultados[mejor]['probs'] >= resultados[mejor]['umbral']).astype(int)):.0%}, "
    f"~6 veces mas, a cambio de mas falsas alarmas (precision mas baja), un costo aceptable.")

# ---------------- 3. Umbral ----------------
st.header("2. Si bajamos el umbral, ¿que ganamos y que perdemos?")
modelo_nom = st.selectbox("Modelo", list(resultados.keys()))
probs = resultados[modelo_nom]["probs"]
umbral = st.slider("Umbral de decision", 0.05, 0.95,
                   float(resultados[modelo_nom]["umbral"]), 0.01)
pred = (probs >= umbral).astype(int)
c1, c2 = st.columns(2)
c1.metric("Sensibilidad (recall)", f"{recall_score(y, pred):.3f}",
          help="Fraccion de eventos reales que detectamos. Umbral mas bajo -> sube.")
c2.metric("Precision", f"{precision_score(y, pred, zero_division=0):.3f}",
          help="Fraccion de alertas que eran reales. Umbral mas bajo -> baja (mas falsas alarmas).")
st.info("Bajar el umbral **gana sensibilidad** (detectamos mas deslizamientos) y "
        "**pierde precision** (mas alertas falsas). Subirlo hace lo contrario.")

# ---------------- 4. Top 10 enero 2025 ----------------
st.header("3. Diez municipios con mayor probabilidad - enero 2025")
st.caption(f"Probabilidades del {modelo_nom.lower()}. Prediccion legitima: usa solo "
           "informacion disponible al inicio de enero (lluvia de diciembre 2024, historial y altitud).")
enero = test[test.mes == 1][["codigo_dane", "municipio"]].copy()
enero["probabilidad"] = probs[test.mes == 1]
top10 = enero.sort_values("probabilidad", ascending=False).head(10).reset_index(drop=True)
top10.index = top10.index + 1

tabcol, barracol = st.columns([1, 2])
tabcol.dataframe(top10.style.format({"probabilidad": "{:.1%}"}), width='stretch')
fig_bar = px.bar(top10.sort_values("probabilidad"), x="probabilidad", y="municipio",
                 orientation="h", labels={"probabilidad": "Probabilidad", "municipio": ""},
                 color="probabilidad", color_continuous_scale="orrd")
fig_bar.update_layout(coloraxis_showscale=False, height=420, margin=dict(l=0, t=20))
barracol.plotly_chart(fig_bar, width='stretch')

# ---------------- 5. Mapa ----------------
st.header("4. Mapa de probabilidad - enero 2025")
geo = cargar_geo()
map_df = enero.copy()
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
- **¿Usaron `lluvia_mm`?** No. Es la lluvia del mismo mes y no se conoce al inicio del mes: es fuga de datos. Usamos `lluvia_mm_mes_anterior`.
- **¿Que significa un falso negativo aqui?** Un municipio que sufre un deslizamiento sin alerta previa: perdida de vidas y viviendas. Por eso priorizamos sensibilidad sobre precision.
- **Linea base**: repetir `hubo_mm` del mismo mes de 2024 (dato del periodo de entrenamiento, legal).
- **Modelos**: regresion logistica y bosque aleatorio con `class_weight='balanced'` por el desbalance (~8% de positivos), umbral elegido por F2 en validacion temporal 2023-2024.
- **Evaluacion**: entrenamiento 2015-2024, prueba 2025. Sin mezclar años.
""")
