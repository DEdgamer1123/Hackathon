# -*- coding: utf-8 -*-
"""
Reto medio: anticipar movimientos en masa por municipio (Santander).

Reglas anti-fuga respetadas (estrictas):
- panel_prueba_2025.csv esta SELLADO: este script ni siquiera lo abre.
  La comparacion contra lo real de 2025 se hace EXTERNAMENTE con el
  archivo exportado predicciones_enero_2025.csv.
- `lluvia_mm` (lluvia del MISMO mes) jamas es feature de un mes: solo se
  usa para construir agregados de meses ANTERIORES (acumulados, desviacion).
- Metricas = validacion temporal INTERNA: entrenar 2015-2022, validar
  2023-2024 (sin mezclar anios). El modelo final se reentrena con TODO
  2015-2024 y predice enero 2025 con entradas reales de dic-2024.
- Los nombres de municipio se leen en UTF-8 (Málaga, San Andrés...).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATOS = Path(__file__).parent / "datos"

FEATURES = [
    "lluvia_mm_mes_anterior", "lluvia_acum_3m", "lluvia_acum_6m",
    "lluvia_desv_clima", "eventos_12m", "eventos_3m", "meses_sin_evento",
    "altitud_m", "lluvia_x_altitud", "mes_sin", "mes_cos",
]

CORTE_VALIDACION = 2023  # <=2022 entrena, 2023-2024 valida


# ---------------------------------------------------------------- utilidades
def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features construidas SOLO con meses anteriores (shift(1)).
    `lluvia_mm` queda fuera salvo agregados con desfase (conocidos al inicio)."""
    df = df.sort_values(["codigo_dane", "anio", "mes"]).reset_index(drop=True)

    def g(col):
        return df.groupby("codigo_dane")[col]

    # acumulados de lluvia de meses anteriores (shift(1) -> excluye el mes actual)
    lluvia = g("lluvia_mm")
    df["_lluvia_prev"] = lluvia.shift(1)
    df["lluvia_acum_3m"] = lluvia.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    df["lluvia_acum_6m"] = lluvia.transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).sum())
    # primer mes del panel: usar la lluvia del mes anterior ya dada en el CSV
    df["lluvia_acum_3m"] = df["lluvia_acum_3m"].fillna(df["lluvia_mm_mes_anterior"])
    df["lluvia_acum_6m"] = df["lluvia_acum_6m"].fillna(df["lluvia_mm_mes_anterior"])
    df["_lluvia_prev"] = df["_lluvia_prev"].fillna(df["lluvia_mm_mes_anterior"])
    # climatologia: media de la lluvia de ese mes del municipio en anios previos
    df["_clima"] = df.groupby(["codigo_dane", "mes"])["lluvia_mm"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean())
    df["lluvia_desv_clima"] = (df["_lluvia_prev"] - df["_clima"]).fillna(0)
    df["_clima"] = df["_clima"].fillna(df["lluvia_mm"])  # primer mes sin cambio

    # eventos recientes
    df["eventos_3m"] = g("eventos_mes").transform(
        lambda s: s.shift(1).rolling(3, min_periods=0).sum().fillna(0))

    # meses transcurridos desde el ultimo evento (al inicio del mes, tope 60)
    contadores = []
    for _, grp in df.groupby("codigo_dane"):
        b = grp["hubo_mm"].to_numpy()
        out = np.full(len(b), 60)
        ultimo = -1
        for i in range(len(b)):
            if ultimo >= 0:
                out[i] = min(i - ultimo, 60)
            if b[i] >= 1:
                ultimo = i
        contadores.append(pd.Series(out, index=grp.index))
    df["meses_sin_evento"] = pd.concat(contadores).sort_index()

    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["lluvia_x_altitud"] = df["_lluvia_prev"] * df["altitud_m"] / 1000.0
    df = df.drop(columns=["_clima", "_lluvia_prev", "lluvia_mm"])  # fuga -> fuera
    return df


def cargar_entrenamiento() -> pd.DataFrame:
    df = pd.read_csv(DATOS / "panel_entrenamiento_2015_2024.csv", encoding="utf-8")
    assert df["anio"].max() == 2024
    return construir_features(df)


# ---------------------------------------------------------------- modelos
def espacios_modelos() -> dict:
    """Modelo -> (estimador, rejilla). class_weight por el desbalance (~8%)."""
    return {
        "Regresión logística": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))]),
            {"clf__C": [0.01, 0.1, 1.0]},
        ),
        "Bosque aleatorio": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"n_estimators": [300, 600], "min_samples_leaf": [3, 10],
             "max_depth": [None, 12]},
        ),
        "Gradient boosting": (
            HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
            {"learning_rate": [0.05, 0.1], "max_iter": [200, 400],
             "max_leaf_nodes": [15, 31]},
        ),
    }


def f2_score(y, pred):
    return fbeta_score(y, pred, beta=2, zero_division=0)


def tunear_modelos(train_sub: pd.DataFrame) -> dict:
    """GridSearchCV con TimeSeriesSplit, SOLO sobre el tramo de entrenamiento."""
    X, y = train_sub[FEATURES], train_sub["hubo_mm"]
    tscv = TimeSeriesSplit(n_splits=4)
    mejores = {}
    for nombre, (est, rejilla) in espacios_modelos().items():
        gs = GridSearchCV(est, rejilla, scoring=fbeta_scorer, cv=tscv,
                          n_jobs=-1, refit=True)
        gs.fit(X, y)
        mejores[nombre] = gs.best_estimator_
    return mejores


from sklearn.metrics import make_scorer
fbeta_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)


def evaluar_validacion(train: pd.DataFrame) -> dict:
    """Entrena 2015-2022, valida 2023-2024. Devuelve tabla y objetos."""
    tr = train[train.anio <= CORTE_VALIDACION - 1]
    va = train[train.anio >= CORTE_VALIDACION]

    # Línea base: repetir hubo_mm del mismo mes del anio anterior.
    ant = train[["codigo_dane", "anio", "mes", "hubo_mm"]].copy()
    ant["anio"] += 1
    va = va.merge(ant.rename(columns={"hubo_mm": "base"}),
                  on=["codigo_dane", "anio", "mes"], how="left").fillna({"base": 0})

    modelos = tunear_modelos(tr)
    y = va["hubo_mm"].to_numpy()
    filas = [{"Modelo": "Línea base (mismo mes año anterior)",
              "Sensibilidad (recall)": recall_score(y, va["base"]),
              "Precisión": precision_score(y, va["base"], zero_division=0),
              "F2": f2_score(y, va["base"]), "Umbral": np.nan}]

    umbrales, probas_val = {}, {}
    for nombre, modelo in modelos.items():
        p = modelo.predict_proba(va[FEATURES])[:, 1]
        probas_val[nombre] = p
        # umbral por F2 dentro de la validacion
        mejor = max(np.arange(0.05, 0.95, 0.01),
                    key=lambda u: f2_score(y, (p >= u).astype(int)))
        umbrales[nombre] = float(mejor)
        pred = (p >= mejor).astype(int)
        filas.append({"Modelo": f"{nombre} (umbral {mejor:.2f})",
                      "Sensibilidad (recall)": recall_score(y, pred, zero_division=0),
                      "Precisión": precision_score(y, pred, zero_division=0),
                      "F2": f2_score(y, pred), "Umbral": mejor})

    return {"tabla": pd.DataFrame(filas), "modelos": modelos,
            "umbrales": umbrales, "probas_val": probas_val, "y_val": y}


# ------------------------------------------------- predicción enero 2025
def construir_enero_2025() -> pd.DataFrame:
    """DataFrame de entradas para enero-2025 desde el panel de entrenamiento."""
    raw = pd.read_csv(DATOS / "panel_entrenamiento_2015_2024.csv", encoding="utf-8")
    raw = raw.sort_values(["codigo_dane", "anio", "mes"]).reset_index(drop=True)
    feat = construir_features(raw)
    enero = []
    for cod, hist in raw.groupby("codigo_dane"):
        hist = hist.sort_values(["anio", "mes"])
        hf = feat[feat.codigo_dane == cod].sort_values(["anio", "mes"])
        h2024 = hist[hist.anio == 2024]
        lluvia_dic24 = h2024[h2024.mes == 12]["lluvia_mm"].iloc[0]
        acum3 = h2024[h2024.mes.isin([10, 11, 12])]["lluvia_mm"].sum()
        acum6 = h2024[h2024.mes >= 7]["lluvia_mm"].sum()
        clima_dic = hist[hist.mes == 1]["lluvia_mm"].mean()  # climatologia de eneros previos
        ev24 = h2024["eventos_mes"].sum()
        ev3 = h2024[h2024.mes.isin([10, 11, 12])]["eventos_mes"].sum()
        # meses sin evento al inicio de enero 2025
        b = hist["hubo_mm"].to_numpy()
        pos = np.where(b >= 1)[0]
        meses_sin = 60 if len(pos) == 0 else min(len(b) - 1 - pos[-1], 60)
        enero.append({
            "codigo_dane": cod, "municipio": hist["municipio"].iloc[0],
            "anio": 2025, "mes": 1,
            "lluvia_mm_mes_anterior": lluvia_dic24,
            "lluvia_acum_3m": acum3, "lluvia_acum_6m": acum6,
            "lluvia_desv_clima": lluvia_dic24 - clima_dic,
            "eventos_12m": ev24, "eventos_3m": ev3,
            "meses_sin_evento": meses_sin,
            "altitud_m": hist["altitud_m"].iloc[0],
            "lluvia_x_altitud": lluvia_dic24 * hist["altitud_m"].iloc[0] / 1000.0,
            "mes_sin": np.sin(2 * np.pi / 12), "mes_cos": np.cos(2 * np.pi / 12),
        })
    return pd.DataFrame(enero)


# ---------------------------------------------------------------- principal
def ejecutar(verbose: bool = True) -> dict:
    train = cargar_entrenamiento()
    val = evaluar_validacion(train)

    # Modelo final: reentrenar con TODO 2015-2024 con los mejores hiperparametros.
    enero = construir_enero_2025()
    finales, probas_enero = {}, {}
    for nombre, modelo_val in val["modelos"].items():
        final = clone(modelo_val)
        final.fit(train[FEATURES], train["hubo_mm"])
        probas_enero[nombre] = final.predict_proba(enero[FEATURES])[:, 1]
        finales[nombre] = final
        enero[nombre] = probas_enero[nombre]

    # mejor modelo = mayor F2 en validacion
    mejor_nombre = max(val["umbrales"],
                       key=lambda n: val["tabla"][val["tabla"].Modelo.str.startswith(n)]["F2"].iloc[0])
    top10 = (enero[["codigo_dane", "municipio", mejor_nombre]]
             .rename(columns={mejor_nombre: "probabilidad"})
             .sort_values("probabilidad", ascending=False).head(10).reset_index(drop=True))

    out = enero[["codigo_dane", "municipio"] + list(finales.keys())].copy()
    pred_path = Path(__file__).parent / "predicciones_enero_2025.csv"
    out.to_csv(pred_path, index=False, encoding="utf-8-sig")

    res = {"tabla": val["tabla"], "umbrales": val["umbrales"],
           "probas_val": val["probas_val"], "y_val": val["y_val"],
           "enero": enero, "top10": top10, "mejor": mejor_nombre,
           "pred_max": None, "pred_path": pred_path}

    if verbose:
        print("=== Validación temporal interna (train 2015-2022 → val. 2023-2024) ===")
        print(val["tabla"].to_string(index=False))
        print(f"\n=== Mejor modelo en validación: {mejor_nombre} ===")
        print("=== Top 10 municipios, enero 2025 (modelo final, entr. 2015-2024) ===")
        print(top10.to_string(index=False))
        print(f"\nPredicciones de los 87 municipios exportadas a: {pred_path.name}")
        print("panel_prueba_2025.csv NO fue abierto en ningun paso.")
    return res


if __name__ == "__main__":
    ejecutar()
