"""Pipeline reutilizable de predicción de demanda (issue #6).

Recibe un CSV con la plantilla del ADR 0001 (`fecha`, `categoria`, `unidades` y, opcional,
`promocion`), crea las variables sin fuga del notebook (issue #7), entrena y predice la venta
total del día siguiente.

Las funciones `recortar_arranque`, `calcular_features` y `construir_tabla_entrenamiento` son las
mismas del notebook `model_development.ipynb`: si cambian allá, hay que cambiarlas aquí.

Uso: python -m src.pipeline --data <csv> [--folds 4]
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Plantilla del ADR 0001 -> nombres internos (los mismos del notebook, que vienen de Favorita).
COLUMNAS_PLANTILLA = {
    "fecha": "date",
    "categoria": "family",
    "unidades": "sales",
    "promocion": "onpromotion",
}
COLUMNAS_OBLIGATORIAS = ["fecha", "categoria", "unidades"]

FEATURES_POR_FAMILIA = [
    "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "onpromotion",
]
# Validación temporal como en el notebook: TimeSeriesSplit con ventanas de prueba de 7 días.
# El notebook evalúa la línea base con todos los folds posibles; aquí se limita a las últimas
# N_FOLDS semanas porque cada fold reentrena el modelo y la app debe responder rápido.
DIAS_POR_FOLD = 7
N_FOLDS = 4
MIN_DIAS = 60


# Hiperparámetros del modelo final. Se copian de la salida de GridSearchCV en la sección
# "Modelo final" del notebook (issue #12). Mientras esté vacío, HGBT usa los valores por defecto.
PARAMETROS_MODELO = {}


def crear_modelo():
    # Misma configuración fija del notebook. early_stopping=False porque el early stopping de HGBT
    # separa al azar una parte del entrenamiento, y la división tiene que ser temporal.
    return HistGradientBoostingRegressor(
        max_iter=200, early_stopping=False, random_state=42, **PARAMETROS_MODELO
    )


def cargar_plantilla(archivo):
    datos = pd.read_csv(archivo)
    datos.columns = [c.strip().lower() for c in datos.columns]

    faltantes = [c for c in COLUMNAS_OBLIGATORIAS if c not in datos.columns]
    if faltantes:
        raise ValueError(
            f"Al archivo le faltan las columnas: {', '.join(faltantes)}. "
            "Debe tener fecha, categoria, unidades y, si las hay, promocion."
        )
    if "promocion" not in datos.columns:
        datos["promocion"] = 0

    datos = datos[list(COLUMNAS_PLANTILLA)].rename(columns=COLUMNAS_PLANTILLA)
    datos["date"] = pd.to_datetime(datos["date"], errors="coerce")
    datos["sales"] = pd.to_numeric(datos["sales"], errors="coerce")
    datos["onpromotion"] = pd.to_numeric(datos["onpromotion"], errors="coerce").fillna(0)
    datos["family"] = datos["family"].astype(str).str.strip()

    if datos["date"].isna().any():
        raise ValueError("Hay fechas que no se pudieron leer. Usa el formato AAAA-MM-DD.")
    if datos["sales"].isna().any():
        raise ValueError("Hay filas sin unidades o con unidades que no son números.")

    # Varias filas del mismo día y categoría (por ejemplo, una por venta) se suman.
    datos = datos.groupby(["date", "family"], as_index=False)[["sales", "onpromotion"]].sum()

    if datos["date"].nunique() < MIN_DIAS:
        raise ValueError(f"Se necesitan al menos {MIN_DIAS} días de ventas para entrenar.")

    fechas_completadas = fechas_sin_registros(datos)
    datos = completar_calendario(datos)
    datos.attrs["fechas_completadas"] = fechas_completadas
    return datos


def fechas_sin_registros(datos):
    calendario = pd.date_range(datos["date"].min(), datos["date"].max(), freq="D")
    return calendario.difference(datos["date"].unique())


def completar_calendario(datos):
    """Una fila por cada día y categoría entre la primera y la última fecha.

    `shift` y `rolling` cuentan filas, no días: si falta un día (por ejemplo, el 25 de diciembre
    en Favorita, cuando la tienda cierra), `lag_7` apuntaría a 8 días atrás. En la plantilla cada
    fila es venta registrada, así que una combinación de día y categoría sin filas es venta 0.
    """
    calendario = pd.date_range(datos["date"].min(), datos["date"].max(), freq="D")
    grilla = pd.MultiIndex.from_product(
        [sorted(datos["family"].unique()), calendario], names=["family", "date"]
    )
    return (
        datos.set_index(["family", "date"])[["sales", "onpromotion"]]
        .reindex(grilla, fill_value=0)
        .reset_index()
    )


def recortar_arranque(df):
    primera_venta = df.loc[df["sales"] > 0].groupby("family")["date"].min()
    df = df[df["date"] >= df["family"].map(primera_venta)]
    return df.reset_index(drop=True)


def calcular_features(df, fechas_festivas=frozenset()):
    df = df.sort_values(["family", "date"]).copy()

    sales_pasado = df.groupby("family")["sales"].shift(1)
    df["lag_7"] = df.groupby("family")["sales"].shift(7)
    df["lag_14"] = df.groupby("family")["sales"].shift(14)

    for ventana in (7, 14):
        agrupado = sales_pasado.groupby(df["family"]).rolling(ventana)
        for stat in ("mean", "min", "max", "std"):
            df[f"rolling_{stat}_{ventana}"] = (
                getattr(agrupado, stat)().reset_index(level=0, drop=True)
            )

    dia_mes = df["date"].dt.day
    fin_mes_anterior = df["date"].dt.to_period("M").dt.start_time - pd.Timedelta(days=1)
    df["dias_desde_quincena"] = np.where(
        dia_mes < 15, (df["date"] - fin_mes_anterior).dt.days, dia_mes - 15
    )

    df["es_festivo"] = df["date"].isin(fechas_festivas)
    df["dia_semana"] = df["date"].dt.day_of_week  # 0 = lunes; entero, el modelo es de árboles
    return df


def sanitizar(nombre):
    return nombre.replace(" ", "_")


def construir_tabla_entrenamiento(df, fechas_festivas=frozenset()):
    df = calcular_features(df, fechas_festivas)

    pivote = df.pivot(index="date", columns="family", values=FEATURES_POR_FAMILIA)
    pivote.columns = [f"{feat}_{sanitizar(fam)}" for feat, fam in pivote.columns]
    pivote = pivote.reset_index()

    calendario = (
        df.groupby("date")[["dias_desde_quincena", "es_festivo", "dia_semana"]].first().reset_index()
    )
    target = df.groupby("date")["sales"].sum().rename("ventas_totales").reset_index()

    return pivote.merge(calendario, on="date").merge(target, on="date")


def columnas_features(tabla):
    return [c for c in tabla.columns if c not in ("date", "ventas_totales")]


def folds_temporales(tabla, n_folds=N_FOLDS, dias_por_fold=DIAS_POR_FOLD):
    """Pares (train, test): cada test es una semana y su train, todos los días anteriores."""
    tabla = tabla.sort_values("date").reset_index(drop=True)
    if len(tabla) < (n_folds + 1) * dias_por_fold:
        raise ValueError(
            f"No hay suficientes días con historia para validar con {n_folds} folds "
            f"de {dias_por_fold} días."
        )
    tscv = TimeSeriesSplit(n_splits=n_folds, test_size=dias_por_fold)
    for train_idx, test_idx in tscv.split(tabla):
        train, test = tabla.iloc[train_idx], tabla.iloc[test_idx]
        assert train["date"].max() < test["date"].min()
        yield train, test


def prediccion_linea_base(tabla):
    # Vender lo mismo que el mismo día de la semana pasada: suma de lag_7 de todas las categorías.
    lags_7 = [c for c in tabla.columns if c.startswith("lag_7_")]
    return tabla[lags_7].sum(axis=1, min_count=len(lags_7))


def entrenar(tabla):
    tabla = tabla.dropna()  # primeras filas sin historia suficiente para los rezagos
    modelo = crear_modelo()
    modelo.fit(tabla[columnas_features(tabla)], tabla["ventas_totales"])
    return modelo


def evaluar(datos, fechas_festivas=frozenset(), n_folds=N_FOLDS):
    """MAE del modelo y de la línea base: promedio simple de los MAE de cada fold."""
    tabla = construir_tabla_entrenamiento(recortar_arranque(datos), fechas_festivas)
    tabla = tabla.dropna()  # primeras fechas sin historia suficiente para los rezagos

    predicciones, por_fold = [], []
    for fold, (train, test) in enumerate(folds_temporales(tabla, n_folds), start=1):
        modelo = entrenar(train)
        test = test.copy()
        test["prediccion"] = modelo.predict(test[columnas_features(test)])
        test["linea_base"] = prediccion_linea_base(test)
        test["fold"] = fold
        predicciones.append(test)
        por_fold.append({
            "fold": fold,
            "desde": test["date"].min(),
            "hasta": test["date"].max(),
            "mae_modelo": mean_absolute_error(test["ventas_totales"], test["prediccion"]),
            "mae_linea_base": mean_absolute_error(test["ventas_totales"], test["linea_base"]),
        })

    por_fold = pd.DataFrame(por_fold)
    return {
        "mae_modelo": por_fold["mae_modelo"].mean(),
        "mae_linea_base": por_fold["mae_linea_base"].mean(),
        "por_fold": por_fold,
        "test": pd.concat(predicciones)[
            ["date", "fold", "ventas_totales", "prediccion", "linea_base"]
        ],
    }


def predecir_dia_siguiente(datos, fechas_festivas=frozenset(), promociones=None):
    """Entrena con todo el historial y predice la venta total del día después de la última fecha."""
    datos = recortar_arranque(datos)
    fecha = datos["date"].max() + pd.Timedelta(days=1)
    if promociones is None:  # por defecto, las promociones del último día
        ultimo_dia = datos[datos["date"] == datos["date"].max()]
        promociones = dict(zip(ultimo_dia["family"], ultimo_dia["onpromotion"]))

    familias = datos["family"].unique()
    manana = pd.DataFrame({
        "date": fecha,
        "family": familias,
        "sales": np.nan,  # lo que se quiere predecir: no se conoce
        "onpromotion": [promociones.get(f, 0) for f in familias],
    })
    tabla = construir_tabla_entrenamiento(pd.concat([datos, manana]), fechas_festivas)

    modelo = entrenar(tabla[tabla["date"] < fecha])
    fila = tabla[tabla["date"] == fecha]
    return fecha, float(modelo.predict(fila[columnas_features(fila)])[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV con la plantilla del ADR 0001")
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="semanas de validación")
    args = parser.parse_args()

    datos = cargar_plantilla(args.data)
    completadas = datos.attrs["fechas_completadas"]
    if len(completadas):
        print(f"Días sin registros, completados con venta 0: {len(completadas)}")
    resultado = evaluar(datos, n_folds=args.folds)
    print(f"MAE línea base (mismo día de la semana anterior): {resultado['mae_linea_base']:.1f}")
    print(f"MAE modelo: {resultado['mae_modelo']:.1f}")

    fecha, prediccion = predecir_dia_siguiente(datos)
    print(f"Predicción de venta total para {fecha.date()}: {prediccion:.0f}")


if __name__ == "__main__":
    main()
