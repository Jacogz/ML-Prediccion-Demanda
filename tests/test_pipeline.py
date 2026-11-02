import io

import numpy as np
import pandas as pd
import pytest

from src import pipeline


def datos_sinteticos(dias=120):
    fechas = pd.date_range("2024-01-01", periods=dias)
    rng = np.random.default_rng(0)
    filas = []
    for familia, nivel in (("A", 100), ("B", 50)):
        ventas = nivel + 20 * (fechas.dayofweek >= 5) + rng.integers(0, 10, dias)
        filas.append(pd.DataFrame({
            "date": fechas, "family": familia, "sales": ventas.astype(float), "onpromotion": 0.0,
        }))
    return pd.concat(filas, ignore_index=True)


def test_division_temporal_sin_mezclar_fechas():
    tabla = pipeline.construir_tabla_entrenamiento(datos_sinteticos())
    train, test = pipeline.dividir_en_el_tiempo(tabla, test_dias=30)
    assert train["date"].max() < test["date"].min()
    assert len(test) == 30


def test_features_no_usan_el_dia_predicho_ni_el_futuro():
    datos = datos_sinteticos()
    corte = pd.Timestamp("2024-03-01")
    original = pipeline.construir_tabla_entrenamiento(datos)

    corrupto = datos.copy()
    corrupto.loc[corrupto["date"] >= corte, "sales"] = np.nan
    recalculado = pipeline.construir_tabla_entrenamiento(corrupto)

    cols = pipeline.columnas_features(original)
    pd.testing.assert_frame_equal(
        original.loc[original["date"] <= corte, cols].reset_index(drop=True),
        recalculado.loc[recalculado["date"] <= corte, cols].reset_index(drop=True),
    )


def test_plantilla_sin_columna_obligatoria_da_error_claro():
    csv = io.StringIO("fecha,categoria\n2024-01-01,A\n")
    with pytest.raises(ValueError, match="unidades"):
        pipeline.cargar_plantilla(csv)


def test_plantilla_suma_filas_repetidas_y_promocion_opcional():
    filas = ["fecha,categoria,unidades"]
    for fecha in pd.date_range("2024-01-01", periods=pipeline.MIN_DIAS):
        filas += [f"{fecha.date()},A,1", f"{fecha.date()},A,2"]
    datos = pipeline.cargar_plantilla(io.StringIO("\n".join(filas)))
    assert (datos["sales"] == 3).all()
    assert (datos["onpromotion"] == 0).all()


def test_prediccion_es_del_dia_siguiente():
    datos = datos_sinteticos()
    fecha, prediccion = pipeline.predecir_dia_siguiente(datos)
    assert fecha == datos["date"].max() + pd.Timedelta(days=1)
    assert np.isfinite(prediccion) and prediccion > 0
