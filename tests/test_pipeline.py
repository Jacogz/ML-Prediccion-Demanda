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


def csv_plantilla(fechas, ventas_por_fecha, familias=("A", "B")):
    filas = ["fecha,categoria,unidades"]
    for fecha in fechas:
        for familia in familias:
            filas.append(f"{fecha.date()},{familia},{ventas_por_fecha.get(fecha, 1)}")
    return io.StringIO("\n".join(filas))


def test_dia_faltante_se_completa_y_lag_7_mira_7_dias_atras():
    fechas = pd.date_range("2024-11-01", "2025-01-31")
    navidad, nochebuena = pd.Timestamp("2024-12-25"), pd.Timestamp("2024-12-24")
    datos = pipeline.cargar_plantilla(
        csv_plantilla(fechas.drop(navidad), {nochebuena: 99})
    )

    assert list(datos.attrs["fechas_completadas"]) == [navidad]
    assert (datos.loc[datos["date"] == navidad, "sales"] == 0).all()

    tabla = pipeline.construir_tabla_entrenamiento(datos).set_index("date")
    assert tabla.loc["2024-12-31", "lag_7_A"] == 99  # el 24 de diciembre, no el 23
    assert tabla.loc["2025-01-01", "lag_7_A"] == 0  # el 25 de diciembre completado


def test_categoria_sin_filas_un_dia_cuenta_como_venta_cero():
    fechas = pd.date_range("2024-01-01", periods=pipeline.MIN_DIAS)
    csv = csv_plantilla(fechas, {})
    lineas = [l for l in csv.getvalue().splitlines() if l != f"{fechas[30].date()},B,1"]
    datos = pipeline.cargar_plantilla(io.StringIO("\n".join(lineas)))

    assert len(datos) == 2 * pipeline.MIN_DIAS
    assert len(datos.attrs["fechas_completadas"]) == 0  # el día sí tiene registros de A
    fila = datos[(datos["date"] == fechas[30]) & (datos["family"] == "B")]
    assert fila["sales"].item() == 0
