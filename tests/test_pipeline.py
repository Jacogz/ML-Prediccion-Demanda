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


def test_folds_temporales_sin_mezclar_fechas():
    tabla = pipeline.construir_tabla_entrenamiento(datos_sinteticos()).dropna()
    folds = list(pipeline.folds_temporales(tabla, n_folds=4, dias_por_fold=7))

    assert len(folds) == 4
    for train, test in folds:
        assert train["date"].max() < test["date"].min()
        assert len(test) == 7
    # Las semanas de prueba son consecutivas y la última termina en la última fecha.
    pruebas = pd.concat(test for _, test in folds)["date"]
    assert pruebas.is_unique and pruebas.is_monotonic_increasing
    assert pruebas.max() == tabla["date"].max()


def test_evaluar_promedia_el_mae_de_los_folds():
    resultado = pipeline.evaluar(datos_sinteticos(), n_folds=3)
    por_fold = resultado["por_fold"]
    assert len(por_fold) == 3
    assert resultado["mae_modelo"] == pytest.approx(por_fold["mae_modelo"].mean())
    assert resultado["mae_linea_base"] == pytest.approx(por_fold["mae_linea_base"].mean())
    assert len(resultado["test"]) == 3 * pipeline.DIAS_POR_FOLD
    assert not resultado["test"]["linea_base"].isna().any()


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


def test_dia_semana_es_variable_del_calendario():
    tabla = pipeline.construir_tabla_entrenamiento(datos_sinteticos())
    assert "dia_semana" in pipeline.columnas_features(tabla)
    assert (tabla["dia_semana"] == tabla["date"].dt.day_of_week).all()
    assert set(tabla["dia_semana"]) == set(range(7))


def test_explicacion_suma_la_prediccion_y_agrupa_por_familia():
    datos = datos_sinteticos()
    explicacion = pipeline.predecir_y_explicar(datos)
    fecha, prediccion = pipeline.predecir_dia_siguiente(datos)

    assert explicacion["fecha"] == fecha
    assert explicacion["prediccion"] == pytest.approx(prediccion)
    # Aditividad de SHAP: valor base + suma de aportes = predicción.
    total = explicacion["valor_base"] + explicacion["aportes"].sum()
    assert total == pytest.approx(explicacion["prediccion"], abs=1e-6)
    assert set(explicacion["aportes"].index) == {"A", "B", *pipeline.CALENDARIO.values()}


def test_wape_es_el_mae_sobre_la_venta_promedio():
    resultado = pipeline.evaluar(datos_sinteticos(), n_folds=3)
    test = resultado["test"]
    venta_promedio = test["ventas_totales"].mean()
    mae = (test["ventas_totales"] - test["prediccion"]).abs().mean()
    assert resultado["wape_modelo"] == pytest.approx(100 * mae / venta_promedio)
    assert resultado["wape_linea_base"] == pytest.approx(
        100 * resultado["mae_linea_base"] / venta_promedio
    )
    assert {"wape_modelo", "wape_linea_base"} <= set(resultado["por_fold"].columns)


def test_atribucion_reparte_toda_la_prediccion():
    explicacion = pipeline.predecir_y_explicar(datos_sinteticos())
    atribucion = explicacion["atribucion"]

    assert set(atribucion.index) == {"A", "B", pipeline.FACTORES_EXTERNOS}
    assert atribucion["unidades"].sum() == pytest.approx(explicacion["prediccion"], abs=1e-6)
    assert atribucion["proporcion"].sum() == pytest.approx(100)
    # Los factores externos son la suma de los aportes del calendario.
    assert atribucion.loc[pipeline.FACTORES_EXTERNOS, "unidades"] == pytest.approx(
        explicacion["aportes_externos"].sum()
    )
    assert set(explicacion["aportes_externos"].index) == set(pipeline.CALENDARIO.values())


def test_datos_de_prueba_cumplen_la_plantilla_y_el_modelo_gana():
    from src import ejemplo

    csv = ejemplo.generar_ventas_ejemplo().to_csv(index=False)
    assert csv == ejemplo.generar_ventas_ejemplo().to_csv(index=False)  # misma semilla
    datos = pipeline.cargar_plantilla(io.StringIO(csv))
    assert set(datos["family"]) == set(ejemplo.FAMILIAS)
    resultado = pipeline.evaluar(datos)
    assert resultado["wape_modelo"] < resultado["wape_linea_base"]
