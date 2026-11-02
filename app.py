"""Interfaz de predicción de demanda (issue #9). Ejecutar con: streamlit run app.py"""

import holidays
import pandas as pd
import streamlit as st

from src import pipeline

PAISES = {"Ninguno": None, "Colombia": "CO", "Ecuador": "EC"}

EJEMPLO = """fecha,categoria,unidades,promocion
2024-01-01,Pan,120,0
2024-01-01,Bebidas,80,2
2024-01-02,Pan,135,0
2024-01-02,Bebidas,75,0
"""


def fechas_festivas(pais, datos):
    if PAISES[pais] is None:
        return frozenset()
    anios = range(datos["date"].dt.year.min(), datos["date"].dt.year.max() + 2)
    return frozenset(pd.to_datetime(list(holidays.country_holidays(PAISES[pais], years=anios))))


@st.cache_data(show_spinner=False)
def calcular(datos, festivos, promociones):
    resultado = pipeline.evaluar(datos, festivos)
    fecha, prediccion = pipeline.predecir_dia_siguiente(datos, festivos, dict(promociones))
    return resultado, fecha, prediccion


st.set_page_config(page_title="Predicción de demanda", page_icon="📈")
st.title("¿Cuánto voy a vender mañana?")
st.write(
    "Sube el historial de ventas diarias de tu negocio. La app aprende de él y estima "
    "las unidades que vas a vender el día siguiente al último día del archivo."
)

with st.expander("¿Qué archivo necesito?"):
    st.markdown(
        "Un archivo **CSV** con una fila por día y categoría de producto, y estas columnas:\n"
        "- `fecha`: el día, en formato AAAA-MM-DD.\n"
        "- `categoria`: el producto o familia de productos.\n"
        "- `unidades`: cuántas unidades se vendieron ese día.\n"
        "- `promocion` (opcional): cuántos productos de esa categoría estaban en promoción.\n\n"
        f"Se necesitan al menos {pipeline.MIN_DIAS} días de ventas."
    )
    st.download_button("Descargar ejemplo de formato", EJEMPLO, "ejemplo_plantilla.csv", "text/csv")

archivo = st.file_uploader("Archivo de ventas (CSV)", type="csv")
pais = st.selectbox("Festivos de qué país tener en cuenta", list(PAISES))

if archivo is None:
    st.info("Sube un archivo para empezar.")
    st.stop()

try:
    datos = pipeline.cargar_plantilla(archivo)
except ValueError as error:
    st.error(str(error))
    st.stop()

categorias = sorted(datos["family"].unique())
st.success(
    f"Archivo leído: {len(categorias)} categorías, del {datos['date'].min():%Y-%m-%d} "
    f"al {datos['date'].max():%Y-%m-%d}."
)

completadas = datos.attrs.get("fechas_completadas", [])
if len(completadas):
    ejemplos = ", ".join(f"{f:%Y-%m-%d}" for f in completadas[:5])
    st.info(
        f"{len(completadas)} días no tenían ninguna venta registrada y se tomaron como venta 0 "
        f"(por ejemplo, días en que el negocio cerró): {ejemplos}"
        + (" y otros." if len(completadas) > 5 else ".")
    )

ultimo_dia = datos[datos["date"] == datos["date"].max()].set_index("family")["onpromotion"]
with st.expander("Promociones planeadas para mañana (opcional)"):
    st.caption("Por defecto se usan las del último día del archivo.")
    promociones = {
        cat: st.number_input(cat, min_value=0, value=int(ultimo_dia.get(cat, 0)), step=1)
        for cat in categorias
    }

if not st.button("Calcular predicción", type="primary"):
    st.stop()

with st.spinner("Entrenando el modelo con tu historial..."):
    resultado, fecha, prediccion = calcular(
        datos, fechas_festivas(pais, datos), tuple(sorted(promociones.items()))
    )

st.metric(f"Ventas estimadas para el {fecha:%Y-%m-%d}", f"{prediccion:,.0f} unidades")

st.subheader("¿Qué tan confiable es?")
st.write(
    f"Para medirlo se simuló su uso en cada una de las últimas {pipeline.N_FOLDS} semanas del "
    f"archivo: el modelo se entrenó solo con los días anteriores a esa semana y luego la "
    f"predijo. Se equivocó en promedio en **{resultado['mae_modelo']:,.0f} unidades "
    f"por día**. La regla simple de repetir lo vendido el mismo día de la semana pasada se "
    f"equivocó en **{resultado['mae_linea_base']:,.0f}**."
)

grafica = resultado["test"].drop(columns="fold").set_index("date").rename(columns={
    "ventas_totales": "Ventas reales",
    "prediccion": "Modelo",
    "linea_base": "Mismo día semana pasada",
})
st.line_chart(grafica, x_label="fecha", y_label="unidades")
