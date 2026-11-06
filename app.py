"""Interfaz de predicción de demanda (issue #9). Ejecutar con: streamlit run app.py"""

from pathlib import Path

import altair as alt
import holidays
import pandas as pd
import streamlit as st

from src import ejemplo, pipeline

PAISES = {"Ninguno": None, "Colombia": "CO", "Ecuador": "EC"}
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]
MAX_FAMILIAS_EN_GRAFICA = 10
REGLA_SIMPLE = "Regla simple (mismo día de la semana pasada)"
# Tienda 44 de Favorita con los meses anómalos reparados (la exporta el notebook). Si el archivo no
# está en el repo, la app no ofrece su descarga.
TIENDA_44 = Path(__file__).parent / "ejemplos" / "tienda44_reparada.csv"

EJEMPLO = """fecha,categoria,unidades,promocion
2024-01-01,Pan,120,0
2024-01-01,Bebidas,80,2
2024-01-02,Pan,135,0
2024-01-02,Bebidas,75,0
"""


def unidades(valor):
    return f"{valor:,.0f}".replace(",", ".")


def porcentaje(valor):
    return f"{valor:.1f}".replace(".", ",") + " %"


def fecha_larga(fecha):
    return f"{DIAS[fecha.day_of_week]} {fecha.day} de {MESES[fecha.month - 1]} de {fecha.year}"


def fecha_corta(fecha):
    return f"{fecha.day} {MESES[fecha.month - 1][:3]}"


def fechas_festivas(pais, datos):
    if PAISES[pais] is None:
        return frozenset()
    anios = range(datos["date"].dt.year.min(), datos["date"].dt.year.max() + 2)
    return frozenset(pd.to_datetime(list(holidays.country_holidays(PAISES[pais], years=anios))))


@st.cache_data(show_spinner=False)
def datos_de_prueba():
    return ejemplo.generar_ventas_ejemplo().to_csv(index=False)


@st.cache_data(show_spinner=False)
def calcular(datos, festivos, promociones):
    resultado = pipeline.evaluar(datos, festivos)
    explicacion = pipeline.predecir_y_explicar(datos, festivos, dict(promociones))
    return resultado, explicacion


def agrupar_familias_pequenas(atribucion):
    """Con muchas familias, las de menor venta se juntan en una sola barra."""
    familias = atribucion.drop(pipeline.FACTORES_EXTERNOS)
    if len(familias) <= MAX_FAMILIAS_EN_GRAFICA:
        return atribucion
    familias = familias.sort_values("unidades", ascending=False)
    grandes = familias.iloc[: MAX_FAMILIAS_EN_GRAFICA - 1]
    resto = familias.iloc[MAX_FAMILIAS_EN_GRAFICA - 1:]
    otras = pd.DataFrame([resto.sum()], index=[f"Otras {len(resto)} familias"])
    return pd.concat([grandes, otras, atribucion.loc[[pipeline.FACTORES_EXTERNOS]]])


def grafica_atribucion(atribucion):
    tabla = atribucion.reset_index(names="parte")
    tabla["tipo"] = tabla["parte"].map(
        lambda p: "Factores externos" if p == pipeline.FACTORES_EXTERNOS else "Familia de producto"
    )
    tabla["etiqueta"] = [
        f"{porcentaje(p)} · {unidades(u)} u." for p, u in zip(tabla["proporcion"], tabla["unidades"])
    ]
    orden = tabla.sort_values("proporcion", ascending=False)["parte"].tolist()
    base = alt.Chart(tabla).encode(
        y=alt.Y("parte:N", sort=orden, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("proporcion:Q", title="% de la venta estimada"),
        tooltip=[
            alt.Tooltip("parte:N", title="Parte"),
            alt.Tooltip("etiqueta:N", title="Aporte"),
        ],
    )
    barras = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.Color(
            "tipo:N",
            scale=alt.Scale(domain=["Familia de producto", "Factores externos"],
                            range=["#1f6fb4", "#e08a1e"]),
            legend=alt.Legend(title=None, orient="bottom"),
        )
    )
    textos = base.mark_text(align="left", dx=4).encode(text="etiqueta:N")
    return (barras + textos).properties(height=40 * len(tabla) + 40)


st.set_page_config(page_title="Predicción de demanda", page_icon="📈", layout="centered")
st.title("¿Cuánto voy a vender mañana?")
st.write(
    "Sube el historial de ventas diarias de tu negocio. La herramienta aprende de él y te dice "
    "cuánto vas a vender el día siguiente, de dónde sale esa venta y qué tanto puedes confiar "
    "en el número."
)

# --- Paso 1: archivo -------------------------------------------------------------------------
st.header("1. Tu historial de ventas")
with st.expander("¿Qué archivo necesito?"):
    st.markdown(
        "Un archivo **CSV** (se puede guardar así desde Excel) con una fila por día y categoría "
        "de producto, y estas columnas:\n"
        "- `fecha`: el día, en formato AAAA-MM-DD (por ejemplo, 2024-01-31).\n"
        "- `categoria`: el producto o la familia de productos.\n"
        "- `unidades`: cuántas unidades se vendieron ese día.\n"
        "- `promocion` (opcional): cuántos productos de esa categoría estaban en promoción.\n\n"
        f"Se necesitan al menos {pipeline.MIN_DIAS} días de ventas. Entre más historia, mejor."
    )
    columnas = st.columns(3 if TIENDA_44.exists() else 2)
    columnas[0].download_button("Ejemplo del formato", EJEMPLO, "ejemplo_plantilla.csv",
                                "text/csv")
    columnas[1].download_button("Datos de prueba (inventados)", datos_de_prueba(),
                                "ventas_de_prueba.csv", "text/csv")
    texto = (
        "Para ver la herramienta funcionando, descarga un archivo de prueba y súbelo aquí abajo. "
        "**Datos inventados:** casi 4 años de un supermercado con 5 familias de producto, hasta "
        "el 14 de noviembre de 2024; elige los festivos de Colombia."
    )
    if TIENDA_44.exists():
        columnas[2].download_button("Datos reales: tienda 44", TIENDA_44.read_bytes(),
                                    TIENDA_44.name, "text/csv")
        texto += (
            " **Tienda 44:** ventas reales de un supermercado de Corporación Favorita en Quito "
            "(Kaggle), sus 5 familias de mayor venta, de 2013 a agosto de 2017; elige los "
            "festivos de Ecuador."
        )
    st.caption(texto)

archivo = st.file_uploader("Archivo de ventas (CSV)", type="csv")
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
    f"Archivo leído: {len(categorias)} categorías y "
    f"{datos['date'].nunique()} días de ventas, del {fecha_larga(datos['date'].min())} "
    f"al {fecha_larga(datos['date'].max())}."
)

completadas = datos.attrs.get("fechas_completadas", [])
if len(completadas):
    ejemplos = ", ".join(f"{f:%Y-%m-%d}" for f in completadas[:5])
    st.info(
        f"{len(completadas)} días no tenían ninguna venta registrada y se tomaron como venta 0 "
        f"(por ejemplo, días en que el negocio cerró): {ejemplos}"
        + (" y otros." if len(completadas) > 5 else ".")
    )

# --- Paso 2: mañana --------------------------------------------------------------------------
manana = datos["date"].max() + pd.Timedelta(days=1)
st.header("2. Lo que sabes de mañana")
st.caption(f"Mañana es el {fecha_larga(manana)}, el día siguiente al último del archivo.")
pais = st.selectbox(
    "¿De qué país son los festivos que afectan tu negocio?", list(PAISES),
    help="Los festivos cambian lo que se vende. Si no aplica, deja «Ninguno».",
)

ultimo_dia = datos[datos["date"] == datos["date"].max()].set_index("family")["onpromotion"]
with st.expander("Promociones planeadas para mañana (opcional)"):
    st.caption(
        "Cuántos productos de cada categoría vas a tener en promoción. Si no lo cambias, se "
        "usan las del último día del archivo."
    )
    tabla_promociones = st.data_editor(
        pd.DataFrame({
            "Categoría": categorias,
            "Productos en promoción": [int(ultimo_dia.get(c, 0)) for c in categorias],
        }),
        column_config={
            "Categoría": st.column_config.TextColumn(disabled=True),
            "Productos en promoción": st.column_config.NumberColumn(min_value=0, step=1),
        },
        hide_index=True,
        width="stretch",
    )
promociones = dict(zip(
    tabla_promociones["Categoría"], tabla_promociones["Productos en promoción"].fillna(0)
))

if not st.button("Calcular la venta de mañana", type="primary"):
    st.stop()

with st.spinner("Aprendiendo de tu historial... (puede tardar unos segundos)"):
    resultado, explicacion = calcular(
        datos, fechas_festivas(pais, datos), tuple(sorted(promociones.items()))
    )
fecha, prediccion = explicacion["fecha"], explicacion["prediccion"]
error_habitual = resultado["mae_modelo"]

# --- Resultado -------------------------------------------------------------------------------
st.header("3. Resultado")
st.subheader(f"Venta estimada para el {fecha_larga(fecha)}")
col1, col2, col3 = st.columns(3)
col1.metric("Venta total estimada", f"{unidades(prediccion)} u.")
col2.metric("Error habitual", f"± {unidades(error_habitual)} u.",
            help="Cuánto se equivocó el modelo, en promedio por día, en las últimas semanas.")
col3.metric("Error habitual en %", porcentaje(resultado["wape_modelo"]),
            help="El error habitual como porcentaje de lo que se vende (WAPE).")
st.write(
    f"Planea para una venta de alrededor de **{unidades(prediccion)} unidades**. En un día "
    f"típico, la venta real queda entre **{unidades(prediccion - error_habitual)}** y "
    f"**{unidades(prediccion + error_habitual)}** unidades; algunos días el error es mayor."
)

# --- Atribución ------------------------------------------------------------------------------
st.subheader("¿De dónde sale esa venta?")
atribucion = explicacion["atribucion"]
externos = atribucion.loc[pipeline.FACTORES_EXTERNOS]
familias = atribucion.drop(pipeline.FACTORES_EXTERNOS).sort_values("proporcion", ascending=False)
principal = familias.index[0]
st.write(
    f"Según el modelo, **{principal}** explica la mayor parte de la venta de mañana "
    f"(**{porcentaje(familias.loc[principal, 'proporcion'])}**). Cada familia aporta según lo "
    "que ha venido vendiendo y sus promociones. Los **factores externos** (día de la semana, "
    "quincena y festivos) aportan **"
    f"{porcentaje(externos['proporcion'])}** ({unidades(externos['unidades'])} unidades)"
    + (": el calendario de mañana resta ventas frente a un día promedio."
       if externos["unidades"] < 0 else ".")
)
st.altair_chart(grafica_atribucion(agrupar_familias_pequenas(atribucion)),
                width="stretch")

with st.expander("¿Qué hay dentro de los factores externos?"):
    detalle = explicacion["aportes_externos"].rename(
        lambda g: f"Día de la semana ({DIAS[fecha.day_of_week]})" if g == "Día de la semana" else g
    )
    st.dataframe(
        pd.DataFrame({
            "Factor": detalle.index,
            "Unidades que suma o resta": [f"{v:+,.0f}".replace(",", ".") for v in detalle],
        }),
        hide_index=True, width="stretch",
    )
    st.caption("Positivo: ese factor sube la venta de mañana frente a un día promedio. "
               "Negativo: la baja.")

with st.expander("¿Cómo se calcula este reparto?"):
    st.markdown(
        "Se usa **SHAP**, un método que reparte la predicción del modelo entre las variables que "
        "usó. SHAP parte de la venta de un día promedio "
        f"({unidades(explicacion['valor_base'])} unidades) y le suma o resta lo que aporta cada "
        "variable.\n\n"
        "- La venta de un día promedio se reparte entre las familias según cuánto ha vendido cada "
        "una en tu historial.\n"
        "- A cada familia se le suma lo que SHAP le atribuye a sus ventas recientes y sus "
        "promociones.\n"
        "- Los factores externos solo cuentan lo que suman o restan frente a un día promedio; por "
        "eso pueden ser negativos.\n\n"
        "Las partes suman exactamente la venta estimada. Es la explicación **del modelo**, no una "
        "medición de causa y efecto."
    )

# --- Confiabilidad ---------------------------------------------------------------------------
st.subheader("¿Qué tan confiable es?")
st.write(
    f"Para medirlo, se simuló el uso de la herramienta en cada una de las últimas "
    f"{pipeline.N_FOLDS} semanas del archivo: el modelo aprendió solo con los días anteriores a "
    "esa semana y luego predijo cada día. Se compara con una regla simple: vender lo mismo que "
    "el mismo día de la semana pasada."
)
# Cambio relativo del error frente a la regla simple: -20 % es equivocarse un 20 % menos.
cambio = 100 * (resultado["wape_modelo"] / resultado["wape_linea_base"] - 1)
col1, col2 = st.columns(2)
col1.metric("Error del modelo", porcentaje(resultado["wape_modelo"]),
            delta=f"{'-' if cambio < 0 else '+'}{porcentaje(abs(cambio))} de error frente a la "
                  "regla simple",
            delta_color="inverse")
col2.metric("Error de la regla simple", porcentaje(resultado["wape_linea_base"]))
st.caption(
    "Error: cuánto se equivoca en total, como % de lo que se vendió en esas semanas (WAPE). "
    "Entre más bajo, mejor."
)

por_semana = resultado["por_fold"]
st.dataframe(
    pd.DataFrame({
        "Semana": [f"{fecha_corta(d)} – {fecha_corta(h)}"
                   for d, h in zip(por_semana["desde"], por_semana["hasta"])],
        "Error del modelo": por_semana["wape_modelo"].map(porcentaje),
        "Error de la regla simple": por_semana["wape_linea_base"].map(porcentaje),
        "¿Ganó el modelo?": (por_semana["wape_modelo"] < por_semana["wape_linea_base"])
        .map({True: "Sí", False: "No"}),
    }),
    hide_index=True, width="stretch",
)

ultima = por_semana.iloc[-1]
if ultima["wape_modelo"] >= ultima["wape_linea_base"]:
    st.warning(
        "En la semana más reciente el modelo fue peor que la regla simple. Toma la estimación "
        "de mañana con más cautela y vuelve a calcularla cuando tengas más días de ventas."
    )

grafica = resultado["test"].drop(columns="fold").set_index("date").rename(columns={
    "ventas_totales": "Ventas reales",
    "prediccion": "Modelo",
    "linea_base": REGLA_SIMPLE,
})
st.line_chart(grafica, x_label="fecha", y_label="unidades vendidas")
