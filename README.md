# ML Predicción de Demanda

Predicción de la venta total del día siguiente de un supermercado, con datos de Store Sales (Favorita), mediante un pipeline reutilizable para cualquier negocio que use la misma plantilla de datos.

**App:** https://fzlsgz3jep2ztsjwtfz3ze.streamlit.app/

## Problema

Un supermercado decide cada día cuánto pedir, preparar y exhibir para mañana, y cuánto personal poner. El modelo le da al administrador de la tienda (o al dueño de un negocio pequeño) la venta estimada del día siguiente y cuánto se suele equivocar, para que no se quede corto ni pierda producto. Se entrena con la tienda 44 de Corporación Favorita (Quito) y sus 5 familias de mayor venta, presentada como "el supermercado cliente". El detalle está en la sección "Problema" de `src/model_development.ipynb`, y la estrategia de reajuste al final del mismo notebook.

## Configuración (una sola vez por integrante)

1. Crear una cuenta en [Kaggle](https://www.kaggle.com) si no tienes una.
2. Aceptar las reglas de la competencia [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules).
3. En Kaggle, ir a https://www.kaggle.com/settings/api → **Generate New Token**. El token (empieza con `KGAT_`) se muestra una sola vez: guárdalo en un lugar seguro, no lo subas a este repo ni a ningún otro.

## Uso

Abre `src/model_development.ipynb` en VS Code con la extensión de Colab, conectado a un runtime de Colab. Al correr la celda de credenciales, te va a pedir que pegues el token del paso anterior con `getpass` (queda solo en el runtime remoto, nunca pasa por este repo). No se usa `files.upload()` porque depende de la interfaz web de Colab y no funciona desde la extensión de VS Code; tampoco se usa `kaggle.json` con `username`+`key`, porque el CLI de `kaggle` migró a este token único.

El resto del notebook descarga los datos a `/content/data` dentro del runtime y los carga con pandas.

## Pipeline y app

El pipeline (`src/pipeline.py`) recibe un CSV con la plantilla del ADR 0001 (`fecha`, `categoria`, `unidades`, `promocion` opcional), crea las variables, entrena y predice la venta total del día siguiente.

```bash
pip install -r requirements.txt pytest
python -m src.pipeline --data tienda44_original.csv   # MAE línea base, MAE modelo y predicción
pytest                                                # división temporal y no-fuga
streamlit run app.py                                  # interfaz en http://localhost:8501
```

Los CSV de la tienda 44 los genera el notebook (sección "Archivos para la app") en `/content/data`. No se suben al repo.

### Despliegue en Streamlit Community Cloud

1. Entrar a https://share.streamlit.io con la cuenta de GitHub que tiene acceso a este repo.
2. **Create app** → **Deploy a public app from GitHub**.
3. Repositorio `Jacogz/ML-Prediccion-Demanda`, rama `main`, archivo `app.py`.
4. **Deploy**. Instala `requirements.txt` solo.
5. Abrir la URL en una ventana de incógnito y subir uno de los CSV.
