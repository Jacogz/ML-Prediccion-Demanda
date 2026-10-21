# ML Predicción de Demanda

Predicción de demanda diaria por familia de producto, con datos de Store Sales (Favorita), mediante un pipeline reutilizable para cualquier negocio que use la misma plantilla de datos.

## Configuración (una sola vez por integrante)

1. Crear una cuenta en [Kaggle](https://www.kaggle.com) si no tienes una.
2. Aceptar las reglas de la competencia [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules).
3. En Kaggle, ir a https://www.kaggle.com/settings/api → **Generate New Token**. El token (empieza con `KGAT_`) se muestra una sola vez: guárdalo en un lugar seguro, no lo subas a este repo ni a ningún otro.

## Uso

Abre `src/model_development.ipynb` en VS Code con la extensión de Colab, conectado a un runtime de Colab. Al correr la celda de credenciales, te va a pedir que pegues el token del paso anterior con `getpass` (queda solo en el runtime remoto, nunca pasa por este repo). No se usa `files.upload()` porque depende de la interfaz web de Colab y no funciona desde la extensión de VS Code; tampoco se usa `kaggle.json` con `username`+`key`, porque el CLI de `kaggle` migró a este token único.

El resto del notebook descarga los datos a `/content/data` dentro del runtime y los carga con pandas.
