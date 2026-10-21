# NEXT

Actualizado: 2026-10-21
Issue activo: #1

## Siguiente paso
Issue #1 casi cerrado: en `src/model_development.ipynb`, la sección de datos (token de Kaggle vía `getpass` → `~/.kaggle/access_token`, descarga de `store-sales-time-series-forecasting` a `/content/data`, carga con pandas) corre de punta a punta y llega a `train.head()` en un runtime nuevo de Colab (probado por Jacobo). Falta que **Nicolás** la corra en su propio VS Code/Colab sin tocar código, que es el criterio de terminado del issue. Luego sigue #2: elegir tienda y familias.

## Estado
- `kaggle.json` con `username`+`key` ya no sirve: el CLI de `kaggle` migró a un token único (prefijo `KGAT_`, generado en https://www.kaggle.com/settings/api) guardado en `~/.kaggle/access_token`. El README y el notebook ya reflejan esto.
- `files.upload()` no funciona desde la extensión de Colab en VS Code (depende de la interfaz web de Colab); por eso el token se pide con `getpass`.
- README nuevo con la configuración de una sola vez (aceptar reglas, generar token).
- Entrega 2026-11-03 (sin cambios).

## Bloqueos y dudas
- Pendiente: que Nicolás confirme que corre en su entorno sin ayuda.
- Falta definir con Nicolás el reparto de issues.
- Horizonte, tienda y familias se deciden tras el EDA (issue #2), a más tardar el 2026-10-21 — revisar si esta fecha sigue vigente.
