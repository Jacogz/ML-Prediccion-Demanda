# NEXT

Actualizado: 2026-10-21
Issue activo: #2 (casi cerrado) → sigue #3

## Siguiente paso
Issue #2 resuelto: tienda y familias elegidas en `src/model_development.ipynb` (celdas tras la sección de datos). Criterio: mayor volumen entre tiendas con <20% de días en cero (umbral laxo porque mezcla las 33 familias); dentro de la tienda, familias con <10% de días en cero.

- **Tienda:** 44 (Quito, tipo A, cluster 5) — volumen 62,087,553, 17.95% días en cero.
- **Familias (5):** GROCERY I, BEVERAGES, PRODUCE, CLEANING, DAIRY.

Falta: Jacobo confirma con Nicolás que esta elección le sirve (o la corre él mismo) y se cierra #2. Sigue el #3: EDA temporal sobre esta tienda/familias.

## Estado
- Issue #1 (datos) cerrado y en main: token de Kaggle vía `getpass` → `~/.kaggle/access_token`, descarga de `store-sales-time-series-forecasting`, carga con pandas. Pendiente que Nicolás lo corra en su entorno.
- Entrega 2026-11-03 (sin cambios).

## Bloqueos y dudas
- Falta definir con Nicolás el reparto de issues.
- Revisar si la fecha límite de EDA (originalmente 2026-10-21) sigue vigente dado el avance real.
