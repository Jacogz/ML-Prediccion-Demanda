# NEXT

Actualizado: 2026-11-03
Issue activo: #11 (SHAP), hecho por Nicolás, pendiente de revisión. Modelo final (#12) hecho por
Jacobo. Falta: publicar la app (#9) y grabar el video.

## SHAP (issue #11, Nicolás, 2026-11-03)

Sección "Explicabilidad con SHAP" al final del notebook, como en el notebook de clase
(`Used_cars_RF_and_GB`): `TreeExplainer` sobre el modelo final (HGBT, datos reparados), entrenado
con el desarrollo y explicado en las 8 semanas de prueba. Aditividad verificada (valor base 32.018
+ suma de SHAP = predicción, diferencia < 1e-10). Summary plot de barras y beeswarm, aporte por
familia (suma de |SHAP| medio de sus columnas) con tabla familia × variable, y force plot del
último día de prueba.
- Más influyentes: `lag_7_PRODUCE` y `dia_semana`; luego BEVERAGES y `dias_desde_quincena`.
- Por grupo: PRODUCE ~5.700 > calendario ~3.900 > BEVERAGES ~3.200 > DAIRY, CLEANING y
  GROCERY I ~1.000 cada una.
- Aún pendiente de la app: Streamlit Cloud no reconoce el rol admin de Nicolás (permisos de
  GitHub guardados antes del cambio). Cerrar sesión en share.streamlit.io y volver a entrar, o que
  Jacobo la publique.

## Revisión de Jacobo (2026-11-02, después del commit de Nicolás)

- **Notebook:** se reintrodujeron dos secciones sobre la versión de Nicolás:
  - "Completar el 25 de diciembre" (antes de "Recorte del arranque"): `train.csv` no trae el
    25-dic; se agrega con `sales=0` y `onpromotion=0`. Si falta otra fecha, la celda falla.
  - "Línea base con validación temporal" (al final, antes de "Notas"): MAE promedio de la línea
    base con `TimeSeriesSplit` en ventanas de 7 días.
  - Falta correrlo completo en un runtime nuevo de Colab.
- **Pipeline:** `cargar_plantilla` ahora llama a `completar_calendario`: una fila por cada día y
  categoría; lo que no tiene registros queda en venta 0. Así `lag_7` siempre mira 7 días atrás
  (antes, el `lag_7` del 31-dic devolvía el 23-dic). Los días completados quedan en
  `datos.attrs["fechas_completadas"]`. 2 tests nuevos; `pytest` en verde (7 tests).
  Esto resuelve la decisión pendiente "Fechas faltantes (25 de diciembre)".

### Pendientes que salieron de la revisión
1. **Interpolación vs 25-dic (decidido: se deja como está).** Si diciembre de 2014 está en
   `meses_a_reparar`, `reparar_con_interpolacion` reemplaza el 0 del 25-dic (tienda cerrada) por
   un valor interpolado en PRODUCE y BEVERAGES. No se corrige: se agregará una nota sobre los
   días faltantes al analizar la diferencia entre los datos reparados y los originales.
2. **Una sola forma de medir el MAE (hecho):** el pipeline ahora usa `TimeSeriesSplit` con
   semanas de prueba de 7 días, como el notebook (`folds_temporales` y `evaluar`). El MAE es el
   promedio simple de los folds; `evaluar` también devuelve `por_fold`. Por velocidad de la app
   usa solo las últimas `N_FOLDS = 4` semanas (CLI: `--folds`); el notebook evalúa la línea base
   con todos los folds posibles. Al comparar el modelo final contra la línea base en el notebook,
   usar los mismos folds para ambos. El 3975 de abajo (últimos 30 días) ya no aplica.
3. **App (hecho):** muestra cuántos días se completaron con venta 0 y explica la validación por
   semanas. Solo se verificó que compila; falta probarla con `streamlit run app.py`.

## Reparto propuesto (por confirmar con Jacobo)
- **Jacobo:** modelo en el notebook (grid search, XGBoost, SHAP).
- **Nicolás:** interpolación de datos, pipeline (`src/pipeline.py`) y app de Streamlit.

## Qué se hizo en esta sesión (Nicolás)

- **Interpolación** (pedido del profe, no eliminar datos): sección "Reparación de datos con
  interpolación" en el notebook, antes de "Notas". Meses con PRODUCE < 1000 de venta media diaria
  después del primer mes normal (2014-01): 10 meses entre 2014-02 y 2015-05. En esos meses se
  marcan como faltantes PRODUCE y BEVERAGES (BEVERAGES cae a la mitad en los mismos meses) y se
  interpolan linealmente por día de la semana (`interpolate(method="time")` por familia y día).
  Resultado: `datos_reparados` (columna `interpolado` marca las filas) y
  `tabla_entrenamiento_reparada`, con el mismo test de no-fuga en verde.
  Falta issue en GitHub para esta tarea (plantilla Tarea, modo Pareja).
- **#6 pipeline:** `src/pipeline.py` con las mismas funciones del notebook (`recortar_arranque`,
  `calcular_features`, `construir_tabla_entrenamiento`), verificado: arma exactamente la misma
  tabla que el notebook con datos originales y reparados. `python -m src.pipeline --data <csv>`
  imprime MAE línea base, MAE modelo y la predicción del día siguiente. `pytest` en verde (5 tests:
  división temporal, no-fuga, plantilla incompleta, filas repetidas, predicción del día siguiente).
- **#9 app:** `app.py` + `requirements.txt`. Sube CSV con la plantilla, festivos por país
  (`holidays`), promociones de mañana editables, predicción del día siguiente, MAE del modelo vs
  línea base en los últimos 30 días con gráfica. Probada localmente. Falta publicarla en
  Streamlit Community Cloud (pasos en el README).
- El notebook exporta `tienda44_original.csv` y `tienda44_reparada.csv` a `/content/data` para
  subirlos a la app (no van al repo).

## Siguiente paso
- Jacobo: elegir modelo final. Cuando esté, cambiar solo `crear_modelo()` en `src/pipeline.py`.
  Si el modelo final agrega o quita variables, cambiarlas en el notebook **y** en
  `src/pipeline.py` (están copiadas, no importadas).
- Nicolás: publicar la app en Streamlit Community Cloud y grabar el video.

## Decisiones pendientes
- **2013 no se interpoló:** PRODUCE en ~30 y BEVERAGES a la mitad todo 2013, pero no hay un mes
  válido antes para interpolar. Opciones: dejarlo así (está ahora) o empezar en 2014-01.
- **Día de la semana:** sigue sin entrar al modelo. Prueba rápida con el Random Forest provisional
  (datos originales, últimos 30 días): MAE 3859 → 3598 al agregar `dia_semana`. La línea base
  da 3975, así que hoy el modelo provisional casi no le gana.
- **Datos reparados vs originales:** con el modelo provisional, los reparados dan MAE 4062 vs
  3859 de los originales. Comparar de nuevo con el modelo final antes de decidir.
- ~~**Fechas faltantes (25 de diciembre)**~~: resuelto en notebook y pipeline (ver "Revisión de
  Jacobo").

## Pendientes menores (heredados, no bloquean nada)

- En la sección EDA temporal, los insights de "día de la semana" y "transacciones" tienen
  redacción un poco imprecisa.

## Entrega

Entrega 1 (repo + notebook + video) el 2026-11-03.
