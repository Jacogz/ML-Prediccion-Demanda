# NEXT

Actualizado: 2026-11-02
Issue activo: #9 (despliegue) y #6 (pipeline), hechos por Nicolás, pendientes de revisión. Candidato siguiente: modelo final en el notebook (Jacobo), luego #11 (SHAP).

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
- **Fechas faltantes (25 de diciembre):** `shift(7)` cuenta filas, no días. Con un día faltante,
  `lag_7` de la semana siguiente mira 8 días atrás. Pequeño en Favorita, puede pesar en un
  negocio con huecos. No se cambió para no diferir del notebook.

## Pendientes menores (heredados, no bloquean nada)

- En la sección EDA temporal, los insights de "día de la semana" y "transacciones" tienen
  redacción un poco imprecisa.

## Entrega

Entrega 1 (repo + notebook + video) el 2026-11-03.
