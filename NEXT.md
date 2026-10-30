# NEXT

Actualizado: 2026-10-30
Issue activo: ninguno en curso. Candidato siguiente: #6 (pipeline, ver más abajo) o #8/#9/#11.

## Cambio de alcance importante (léelo antes de seguir)

El modelo **ya no predice demanda por familia**. Ahora predice la **venta total de la tienda**
(suma de las 5 familias seleccionadas), usando la tendencia de cada familia como señal — por eso
las notas viejas de "SHAP por familia" siguen vigentes, pero el target cambió. Está documentado en:
- La descripción del proyecto (arriba en este archivo, en `CLAUDE.local.md`).
- La nota de actualización al final de `docs/adr/0001-plantilla-y-horizonte.md`.
- La sección "De demanda por familia a venta total de la tienda" en `model_development.ipynb`.

## Qué se cerró en esta sesión

- **#4** (EDA promociones/ceros/atípicos): promoción → se incluye `onpromotion` como feature;
  ceros → se recorta el arranque de cada familia hasta su primera venta (implementado); atípicos
  → se dejan sin modificar.
- **#5** (ADR 0001): plantilla `fecha`, `categoria`, `unidades`, `[promocion]` (sin `precio`),
  horizonte = 1 día. `docs/adr/0001-plantilla-y-horizonte.md`.
- **#7** (variables sin fuga): en `model_development.ipynb` (todavía no en `src/` — se decidió
  priorizar el modelo en notebook antes de retomar el pipeline). Variables: `lag_7`, `lag_14`,
  medias móviles de 7/14 con mínimo/máximo/desviación estándar, `dias_desde_quincena`,
  `es_festivo`, `onpromotion`. Garantía de no-fuga con `groupby("family").shift(1)` antes de
  cualquier ventana, verificada con un test de propiedad (corromper el futuro y recalcular) en dos
  niveles: tabla larga por familia y tabla final pivoteada.
- Tabla de entrenamiento final: una fila por fecha, target = `ventas_totales` (suma de las 5
  familias), features por familia reducidas a `lag_7`, `lag_14`, `rolling_mean_7`,
  `rolling_mean_14`, `onpromotion` (no las 13 completas, para no inflar columnas frente a las
  ~1400 fechas disponibles). Calendario (`dias_desde_quincena`, `es_festivo`) sin duplicar por
  familia.

## Estado del notebook

Todo lo anterior (EDA temporal #3, EDA promociones/ceros/atípicos #4, variables #7, tabla de
entrenamiento) está en `model_development.ipynb`, corrido y verificado por Jacobo en su runtime
de Colab. **Si vas a seguir desde ahí, corre todo el notebook de nuevo de arriba a abajo** — el
recorte de arranque por familia se agregó después de que se corrieran las celdas de atípicos, así
que el orden de ejecución importa.

## Issue #6 (pipeline) — pendiente, no implementado

Modo cambiado de "Solo primero" a "Pareja" en esta sesión. Se dejó para después de resolver el
modelo en notebook. Propuesta hecha (sin confirmar ni escribir código):
- Un solo archivo `src/pipeline.py` por ahora, sin dividir en módulos.
- División temporal con `--test-days N` (default propuesto: 30).
- Baseline: `sales(fecha - 7 días, categoria)`, MAE agregado.
- Test con DataFrame sintético para `max(fecha_train) < min(fecha_test)`.

Cuando se retome: falta decidir si el pipeline reproduce el pivote a venta total (no por
categoría) que se decidió en el #7, y confirmar el default de `--test-days`.

## Pendientes menores (heredados, no bloquean nada)

- En la sección EDA temporal, los insights de "día de la semana" y "transacciones" tienen
  redacción un poco imprecisa (ver sesiones anteriores). El de "día de la semana" tampoco tiene
  una decisión de feature explícita — quedó sin usar, nadie decidió si entra o no al modelo.
- Falta definir con Nicolás el reparto de issues.

## Entrega

Entrega 1 (repo + notebook + video) el 2026-11-03 — quedan 4 días.
