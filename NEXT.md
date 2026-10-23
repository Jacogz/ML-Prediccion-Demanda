# NEXT

Actualizado: 2026-10-22
Issue activo: ninguno en curso. Candidato siguiente: #4 (EDA de promociones, ceros y atípicos) o #11 (explicabilidad SHAP, bloqueado por #6/#7)

## Siguiente paso
Issue #3 (EDA temporal) cerrado y mergeado en `main` (PR #10). De aquí en adelante trabajamos directo en `main`, sin ramas/PR, por pedido explícito de Jacobo.

Issue nuevo creado en esta sesión: **#11 — explicabilidad con SHAP sobre ventas por familia de producto**. No cabe en 1 hora tal cual: depende de #6 (esqueleto del pipeline) y #7 (variables sin fuga) primero. Retomar después de esos dos, partiéndolo en un spike más chico (SHAP sobre un modelo baseline) antes de generalizar a las 5 familias.

Decisiones de features que salieron del EDA temporal (para retomar en el #7, "Variables sin fuga de información"):
- Incluir **días desde la última quincena** (calendario, sin fuga).
- Incluir **es_festivo** (calendario/holidays_events.csv, sin fuga).
- **No** incluir petróleo ni transacciones en el modelo: transacciones por invariante del proyecto (esa tabla no entra al modelo); petróleo porque la correlación con ventas (-0.497) es probablemente espuria (ambas series comparten tendencia temporal 2013-2016, no hay relación causal clara) y su valor futuro tampoco se conoce al predecir.

Pendiente menor (no bloqueante): en `model_development.ipynb`, sección "EDA temporal", los insights de "día de la semana" y "transacciones en el tiempo" quedaron con redacción un poco imprecisa (el primero subestima la magnitud del patrón de fin de semana y le falta una decisión de feature explícita; el segundo mezcla "ventas" y "transacciones" en el texto). Se puede ajustar en cualquier momento, no afecta el contenido de las gráficas ni las decisiones de arriba.

## Estado
- Issue #1 (datos) cerrado y en main: token de Kaggle vía `getpass` → `~/.kaggle/access_token`, descarga de `store-sales-time-series-forecasting`, carga con pandas. Pendiente que Nicolás lo corra en su entorno.
- Issue #2 (tienda/familias) cerrado y en main.
- Issue #3 (EDA temporal) cerrado y en main.
- Issue #11 (explicabilidad SHAP) abierto, bloqueado por #6 y #7.
- Entrega 1 (repo + notebook + video) el 2026-11-03.

## Bloqueos y dudas
- Falta definir con Nicolás el reparto de issues.
- Decidir cuándo ajustar los dos insights con redacción imprecisa (día de semana, transacciones) en el EDA temporal.
