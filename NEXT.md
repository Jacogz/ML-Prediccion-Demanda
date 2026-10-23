# NEXT

Actualizado: 2026-10-22
Issue activo: #3 (EDA temporal) → PR abierto, pendiente de merge

## Siguiente paso
Issue #3 resuelto en `src/model_development.ipynb`, sección "EDA temporal (issue #3)" (después de `train_1.info()`): PR #10 — https://github.com/Jacogz/ML-Prediccion-Demanda/pull/10. Falta que alguien (Jacobo o Nicolás) lo revise y mergee; `Closes #3` cierra el issue automáticamente al mergear.

Contenido del PR:
- 7 gráficas con insight debajo de cada una: tendencia (con el terremoto de 2016-04-16 y los 6 meses posteriores marcados), día de la semana, patrón quincenal, festivos vs normales, transacciones en el tiempo, petróleo vs ventas/transacciones (scatter, lado a lado), transacciones vs ventas (scatter).
- Ventas agregadas = suma diaria de las 5 familias (una sola serie, no una por familia).
- `oil.csv` se llena con forward fill para fines de semana/feriados.
- También incluye el diff pendiente del issue #2 (selección de tienda/familias) que no se había pusheado antes.

Decisiones de features que salieron de esta sesión (para retomar en el #7, "Variables sin fuga de información"):
- Incluir **días desde la última quincena** (calendario, sin fuga).
- Incluir **es_festivo** (calendario/holidays_events.csv, sin fuga).
- **No** incluir petróleo ni transacciones en el modelo: transacciones por invariante del proyecto (esa tabla no entra al modelo); petróleo porque la correlación con ventas (-0.497) es probablemente espuria (ambas series comparten tendencia temporal 2013-2016, no hay relación causal clara) y su valor futuro tampoco se conoce al predecir.

Pendiente menor (no bloqueante, se decidió dejarlo así por ahora): los insights de "día de la semana" y "transacciones en el tiempo" quedaron con redacción un poco imprecisa (revisados en sesión — el primero subestima la magnitud del patrón de fin de semana y le falta una decisión de feature explícita; el segundo mezcla "ventas" y "transacciones" en el texto). Se puede ajustar en cualquier momento, no afecta el contenido de las gráficas ni las decisiones de arriba.

## Estado
- Issue #1 (datos) cerrado y en main: token de Kaggle vía `getpass` → `~/.kaggle/access_token`, descarga de `store-sales-time-series-forecasting`, carga con pandas. Pendiente que Nicolás lo corra en su entorno.
- Issue #2 (tienda/familias) cerrado en GitHub; el código vive en la misma sección de `model_development.ipynb`, ahora sí pusheado vía el PR #10.
- Issue #3 (EDA temporal): código listo, PR #10 abierto, pendiente merge.
- Entrega 1 (repo + notebook + video) el 2026-11-03.

## Bloqueos y dudas
- Falta definir con Nicolás el reparto de issues.
- Decidir si se ajustan los dos insights con redacción imprecia (día de semana, transacciones) antes o después del merge del #10.
