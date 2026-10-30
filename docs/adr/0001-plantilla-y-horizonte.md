# ADR 0001: Plantilla de datos y horizonte de predicción

## Estado

Aceptado — 2026-10-30

## Contexto

El proyecto predice demanda diaria por familia de producto, mediante un pipeline reutilizable
para cualquier negocio que use la misma plantilla de datos (ver invariantes del proyecto). Falta
fijar: (1) las columnas de esa plantilla, (2) el horizonte de predicción, y (3) qué variables se
conocen antes del día predicho. Esto bloquea el esqueleto del pipeline (issue #6) y las variables
sin fuga de información (issue #7).

## Decisión

### Plantilla de columnas

| Columna      | Obligatoria | Descripción                                          | Ejemplo en Favorita |
|--------------|-------------|-------------------------------------------------------|----------------------|
| `fecha`      | sí          | fecha diaria de la observación                        | `date`               |
| `categoria`  | sí          | identificador del producto/familia a predecir         | `family`             |
| `unidades`   | sí          | cantidad vendida (no ingresos)                         | `sales`              |
| `promocion`  | no          | indicador/cantidad de unidades en promoción ese día    | `onpromotion`        |

No se incluye `precio`: Favorita no lo trae y no hay un caso de uso inmediato que lo requiera. Si
un negocio futuro lo necesita, se agrega como columna opcional nueva — no se reserva un lugar para
ella en la plantilla actual.

### Horizonte

**1 día**: el pipeline predice `unidades` de la fecha t+1 usando todo lo conocido hasta la fecha t.

Se eligió sobre un horizonte de 7 días por simplicidad: evita decidir entre modelo multi-output,
modelos independientes por día de horizonte o forecast recursivo, y cabe en el tiempo disponible
hasta la entrega del 2026-11-03. El baseline (mismo día de la semana anterior) es compatible con
cualquier horizonte, así que no fue un factor de decisión.

### Variables conocidas antes del día predicho

- **Calendario** (día de la semana, día del mes, días desde la última quincena, `es_festivo`):
  siempre conocidas, se derivan de la fecha.
- **`promocion`**: conocida de antemano — el negocio planea sus promociones con anticipación; en
  Favorita, `test.csv` también la trae para fechas futuras.
- **Rezagos y medias móviles de `unidades`**: solo usan datos de ≥ 1 día atrás, por construcción
  (detalle de implementación y test de no-fuga en el issue #7).

### Explícitamente excluidas del modelo

- **Transacciones**: invariante del proyecto — la tabla de transacciones de Favorita no entra al
  modelo.
- **Precio del petróleo**: variable macro, no se conoce su valor futuro al momento de predecir, y
  su correlación con las ventas es probablemente espuria (EDA temporal, issue #3).

## Consecuencias

- El pipeline (issue #6) recibe un CSV con las columnas `fecha`, `categoria`, `unidades` y,
  opcionalmente, `promocion`.
- Las variables del issue #7 (rezagos de 7/14 días, medias móviles, calendario) se calculan
  agrupando por `categoria`.
- Si se necesita cambiar a un horizonte distinto de 1 día, este ADR debe revisarse — no es una
  decisión implícita del pipeline.

## Nota de actualización — 2026-10-30 (issue #7)

Al construir las variables (issue #7) se decidió que el *target* del modelo es la **venta total
de la tienda** (suma de `unidades` de las `categoria` seleccionadas), no `unidades` por categoría
individual. La plantilla de columnas y el horizonte de este ADR no cambian: siguen describiendo
los datos de entrada por `fecha × categoria`. Lo que cambia es el paso de agregación posterior
— se pivotea esa tabla a una fila por `fecha`, con las features de cada categoría como columnas
separadas y el target como la suma de `unidades` de ese día. Ver la sección "Variables sin fuga
de información" del notebook.
