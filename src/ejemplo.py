"""Historial de ventas sintético para probar la app sin datos propios.

Imita la forma de la tienda 44 de Favorita (5 familias, unos 4 años), pero los números son
inventados: no redistribuye datos de Kaggle. Tiene lo que el modelo sabe aprovechar: patrón
semanal, quincenas, festivos de Colombia, temporada de diciembre, promociones y rachas de días
buenos o malos que duran varias semanas. La tienda cierra el 25 de diciembre y el 1 de enero
(esos días no tienen filas, como en Favorita).

Uso: python -m src.ejemplo > ventas_ejemplo.csv
"""

import holidays
import numpy as np
import pandas as pd

INICIO, FIN = "2021-01-01", "2024-11-14"
SEMILLA = 44

# Venta diaria promedio y sensibilidad a las promociones de cada familia.
FAMILIAS = {
    "Frutas y verduras": (12000, 0.010),
    "Abarrotes": (9000, 0.006),
    "Bebidas": (7000, 0.012),
    "Lácteos": (2800, 0.008),
    "Aseo": (1400, 0.015),
}
DIA_SEMANA = np.array([0.90, 0.85, 0.88, 0.95, 1.05, 1.30, 1.15])  # lunes ... domingo


def generar_ventas_ejemplo(semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range(INICIO, FIN, freq="D")
    fechas = fechas[~((fechas.month == 12) & (fechas.day == 25) | (fechas.month == 1) & (fechas.day == 1))]
    festivos = pd.to_datetime(list(holidays.country_holidays("CO", years=range(2021, 2025))))

    dia_mes = fechas.day
    quincena = np.where((dia_mes >= 15) & (dia_mes <= 17) | (dia_mes >= 30) | (dia_mes <= 2), 1.10, 1.0)
    diciembre = np.where(fechas.month == 12, 1.0 + 0.25 * dia_mes / 31, 1.0)
    festivo = np.where(fechas.isin(festivos), 1.12, 1.0)
    tendencia = 1.0 + 0.04 * np.arange(len(fechas)) / 365  # crece un 4 % al año
    calendario = DIA_SEMANA[fechas.day_of_week] * quincena * diciembre * festivo * tendencia

    # Rachas compartidas por toda la tienda (clima, economía): AR(1) sobre el logaritmo.
    racha_tienda = _ar1(rng, len(fechas), phi=0.97, sigma=0.015)

    filas = []
    for familia, (nivel, efecto_promo) in FAMILIAS.items():
        promocion = rng.poisson(rng.uniform(2, 15, len(fechas)))
        racha = racha_tienda + _ar1(rng, len(fechas), phi=0.95, sigma=0.02)
        ruido = rng.normal(0, 0.04, len(fechas))
        ventas = nivel * calendario * (1 + efecto_promo * promocion) * np.exp(racha + ruido)
        filas.append(pd.DataFrame({
            "fecha": fechas.strftime("%Y-%m-%d"),
            "categoria": familia,
            "unidades": ventas.round().astype(int),
            "promocion": promocion,
        }))
    return pd.concat(filas, ignore_index=True).sort_values(["fecha", "categoria"], ignore_index=True)


def _ar1(rng, n, phi, sigma):
    choques = rng.normal(0, sigma, n)
    serie = np.empty(n)
    serie[0] = choques[0]
    for t in range(1, n):
        serie[t] = phi * serie[t - 1] + choques[t]
    return serie


if __name__ == "__main__":
    print(generar_ventas_ejemplo().to_csv(index=False), end="")
