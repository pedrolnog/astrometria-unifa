"""Cálculos de mecânica orbital e determinação da posição inercial do satélite."""

import math

import numpy as np

from src.astrometria.constantes import MU_TERRA
from src.astrometria.coordenadas import Rx, Rz

# Próx. passo
def propagar_anomalia_media():
    pass

def calcular_semieixo_maior(
    movimento_medio_rad_min: float,
    mu: float = MU_TERRA,
) -> float:
    """Calcula o semieixo maior (em km) a partir do movimento médio (rad/min)."""
    n_rad_s = movimento_medio_rad_min / 60.0
    return (mu / (n_rad_s**2)) ** (1.0 / 3.0)


def resolver_kepler(
    M: float,
    e: float,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> float:
    """Resolve a equação de Kepler (M = E - e*sin(E)) via método de Newton-Raphson."""
    E = M

    for _ in range(max_iter):
        delta = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= delta
        if abs(delta) < tol:
            return E

    raise RuntimeError("Equação de Kepler não convergiu.")


def calcular_anom_verdadeira(e: float, E: float) -> float:
    """Calcula a anomalia verdadeira (radianos) a partir da excentricidade e anomalia excêntrica."""
    return float(np.arctan2(math.sqrt(1.0 - e**2) * math.sin(E), math.cos(E) - e))
