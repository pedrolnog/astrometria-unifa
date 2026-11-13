"""Cálculos de mecânica orbital e determinação da posição inercial do satélite."""

import math

import numpy as np

from astrometria.coordenadas import Rx, Rz
from astrometria.constantes import MU_TERRA

"""
Função para prever a posição angular num instante a partir da anomalia média
no instante da TLE.

Usa a anomalia média, o tempo da TLE, um instante qualquer e
a velocidade angular média em rad/s
"""
def propagar_anomalia_media(
        anom_media : float,
        tempo_tle : float,
        tempo_instante : float,
        vel_ang_med : float, # Tem que estar em rad/s
) -> float:
    delta_t = (tempo_instante - tempo_tle).total_seconds()

    return ((anom_media + vel_ang_med * delta_t) % (2.0 * math.pi))


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
