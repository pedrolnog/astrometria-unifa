"""Cálculos de observação astrométrica topocêntrica (visibilidade, RA e Dec)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservacaoAstrometrica:
    """Resultado de uma observação topocêntrica em coordenadas equatoriais."""

    visivel: bool
    ra: float  # Ascensão reta em radianos [0, 2*pi)
    dec: float  # Declinação em radianos [-pi/2, pi/2]
    vetor_topocentrico: np.ndarray  # p = r_sat - r_est (km)
    direcao: np.ndarray  # Vetor unitário L = p / ||p||


def calcular_observacao(
    r_sat: np.ndarray,
    r_est: np.ndarray,
) -> ObservacaoAstrometrica:
    """Calcula visibilidade, ascensão reta (RA) e declinação (Dec) do satélite em relação à estação."""
    p = r_sat - r_est
    L = p / np.linalg.norm(p)

    # Produto escalar > 0 significa elevação > 0°
    visivel = bool(np.dot(r_est, p) > 0)

    # Declinação (delta) em [-pi/2, pi/2]
    dec = float(np.arcsin(L[2]))

    # Ascensão Reta (alpha) normalizada para [0, 2*pi)
    ra = float(np.arctan2(L[1], L[0]))
    if ra < 0:
        ra += 2.0 * np.pi

    return ObservacaoAstrometrica(
        visivel=visivel,
        ra=ra,
        dec=dec,
        vetor_topocentrico=p,
        direcao=L,
    )
