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
    dec = float(np.arcsin(np.clip(L[2], -1.0, 1.0)))

    # Ascensão Reta (alpha) normalizada para [0, 2*pi)
    ra = float(np.arctan2(L[1], L[0]) % (2.0 * np.pi))

    return ObservacaoAstrometrica(
        visivel=visivel,
        ra=ra,
        dec=dec,
        vetor_topocentrico=p,
        direcao=L,
    )

def adicionar_ruido(
    ra: float,
    dec: float,
    sigma_arcsec: float,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Adiciona ruído gaussiano isotrópico na esfera celeste às coordenadas (RA, Dec)."""
    if sigma_arcsec <= 0.0:
        return float(ra % (2.0 * np.pi)), float(np.clip(dec, -np.pi / 2.0, np.pi / 2.0))

    if rng is None:
        rng = np.random.default_rng()

    sigma_rad = float(np.radians(sigma_arcsec / 3600.0))
    sigma_ra = sigma_rad / max(float(np.cos(dec)), 1e-6)

    dec_obs = dec + float(rng.normal(0.0, sigma_rad))
    ra_obs = ra + float(rng.normal(0.0, sigma_ra))

    ra_obs = float(ra_obs % (2.0 * np.pi))
    dec_obs = float(np.clip(dec_obs, -np.pi / 2.0, np.pi / 2.0))

    return ra_obs, dec_obs