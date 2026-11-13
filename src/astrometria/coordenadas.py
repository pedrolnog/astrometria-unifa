"""Transformações de coordenadas, matrizes de rotação e posição da estação no referencial ECI."""

from datetime import UTC, datetime
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
import math

from astrometria.constantes import (
    OBS_ALTITUDE,
    OBS_LATITUDE,
    OBS_LONGITUDE,
    PRIMEIRA_EXC_QUAD_TERRA,
    SEMIEIXO_MAIOR_TERRA,
)


def Rz(theta: float) -> np.ndarray:
    """Matriz de rotação elementar em torno do eixo Z (ângulo em radianos)."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def Rx(theta: float) -> np.ndarray:
    """Matriz de rotação elementar em torno do eixo X (ângulo em radianos)."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, s],
        [0.0, -s, c],
    ])


def calcular_tempo_sideral(
    tempo: datetime | None = None,
    latitude: float = OBS_LATITUDE,
    longitude: float = OBS_LONGITUDE,
) -> float:
    """Calcula o tempo sideral local aparente (em radianos) para o instante e local informados."""
    instante = tempo if tempo is not None else datetime.now(UTC)
    local = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg)
    tempo_obs = Time(instante, scale="utc", location=local)
    tempo_sideral = tempo_obs.sidereal_time("apparent")
    return float(tempo_sideral.to_value(u.rad))


def calc_r_estacao(
    tempo: datetime | None = None,
    latitude: float = OBS_LATITUDE,
    longitude: float = OBS_LONGITUDE,
    altitude_m: float = OBS_ALTITUDE,
) -> np.ndarray:
    """Calcula o vetor posição da estação observadora no referencial ECI (em km)."""
    lat_rad = np.radians(latitude)
    alt_km = altitude_m / 1000.0

    # O tempo sideral local (em radianos) depende do instante e da longitude
    lst = calcular_tempo_sideral(tempo=tempo, latitude=latitude, longitude=longitude)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    # Raio de curvatura na vertical primeira (em km)
    n_curv = SEMIEIXO_MAIOR_TERRA / np.sqrt(1.0 - PRIMEIRA_EXC_QUAD_TERRA * (sin_lat**2))

    return np.array([
        (n_curv + alt_km) * cos_lat * np.cos(lst),
        (n_curv + alt_km) * cos_lat * np.sin(lst),
        (n_curv * (1.0 - PRIMEIRA_EXC_QUAD_TERRA) + alt_km) * sin_lat,
    ])

def calcular_pos_inercial(
    a: float,
    e: float,
    v: float,
    omega: float,
    i: float,
    w: float,
) -> np.ndarray:
    """Calcula o vetor posição inercial ECI do satélite (em km) a partir dos elementos orbitais."""
    r = (a * (1.0 - e**2)) / (1.0 + e * math.cos(v))
    r_pqw = np.array([r * math.cos(v), r * math.sin(v), 0.0])

    Q = Rz(-omega) @ Rx(-i) @ Rz(-w)
    return Q @ r_pqw