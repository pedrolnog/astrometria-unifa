from astropy.coordinates import EarthLocation
from skyfield.api import EarthSatellite, load
from datetime import datetime, UTC
from dotenv import load_dotenv
from astropy.time import Time
import astropy.units as u
import numpy as np
import requests
import math
import os

load_dotenv()

# Coordenadas do observador
OBS_LATITUDE=-9.39
OBS_LONGITUDE=-40.50
OBS_ALTITUDE=376 # Altitude em metros

# ID do satélite
NORAD_ID = 47699

API_KEY = os.environ.get("N2YO_API_KEY")

# Constantes
PRIMEIRA_EXC_QUAD_TERRA=0.00669437999

# Semieixo da terra (em km)
SEMIEIXO_MAIOR_TERRA=6378.137
SEMIEIXO_MENOR_TERRA=6356.752


def traduzir_tle(name : str, tle : tuple):
    ts = load.timescale()

    satellite = EarthSatellite(tle[0], tle[1], name, ts)
    return satellite

# Resolve a equação de Kepler usando o método Newton-Raphson
def resolver_kepler(M : float, e : float, tol=1e-10, max_iter=50):
    E = M

    for _ in range(max_iter):
        delta = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E = E - delta
        if abs(delta) < tol:
            return E

    raise RuntimeError("Equação de Kepler não convergiu.")

def calcular_anom_verdadeira(e, E):
    # e: excentricidade orbital
    # E: anomalia excêntrica

    return np.arctan2(math.sqrt(1 - e**2) * math.sin(E), math.cos(E) - e)

# Cria uma das matrizes de transformação para calcular posição inercial
def Rz(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

# Cria uma das matrizes de transformação para calcular posição inercial
def Rx(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c,  s],
        [0.0, -s,  c]
    ])

# Calcula a posição inercial do satélite usando, respectivamente o
# Semieixo maior, excentricidade e anomalia verdadeira
# Longitude do nó ascendente, inclinação e argumento do perigeu
def calcular_pos_inercial(a, e, v, omega, i, w):
    r = (a * (1 - e**2))/(1 + e * math.cos(v))

    r_pqw = np.array([[r * math.cos(v)], [r * math.sin(v)], [0]])

    Q = Rz(-omega) @ Rx(-i) @ Rz(-w)

    r_eci = Q @ r_pqw

    return r_eci

# Calcula o tempo sideral local
def calcular_tempo_sideral(tempo : datetime | None = None):
    lat = OBS_LATITUDE * u.deg
    lon = OBS_LONGITUDE * u.deg

    local = EarthLocation(lat=lat, lon=lon)


    if tempo:
        tempo_utc = tempo.astimezone(UTC)
    else:
        tempo_utc = datetime.now(UTC)

    tempo_obs = Time(tempo_utc, scale="utc", location=local)

    tempo_sideral = tempo_obs.sidereal_time('apparent')

    return tempo_sideral

def calc_r_estacao():
    lat = OBS_LATITUDE * u.deg
    lon = OBS_LONGITUDE * u.deg
    alt = OBS_ALTITUDE * u.m

    tempo_sideral = calcular_tempo_sideral()

    # Raio de curvatura na vertical primeira
    N = SEMIEIXO_MAIOR_TERRA / (math.sqrt(1 - PRIMEIRA_EXC_QUAD_TERRA * (math.sen(lat)**2)))




url = f"https://api.n2yo.com/rest/v1/satellite/tle/{NORAD_ID}&apiKey={API_KEY}"

response = requests.get(url).json()

info = response["info"]
tle = (response["tle"].split("\n")[0], response["tle"].split("\n")[1])


satellite = traduzir_tle(info["satname"], tle)

model = satellite.model