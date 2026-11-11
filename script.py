from skyfield.api import EarthSatellite, load
from dotenv import load_dotenv
import numpy as np
import requests
import math
import os

def traduzir_tle(name : str, tle : tuple):
    ts = load.timescale()

    satellite = EarthSatellite(tle[0], tle[1], name, ts)
    return satellite

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

def Rz(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

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

    r_pqw = np.array([r * math.cos(v)], [r * math.sin(v)], [0])

    Q = Rz(-omega) @ Rx(-i) @ Rz(-w)

    r_eci = Q @ r_pqw

load_dotenv()

# Coordenadas do observador
OBS_LATITUDE=-9.39
OBS_LONGITUDE=-40.50
OBS_ALTITUDE=376 # Altitude em metros

# ID do satélite
NORAD_ID = 47699

API_KEY = os.environ.get("N2YO_API_KEY")

url = f"https://api.n2yo.com/rest/v1/satellite/tle/{NORAD_ID}&apiKey={API_KEY}"

response = requests.get(url).json()

info = response["info"]
tle = (response["tle"].split("\n")[0][2:], response["tle"].split("\n")[1][2:])


satellite = traduzir_tle(info["satname"], tle)

model = satellite.model

print(f"Número NORAD: {model.satnum}")          # Número de catálogo do satélite
print(f"Inclinação (rad): {model.inclo}")        # Inclinação em radianos
print(f"RAAN (rad): {model.nodeo}")             # Ascensão Reta do Nó Ascendente
print(f"Excentricidade: {model.ecco}")          # Excentricidade orbital
print(f"Argumento do Perigeu: {model.argpo}")   # Argumento do perigeu (rad)
print(f"Anomalia Média: {model.mo}")            # Anomalia média (rad)
print(f"Movimento Médio (rad/min): {model.no_kozai}") # Velocidade angular orbital

anom_excentrica = resolver_kepler(np.radians(model.mo), model.ecco)
anom_verdadeira = calcular_anom_verdadeira(anom_excentrica, model.ecco)

print(anom_verdadeira)

