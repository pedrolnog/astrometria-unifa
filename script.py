from skyfield.api import EarthSatellite, load
from dotenv import load_dotenv
import requests
import os

def translate_tle(name : str, tle : tuple):
    ts = load.timescale()

    satellite = EarthSatellite(tle[0], tle[1], name, ts)
    return satellite

load_dotenv()

NORAD_ID = 47699
API_KEY = os.environ.get("N2YO_API_KEY")

# Exemplo consultando a posição para o próximo 1 segundo, assumindo coordenadas do observador zeradas
url = f"https://api.n2yo.com/rest/v1/satellite/tle/{NORAD_ID}&apiKey={API_KEY}"

response = requests.get(url).json()

info = response["info"]
tle = (response["tle"].split("\n")[0][2:], response["tle"].split("\n")[1][2:])


satellite = translate_tle(info["satname"], tle)

# Acessando variáveis específicas do modelo SGP4 gerado pelo TLE
model = satellite.model

print(f"Número NORAD: {model.satnum}")          # Número de catálogo do satélite
print(f"Inclinação (rad): {model.inclo}")        # Inclinação em radianos
print(f"RAAN (rad): {model.nodeo}")             # Ascensão Reta do Nó Ascendente
print(f"Excentricidade: {model.ecco}")          # Excentricidade orbital
print(f"Argumento do Perigeu: {model.argpo}")   # Argumento do perigeu (rad)
print(f"Anomalia Média: {model.mo}")            # Anomalia média (rad)
print(f"Movimento Médio (rad/min): {model.no_kozai}") # Velocidade angular orbital
