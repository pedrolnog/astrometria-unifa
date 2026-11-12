"""Leitura e conversão de dados TLE (Two-Line Element)."""

import os

import requests
from skyfield.api import EarthSatellite, load


def traduzir_tle(name: str, tle: tuple[str, str]) -> EarthSatellite:
    """Cria um objeto EarthSatellite da Skyfield a partir do nome e das duas linhas do TLE."""
    ts = load.timescale()
    return EarthSatellite(tle[0], tle[1], name, ts)


def buscar_tle_n2yo(
    norad_id: int,
    api_key: str | None = None,
    timeout: float = 15.0,
) -> tuple[str, tuple[str, str]]:
    """Consulta a API do N2YO e retorna o nome do satélite e o par de linhas do TLE."""
    chave = api_key or os.environ.get("N2YO_API_KEY")
    if not chave:
        raise RuntimeError("Variável de ambiente N2YO_API_KEY não definida.")

    url = f"https://api.n2yo.com/rest/v1/satellite/tle/{norad_id}&apiKey={chave}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    dados = response.json()

    info = dados["info"]
    linhas_tle = dados["tle"].splitlines()
    if len(linhas_tle) < 2:
        raise ValueError(f"TLE inválido retornado pela API para o NORAD ID {norad_id}.")

    return info["satname"], (linhas_tle[0], linhas_tle[1])
