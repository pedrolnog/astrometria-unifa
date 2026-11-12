"""Ponto de entrada para cálculo de posição e observação astrométrica de satélite."""

import numpy as np
from dotenv import load_dotenv

from src.astrometria.constantes import NORAD_ID
from src.astrometria.coordenadas import calc_r_estacao, calcular_pos_inercial
from src.astrometria.io_tle import buscar_tle_n2yo, traduzir_tle
from src.astrometria.observacao import calcular_observacao
from src.astrometria.orbita import (
    calcular_anom_verdadeira,
    calcular_semieixo_maior,
    resolver_kepler,
)


def main() -> None:
    """Executa a obtenção do TLE e o cálculo de visibilidade, RA e Dec."""
    load_dotenv()

    nome, tle = buscar_tle_n2yo(NORAD_ID)
    satellite = traduzir_tle(nome, tle)
    model = satellite.model

    a = calcular_semieixo_maior(model.no_kozai)
    anom_exc = resolver_kepler(model.mo, model.ecco)
    anom_verdadeira = calcular_anom_verdadeira(model.ecco, anom_exc)

    r_sat = calcular_pos_inercial(
        a,
        model.ecco,
        anom_verdadeira,
        model.nodeo,
        model.inclo,
        model.argpo,
    )
    r_est = calc_r_estacao(tempo=satellite.epoch.utc_datetime())

    obs = calcular_observacao(r_sat, r_est)
    print(
        f"Visível: {obs.visivel} | "
        f"RA: {np.degrees(obs.ra):.4f}° | "
        f"Dec: {np.degrees(obs.dec):.4f}°"
    )


if __name__ == "__main__":
    main()
