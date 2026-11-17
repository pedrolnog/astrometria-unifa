"""Ponto de entrada para simulação de passagem e geração de observações astrométricas."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from dotenv import load_dotenv
from skyfield.api import EarthSatellite

from astrometria.constantes import NORAD_ID
from astrometria.coordenadas import calc_r_estacao, calcular_pos_inercial
from astrometria.io_tle import buscar_tle_n2yo, traduzir_tle
from astrometria.observacao import ObservacaoAstrometrica, adicionar_ruido, calcular_observacao
from astrometria.orbita import (
    calcular_anom_verdadeira,
    calcular_semieixo_maior,
    propagar_anomalia_media,
    resolver_kepler,
)


@dataclass(frozen=True)
class RegistroObservacao:
    """Agrupa instante UTC, posição da estação e medições exata e ruidosa."""

    tempo_utc: datetime
    r_est: np.ndarray
    exata: ObservacaoAstrometrica
    ra_obs: float
    dec_obs: float


def simular_instante(
    satellite: EarthSatellite,
    a: float,
    tempo_utc: datetime,
) -> tuple[np.ndarray, ObservacaoAstrometrica]:
    """Propaga o satélite para `tempo_utc` e calcula sua observação topocêntrica exata."""
    model = satellite.model
    tempo_tle = satellite.epoch.utc_datetime()
    dt_segundos = (tempo_utc - tempo_tle).total_seconds()
    n_rad_s = model.no_kozai / 60.0

    anom_media = propagar_anomalia_media(model.mo, n_rad_s, dt_segundos)
    anom_exc = resolver_kepler(anom_media, model.ecco)
    anom_verdadeira = calcular_anom_verdadeira(model.ecco, anom_exc)

    r_sat = calcular_pos_inercial(
        a,
        model.ecco,
        anom_verdadeira,
        model.nodeo,
        model.inclo,
        model.argpo,
    )
    r_est = calc_r_estacao(tempo=tempo_utc)
    return r_est, calcular_observacao(r_sat, r_est)


def coletar_observacoes_passagem(
    satellite: EarthSatellite,
    inicio_utc: datetime,
    num_obs: int = 3,
    espacamento_s: int = 120,
    passo_busca_s: int = 30,
    janela_max_horas: int = 24,
    sigma_arcsec: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[RegistroObservacao]:
    """Busca uma única passagem visível contínua e extrai `num_obs` medições espaçadas."""
    a = calcular_semieixo_maior(satellite.model.no_kozai)
    fim_utc = inicio_utc + timedelta(hours=janela_max_horas)
    tempo_atual = inicio_utc
    registros: list[RegistroObservacao] = []

    while tempo_atual <= fim_utc:
        r_est, obs = simular_instante(satellite, a, tempo_atual)

        if obs.visivel:
            ra_obs, dec_obs = adicionar_ruido(obs.ra, obs.dec, sigma_arcsec, rng)
            registros.append(
                RegistroObservacao(
                    tempo_utc=tempo_atual,
                    r_est=r_est,
                    exata=obs,
                    ra_obs=ra_obs,
                    dec_obs=dec_obs,
                )
            )
            if len(registros) == num_obs:
                return registros
            tempo_atual += timedelta(seconds=espacamento_s)
        else:
            # Garante que todas as observações pertençam à mesma passagem contínua
            registros.clear()
            tempo_atual += timedelta(seconds=passo_busca_s)

    raise RuntimeError(
        f"Nenhuma passagem com {num_obs} observações contíguas encontrada em {janela_max_horas}h."
    )


def main() -> None:
    """Executa a simulação e imprime as 3 observações da primeira passagem encontrada."""
    load_dotenv()

    rng = np.random.default_rng(seed=42)
    sigma_arcsec = 0.0

    nome, tle = buscar_tle_n2yo(NORAD_ID)
    satellite = traduzir_tle(nome, tle)

    registros = coletar_observacoes_passagem(
        satellite=satellite,
        inicio_utc=datetime.now(UTC),
        sigma_arcsec=sigma_arcsec,
        rng=rng,
    )

    for idx, reg in enumerate(registros, start=1):
        print(
            f"[Obs {idx}] {reg.tempo_utc:%Y-%m-%d %H:%M:%S} UTC | "
            f"RA: {np.degrees(reg.ra_obs):.4f}° | "
            f"Dec: {np.degrees(reg.dec_obs):.4f}°"
        )


if __name__ == "__main__":
    main()
