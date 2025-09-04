import os
import sys
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import logging
from src.utils.logging_setup import setup_logging

# --- Configuración de logging ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Utilidad: DMS AEMET -> decimal ---
def aemet_dms_to_decimal(dms: str):
    """
    Convierte coordenadas AEMET en formato DMS compacto (p.ej. '394924N', '025309E')
    a grados decimales. Soporta 2 o 3 dígitos de grados.
    """
    if dms is None:
        return None
    s = str(dms).strip().upper()
    if not s or s[-1] not in ("N","S","E","W"):
        try:
            return float(s)
        except Exception:
            return None
    hemi = s[-1]
    body = s[:-1]
    if len(body) < 6:
        return None
    deg_str = body[:-4]
    min_str = body[-4:-2]
    sec_str = body[-2:]
    try:
        deg = int(deg_str)
        minu = int(min_str)
        sec = int(sec_str)
    except ValueError:
        return None
    sign = -1 if hemi in ("S","W") else 1
    return sign * (deg + minu/60 + sec/3600)


# --- Función principal ---
def add_postcodes(
    input_list: list[dict],
    user_agent: str = "tfm-geocoder",
) -> str:

    df = pd.DataFrame(input_list)

    cols_lower = {c.lower(): c for c in df.columns}
    col_lat = cols_lower.get("latitud") or cols_lower.get("latitude") or "latitud"
    col_lon = cols_lower.get("longitud") or cols_lower.get("longitude") or "longitud"
    col_nombre = cols_lower.get("nombre") or cols_lower.get("estacion") or cols_lower.get("indicativo")

    if col_lat not in df.columns or col_lon not in df.columns:
        raise ValueError(f"No encuentro columnas de coordenadas. Tengo columnas: {list(df.columns)}")

    def _to_dec(x):
        return aemet_dms_to_decimal(x)

    logger.info("Consultando códigos postales…")
    geolocator = Nominatim(user_agent=user_agent)
    geocode = RateLimiter(
        geolocator.reverse,
        min_delay_seconds=1,
        max_retries=2,
        error_wait_seconds=5
    )

    codigos_postales: list[str | None] = []
    total = len(df)
    for i, row in df.iterrows():
        lat = _to_dec(row[col_lat])
        lon = _to_dec(row[col_lon])
        nombre = row[col_nombre] if col_nombre in df.columns else ""
        cp = None
        try:
            if lat is not None and lon is not None:
                location = geocode((lat, lon), language="es", exactly_one=True, addressdetails=True)
                cp = location.raw["address"].get("postcode") if location else None
        except Exception as e:
            logger.error(f"Error geocodificando ({lat}, {lon}): {e}")
            cp = None

        codigos_postales.append(cp)
        etiqueta = nombre if isinstance(nombre, str) and nombre else (row.get("indicativo") or "")
        logger.info(f"[{i + 1}/{total}] {etiqueta} → CP: {cp}")

    df_out = df.copy()
    df_out["codigo_postal"] = codigos_postales

    return df_out

