import os
import logging
import pandas as pd
from src.data_management.integration.regression.readers import load_claims, load_aemet, load_postal_codes
from src.data_management.integration.regression.preprocess import (
    normalize_postal_codes,
    compute_centroids,
    merge_claims_with_cp,
    normalize_aemet_coords,
)
from src.data_management.integration.regression.interpolation import interpolate_by_date


# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def run_pipeline(claims_path, aemet_path, cp_base_path, output_path):
    # 1. Cargar
    logger.info("Cargando datos...")
    df_claims = load_claims(claims_path)
    df_aemet = load_aemet(aemet_path)
    gdf_cp = load_postal_codes(cp_base_path)

    # 2. Normalizar
    logger.info("Normalizando códigos postales y calculando centroides...")
    df_claims, gdf_cp = normalize_postal_codes(df_claims, gdf_cp)
    df_cp_centroids = compute_centroids(gdf_cp)
    df_claims_geo = merge_claims_with_cp(df_claims, df_cp_centroids)

    # 3. AEMET → normalización de coordenadas
    logger.info("Normalizando coordenadas de AEMET...")
    df_aemet = normalize_aemet_coords(df_aemet)

    # 4. Interpolación
    logger.info("Interpolando variables meteorológicas...")
    vars_meteo = ["tmed", "tmin", "tmax", "prec",
                  "hrmedia", "hrmax", "hrmin", "velmedia", "racha"]

    df_claims_geo["fecha_ocurrencia"] = pd.to_datetime(df_claims_geo["fecha_ocurrencia"])
    df_aemet["fecha"] = pd.to_datetime(df_aemet["fecha"])

    df_enriched = interpolate_by_date(
        df_claims_geo,
        df_aemet,
        vars_meteo,
        k=5,
        max_radius_km=50
    )

    # 5. Validación rápida de cobertura
    missing = df_enriched[vars_meteo].isna().all(axis=1).sum()
    total = df_enriched.shape[0]
    logger.info(f"{missing}/{total} siniestros sin estaciones válidas dentro del radio.")

    # 6. Export
    logger.info("Exportando dataset final...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_enriched.to_parquet(output_path, index=False)

    logger.info(
        f"✅ Dataset enriquecido exportado a {output_path}, "
        f"con {df_enriched.shape[0]} registros y {df_enriched.shape[1]} columnas."
    )
