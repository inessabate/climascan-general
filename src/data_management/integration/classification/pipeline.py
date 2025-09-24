import os
import logging
import pandas as pd
import geopandas as gpd

from src.data_management.integration.classification.readers import (
    load_claims,
    load_aemet,
    load_postal_codes
)
from src.data_management.integration.classification.preprocess import (
    normalize_postal_codes,
    normalize_aemet_coords,
    compute_centroids,
    merge_claims_with_cp,
    build_grid
)
from src.data_management.integration.classification.interpolation import interpolate_weather

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Función para añadir variable objetivo
# -------------------------------------------------
def add_target_variable(df_grid_enriched: pd.DataFrame, df_claims_geo: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la variable objetivo binaria ocurre_siniestro (0/1) al dataset interpolado.
    """
    claims_per_day_cp = (
        df_claims_geo
        .groupby(["fecha_ocurrencia", "codigo_postal_norm"])
        .size()
        .reset_index(name="num_siniestros")
    )

    claims_per_day_cp["ocurre_siniestro"] = (claims_per_day_cp["num_siniestros"] > 0).astype(int)

    df_final = df_grid_enriched.merge(
        claims_per_day_cp[["fecha_ocurrencia", "codigo_postal_norm", "ocurre_siniestro"]],
        left_on=["fecha", "codigo_postal"],
        right_on=["fecha_ocurrencia", "codigo_postal_norm"],
        how="left"
    )

    df_final["ocurre_siniestro"] = df_final["ocurre_siniestro"].fillna(0).astype(int)

    # Limpiar columnas duplicadas
    df_final = df_final.drop(columns=["fecha_ocurrencia", "codigo_postal_norm"], errors="ignore")

    return df_final


# -------------------------------------------------
# Pipeline principal
# -------------------------------------------------
def run_pipeline(claims_path: str, aemet_path: str, cp_base_path: str, output_path: str,
                 year: int = None, k: int = 5, max_radius_km: float = 50.0, n_jobs: int = -1):
    """
    Ejecuta la pipeline de clasificación:
    - Carga de datos
    - Preprocesado (coords, centroides, normalización)
    - Construcción de rejilla espacio-temporal
    - Interpolación meteorológica (paralelizada)
    - Creación de variable objetivo (ocurre_siniestro)
    - Exportación a capa aggregated
    """

    logger.info("Cargando datos...")
    df_claims = load_claims(claims_path)
    df_aemet = load_aemet(aemet_path)
    gdf_cp = load_postal_codes(cp_base_path)

    # Normalizar coordenadas AEMET
    df_aemet = normalize_aemet_coords(df_aemet)

    # Filtro opcional por año
    if year:
        df_claims["Fecha ocurrencia ID"] = pd.to_datetime(df_claims["Fecha ocurrencia ID"])
        df_claims = df_claims[df_claims["Fecha ocurrencia ID"].dt.year == year].copy()
        df_aemet["fecha"] = pd.to_datetime(df_aemet["fecha"])
        df_aemet = df_aemet[df_aemet["fecha"].dt.year == year].copy()
        logger.info(f"Filtrado año {year}: {df_claims.shape[0]} claims, {df_aemet.shape[0]} observaciones AEMET.")

    # Normalizar CP y centroides
    df_claims, gdf_cp = normalize_postal_codes(df_claims, gdf_cp)
    df_cp_centroids = compute_centroids(gdf_cp)
    df_claims_geo = merge_claims_with_cp(df_claims, df_cp_centroids)

    # Construir rejilla espacio-temporal
    start_date = df_claims_geo["fecha_ocurrencia"].min()
    end_date = df_claims_geo["fecha_ocurrencia"].max()
    df_grid = build_grid(df_claims_geo, df_cp_centroids, start_date, end_date)

    logger.info(f"Rejilla creada: {df_grid.shape[0]} combinaciones fecha-CP")

    # Interpolación meteorológica (paralelizada)
    vars_meteo = ["tmed", "tmin", "tmax", "prec",
                  "hrmedia", "hrmax", "hrmin",
                  "velmedia", "racha"]

    df_grid_enriched = interpolate_weather(
        df_grid=df_grid,
        df_aemet=df_aemet,
        vars_meteo=vars_meteo,
        k=k,
        max_radius_km=max_radius_km,
        n_jobs=n_jobs  # paralelización
    )

    # Añadir variable objetivo
    df_final = add_target_variable(df_grid_enriched, df_claims_geo)

    # Exportar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_parquet(output_path, index=False)

    logger.info(f"✅ Dataset de clasificación exportado a {output_path}, "
                f"con {df_final.shape[0]} registros y {df_final.shape[1]} columnas.")
