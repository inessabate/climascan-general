import pandas as pd
import geopandas as gpd

# ----------------------------
# Postal Codes
# ----------------------------
def normalize_postal_codes(df_claims: pd.DataFrame, df_cp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Aseguramos nombres consistentes
    if "COD_POSTAL" in df_cp.columns and "codigo_postal" not in df_cp.columns:
        df_cp = df_cp.rename(columns={"COD_POSTAL": "codigo_postal"})

    # Normalizar
    df_claims["codigo_postal_norm"] = df_claims["codigo_postal_norm"].astype(str).str.zfill(5)
    df_cp["codigo_postal"] = df_cp["codigo_postal"].astype(str).str.zfill(5)

    return df_claims, df_cp


def compute_centroids(df_cp: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calcula centroides de los polígonos de códigos postales.
    - Reproyecta a CRS métrico (UTM 30N España).
    - Calcula el área y centroides.
    - Selecciona el polígono de mayor área por código postal.
    - Devuelve CP único con lat/lon en WGS84.
    """
    # Reproyectar a CRS métrico
    gdf = df_cp.to_crs(epsg=25830)

    # Calcular área y centroides
    gdf["area"] = gdf.geometry.area
    gdf["centroid"] = gdf.geometry.centroid

    # Quedarnos con el polígono más grande por CP
    gdf = gdf.loc[gdf.groupby("codigo_postal")["area"].idxmax()]

    # Reproyectar a WGS84
    gdf = gdf.set_geometry("centroid").to_crs(epsg=4326)

    # Extraer lat/lon
    gdf["lat"] = gdf.geometry.y
    gdf["lon"] = gdf.geometry.x

    # Tabla limpia
    df_cp = gdf[["codigo_postal", "lat", "lon"]]

    return df_cp


def merge_claims_with_cp(df_claims, df_cp):
    """
    Une los siniestros con los centroides de CP,
    limpia CP inválidos y estandariza las columnas.
    """
    # Merge
    df = df_claims.merge(df_cp,left_on="codigo_postal_norm",right_on="codigo_postal",how="left")

    # Filtrar los siniestros que no tienen centroides válidos
    df = df.dropna(subset=["lat", "lon"]).copy()

    # Renombrar columnas
    rename_map = {
        "LOB ID": "lob",
        "Fecha ocurrencia ID": "fecha_ocurrencia",
        "Estructura Unificada (Segmento Cliente Detalle) ID": "segmento_cliente_detalle",
        "Carga": "carga",
    }
    df = df.rename(columns=rename_map)

    # Eliminar columnas innecesarias
    cols_to_drop = [
        "Código Postal-Población (siniestro) ID_código_postal_siniestro",
        "codigo_postal",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Reordenar columnas principales
    ordered_cols = ["siniestro_hash", "lob", "segmento_cliente_detalle", "fecha_ocurrencia", "codigo_postal_norm", "lat", "lon", "carga"]
    df = df[[col for col in ordered_cols if col in df.columns]]

    return df


# ----------------------------
# AEMET coordinates
# ----------------------------
def parse_coord(coord_str: str) -> float:
    """
    Convierte coordenadas AEMET tipo '390806N', '003123W' a float decimal.
    """
    value, direction = coord_str[:-1], coord_str[-1]  # separar número y dirección
    value = float(value) / 10000  # '390806' → 39.0806
    if direction in ["S", "W"]:
        value = -value
    return value


def normalize_aemet_coords(df_aemet: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza las columnas latitud/longitud de AEMET a grados decimales.
    """
    df_aemet["latitud"] = df_aemet["latitud"].apply(parse_coord)
    df_aemet["longitud"] = df_aemet["longitud"].apply(parse_coord)
    return df_aemet
