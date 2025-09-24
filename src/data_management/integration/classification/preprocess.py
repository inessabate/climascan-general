import pandas as pd
import geopandas as gpd

def normalize_postal_codes(df_claims: pd.DataFrame, gdf_cp: gpd.GeoDataFrame) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Normaliza códigos postales en claims y CP oficiales.
    - Convierte a string con 5 dígitos.
    - Renombra la columna 'COD_POSTAL' de gdf_cp a 'codigo_postal'.
    """

    df_claims["codigo_postal_norm"] = df_claims["codigo_postal_norm"].astype(str).str.zfill(5)

    if "COD_POSTAL" in gdf_cp.columns:
        gdf_cp = gdf_cp.rename(columns={"COD_POSTAL": "codigo_postal"})

    gdf_cp["codigo_postal"] = gdf_cp["codigo_postal"].astype(str).str.zfill(5)

    return df_claims, gdf_cp


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



def merge_claims_with_cp(df_claims: pd.DataFrame, df_cp: pd.DataFrame) -> pd.DataFrame:
    """
    Une los siniestros con los centroides de CP,
    elimina CP inválidos y estandariza columnas clave.
    """
    # Merge
    df = df_claims.merge(
        df_cp,
        left_on="codigo_postal_norm",
        right_on="codigo_postal",
        how="left"
    )

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

    # Reordenar columnas principales (si existen en df)
    ordered_cols = [
        "siniestro_hash",
        "lob",
        "segmento_cliente_detalle",
        "fecha_ocurrencia",
        "codigo_postal_norm",
        "lat",
        "lon",
        "carga",
    ]
    df = df[[col for col in ordered_cols if col in df.columns]]

    return df


def select_reference_cps(df_claims: pd.DataFrame, df_cp: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona solo los CP de referencia = aquellos donde hay al menos un siniestro.
    Devuelve los centroides de esos CP.
    """
    cps_with_claims = df_claims["codigo_postal_norm"].unique()
    return df_cp[df_cp["codigo_postal"].isin(cps_with_claims)].reset_index(drop=True)


def normalize_dates(df_claims: pd.DataFrame, df_aemet: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convierte columnas de fecha a datetime64 en claims y AEMET.
    """
    df_claims = df_claims.copy()
    df_aemet = df_aemet.copy()
    df_claims["fecha_ocurrencia"] = pd.to_datetime(df_claims["fecha_ocurrencia"])
    df_aemet["fecha"] = pd.to_datetime(df_aemet["fecha"])
    return df_claims, df_aemet


def build_grid(df_claims: pd.DataFrame, df_cp: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """
    Construye la rejilla de referencia (fecha × CP).
    """
    fechas = pd.date_range(start=start_date, end=end_date, freq="D")

    # CP únicos que nos interesan (de claims con centroides válidos)
    cps = df_claims["codigo_postal_norm"].unique()

    # Expandir producto cartesiano fecha × CP
    grid = pd.MultiIndex.from_product([fechas, cps], names=["fecha", "codigo_postal"]).to_frame(index=False)

    # Añadir lat/lon desde centroides
    grid = grid.merge(df_cp, on="codigo_postal", how="left")

    return grid


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
