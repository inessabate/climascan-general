import pandas as pd
import geopandas as gpd
import os

def load_claims(path: str) -> pd.DataFrame:
    """Carga datos de siniestros desde un archivo Parquet."""
    return pd.read_parquet(path)

def load_aemet(path: str) -> pd.DataFrame:
    """Carga datos meteorológicos (AEMET) desde un archivo Parquet."""
    return pd.read_parquet(path)

def load_postal_codes(cp_base_path: str) -> gpd.GeoDataFrame:
    """Carga y concatena todos los GeoJSON de códigos postales en un único GeoDataFrame."""
    files = [os.path.join(cp_base_path, f) for f in os.listdir(cp_base_path) if f.endswith(".geojson")]
    gdfs = [gpd.read_file(f) for f in files]
    gdf_cp = pd.concat(gdfs, ignore_index=True)
    return gdf_cp

