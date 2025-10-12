import pandas as pd
import geopandas as gpd
import os

def load_claims(path: str) -> pd.DataFrame:
    """Carga datos de siniestros desde parquet."""
    return pd.read_parquet(path)

def load_aemet(path: str) -> pd.DataFrame:
    """Carga datos meteorológicos diarios de AEMET (parquet)."""
    return pd.read_parquet(path)

def load_postal_codes(cp_base_path: str) -> gpd.GeoDataFrame:
    """Carga los GeoJSON de códigos postales (todas las provincias)."""
    files = [os.path.join(cp_base_path, f) for f in os.listdir(cp_base_path) if f.endswith(".geojson")]
    gdfs = [gpd.read_file(f) for f in files]
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
