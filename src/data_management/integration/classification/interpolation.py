import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

EARTH_RADIUS_KM = 6371.0088

def _interpolate_day(grid_day, aemet_day, vars_meteo, k, max_radius_km):
    """
    Interpolación para un solo día (sub-función interna).
    """
    results = []
    if aemet_day.empty:
        return results

    # Coordenadas en radianes
    X_aemet = np.radians(aemet_day[["latitud", "longitud"]].to_numpy(dtype=float))
    X_grid = np.radians(grid_day[["lat", "lon"]].to_numpy(dtype=float))

    # KNN en haversine
    nbrs = NearestNeighbors(n_neighbors=min(k, len(aemet_day)), metric="haversine")
    nbrs.fit(X_aemet)

    dist_rad, idx = nbrs.kneighbors(X_grid)
    dist_km = dist_rad * EARTH_RADIUS_KM

    for i, (distances, indices) in enumerate(zip(dist_km, idx)):
        row = grid_day.iloc[i].to_dict()

        in_radius = distances <= max_radius_km
        if not np.any(in_radius):
            # sin estaciones válidas → NaN
            results.append({**row, **{v: np.nan for v in vars_meteo}})
            continue

        use_indices = indices[in_radius]
        use_distances = distances[in_radius]

        meteo_vals = {}
        for var in vars_meteo:
            vals = aemet_day.iloc[use_indices][var].to_numpy(dtype=float, copy=False)

            mask = ~np.isnan(vals)
            if np.any(mask):
                vvals = vals[mask]
                vdists = use_distances[mask]

                w = 1.0 / np.maximum(vdists, 1e-12)
                w /= w.sum()

                meteo_vals[var] = float(np.average(vvals, weights=w))
            else:
                meteo_vals[var] = np.nan

        results.append({**row, **meteo_vals})

    return results


def interpolate_weather(df_grid, df_aemet, vars_meteo, k=5, max_radius_km=50, n_jobs=-1):
    """
    Interpola variables meteorológicas para CP-fecha usando multiprocessing.
    """
    fechas = df_grid["fecha"].unique()

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_interpolate_day)(
            df_grid[df_grid["fecha"] == fecha],
            df_aemet[df_aemet["fecha"] == fecha],
            vars_meteo,
            k,
            max_radius_km,
        )
        for fecha in fechas
    )

    # Flatten results (lista de listas)
    flat_results = [item for sublist in results for item in sublist]

    return pd.DataFrame(flat_results)
