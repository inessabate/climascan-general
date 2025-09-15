import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

EARTH_RADIUS_KM = 6371.0088

def interpolate_by_date(
    df_claims: pd.DataFrame,
    df_aemet: pd.DataFrame,
    vars_meteo: list[str],
    k: int = 5,
    max_radius_km: float = 50.0,
) -> pd.DataFrame:
    """
    Interpola variables meteorológicas para todos los siniestros, día a día,
    usando k vecinos más cercanos con métrica haversine e IDW (1/dist).
    Aplica radio máximo y maneja NaN de forma robusta (recalcula pesos tras filtrar).

    Requisitos:
      - df_claims: columnas ['fecha_ocurrencia','lat','lon', ...]
      - df_aemet: columnas ['fecha','latitud','longitud', ...vars_meteo]
      - Fechas en datetime64 y coords en grados decimales
    """
    results = []

    # Asegurar tipos
    if not np.issubdtype(df_claims["fecha_ocurrencia"].dtype, np.datetime64):
        df_claims = df_claims.copy()
        df_claims["fecha_ocurrencia"] = pd.to_datetime(df_claims["fecha_ocurrencia"])
    if not np.issubdtype(df_aemet["fecha"].dtype, np.datetime64):
        df_aemet = df_aemet.copy()
        df_aemet["fecha"] = pd.to_datetime(df_aemet["fecha"])

    # Procesar por fecha
    for fecha, claims_day in df_claims.groupby("fecha_ocurrencia", sort=False):
        aemet_day = df_aemet[df_aemet["fecha"] == fecha]
        if aemet_day.empty:
            continue

        # Coordenadas en radianes
        X_aemet = np.radians(aemet_day[["latitud", "longitud"]].to_numpy())
        X_claims = np.radians(claims_day[["lat", "lon"]].to_numpy())

        # Índice KNN (haversine). Obtenemos k vecinas y luego aplicamos radio.
        nbrs = NearestNeighbors(n_neighbors=min(k, len(aemet_day)), metric="haversine")
        nbrs.fit(X_aemet)

        dist_rad, idx = nbrs.kneighbors(X_claims)
        dist_km = dist_rad * EARTH_RADIUS_KM

        # Para mantener relación ordenada
        claims_day = claims_day.reset_index(drop=True)

        # Interpolación por siniestro
        for i, (distances, indices) in enumerate(zip(dist_km, idx)):
            row_claim = claims_day.iloc[i].to_dict()

            # Aplicar radio máximo
            in_radius = distances <= max_radius_km
            if not np.any(in_radius):
                # Ninguna estación válida en radio → todo NaN para este siniestro
                results.append({**row_claim, **{v: np.nan for v in vars_meteo}})
                continue

            # Subconjunto de estaciones dentro de radio
            use_indices = indices[in_radius]
            use_distances = distances[in_radius]

            meteo_vals = {}
            for var in vars_meteo:
                vals = aemet_day.iloc[use_indices][var].to_numpy(dtype=float, copy=False)

                # Filtrar NaN por variable y recalcular pesos
                mask = ~np.isnan(vals)
                if np.any(mask):
                    vvals = vals[mask]
                    vdists = use_distances[mask]

                    # IDW (1/dist) normalizado
                    w = 1.0 / np.maximum(vdists, 1e-12)  # evitar división por 0
                    w /= w.sum()

                    meteo_vals[var] = float(np.average(vvals, weights=w))
                else:
                    meteo_vals[var] = np.nan

            results.append({**row_claim, **meteo_vals})

    return pd.DataFrame(results)
