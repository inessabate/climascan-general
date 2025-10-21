

def generar_mapas_provincias(lista_provincias, ruta_csv, ruta_geojsons, ruta_salida="data/output/map_classification"):
    """
    Genera un mapa de probabilidad de siniestro por provincia a partir de archivos GeoJSON y CSV.
    Las zonas sin datos se pintan en blanco.
    """

    # Crear carpeta de salida si no existe
    Path(ruta_salida).mkdir(parents=True, exist_ok=True)

    # Cargar CSV de predicciones
    df = pd.read_csv(ruta_csv)
    df["codigo_postal"] = df["codigo_postal"].astype(str).str.zfill(5)

    print(f"CSV cargado con {len(df)} filas.")

    for provincia in lista_provincias:
        nombre_geojson = f"{provincia.upper()}.geojson"
        path_geojson = Path(ruta_geojsons) / nombre_geojson

        if not path_geojson.exists():
            print(f"Archivo no encontrado para {provincia}: {path_geojson}")
            continue

        print(f"Generando mapa para {provincia}...")

        # Cargar GeoJSON
        gdf = gpd.read_file(path_geojson)
        gdf["COD_POSTAL"] = gdf["COD_POSTAL"].astype(str).str.zfill(5)

        # Filtrar CSV por prefijo de provincia
        prefijo = gdf["COD_POSTAL"].iloc[0][:2]
        df_prov = df[df["codigo_postal"].str.startswith(prefijo)]

        # Unir datasets
        gdf = gdf.merge(df_prov, left_on="COD_POSTAL", right_on="codigo_postal", how="left")

        # Limpiar y asegurar tipos
        gdf = gdf[gdf.geometry.notnull()].copy()
        for col in gdf.columns:
            if col != "geometry":
                if pd.api.types.is_datetime64_any_dtype(gdf[col]) or gdf[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
                    gdf[col] = gdf[col].astype(str)

        # Calcular centro de la provincia
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
        gdf_proj = gdf.to_crs(epsg=3857)
        centroid = gdf_proj.geometry.centroid.to_crs(epsg=4326)
        lat, lon = centroid.y.mean(), centroid.x.mean()

        # Crear mapa base
        m = folium.Map(location=[lat, lon], zoom_start=9, tiles="cartodb positron")

        # Añadir capa choropleth con color blanco para sin datos
        folium.Choropleth(
            geo_data=gdf.to_json(),
            name="Probabilidad de siniestro",
            data=gdf,
            columns=["COD_POSTAL", "prob_ocurre_siniestro"],
            key_on="feature.properties.COD_POSTAL",
            fill_color="YlOrRd",
            fill_opacity=0.8,
            line_opacity=0.3,
            nan_fill_color="white",   # ⬜ blanco para sin datos
            nan_fill_opacity=1.0,
            legend_name="Probabilidad de siniestro"
        ).add_to(m)

        # Añadir recuadro 'Sin datos' (blanco)
        legend_html = """
        <div style="position: fixed;
             bottom: 50px; left: 50px; width: 150px; height: 40px;
             background-color: white; z-index:9999; font-size:14px;
             border:2px solid grey; border-radius:8px; padding: 6px;">
          <b>⬜ Sin datos</b>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        # Tooltips
        folium.GeoJson(
            gdf.to_json(),
            name="tooltips",
            tooltip=folium.GeoJsonTooltip(
                fields=["COD_POSTAL", "prob_ocurre_siniestro"],
                aliases=["Código postal", "Probabilidad"],
                localize=True
            )
        ).add_to(m)

        # Guardar mapa
        output_file = Path(ruta_salida) / f"mapa_{provincia}.html"
        m.save(output_file)
        print(f"Mapa de {provincia} guardado en: {output_file}")

    print("Todos los mapas solicitados se han generado correctamente.")

import pandas as pd
import geopandas as gpd
import folium
from pathlib import Path
import branca


def generar_mapas_coste(lista_provincias, ruta_csv, ruta_geojsons, ruta_salida="data/output/map_regresion", lob_filtrado="Household"):
    """
    Genera un mapa de coste medio por siniestro (modelo GALA) por provincia,
    usando los archivos GeoJSON provinciales. Cada provincia se guarda en un archivo HTML independiente.

    Parámetros:
    -----------
    lista_provincias : list[str]
        Lista con los nombres de las provincias (en mayúsculas), por ejemplo: ["BARCELONA", "GIRONA"].

    ruta_csv : str
        Ruta al CSV con columnas ['codigo_postal_norm', 'segmento_cliente_detalle', 'lob', 'pred_euros'].

    ruta_geojsons : str
        Carpeta donde se encuentran los archivos .geojson de cada provincia.

    ruta_salida : str
        Carpeta donde se guardarán los mapas HTML generados.

    lob_filtrado : str
        Línea de negocio a visualizar ('Household' por defecto).
    """

    Path(ruta_salida).mkdir(parents=True, exist_ok=True)

    # Cargar CSV y filtrar por línea de negocio
    df = pd.read_csv(ruta_csv)
    df["codigo_postal_norm"] = df["codigo_postal_norm"].astype(str).str.zfill(5)
    df["lob"] = df["lob"].astype(str).str.strip().str.lower()
    df_filtrado = df[df["lob"] == lob_filtrado.lower()].copy()

    if df_filtrado.empty:
        raise ValueError(f"No se encontraron registros con lob = '{lob_filtrado}' en el CSV.")

    print(f"Cargados {len(df_filtrado)} registros para el segmento '{lob_filtrado}'.")

    # Recorrer provincias
    for provincia in lista_provincias:
        nombre_geojson = f"{provincia.upper()}.geojson"
        path_geojson = Path(ruta_geojsons) / nombre_geojson

        if not path_geojson.exists():
            print(f"Archivo no encontrado para {provincia}: {path_geojson}")
            continue

        print(f"Generando mapa de coste medio para {provincia}...")

        # Cargar GeoJSON de la provincia
        gdf = gpd.read_file(path_geojson)
        gdf["COD_POSTAL"] = gdf["COD_POSTAL"].astype(str).str.zfill(5)

        # Filtrar CSV por prefijo de código postal (2 dígitos iniciales)
        prefijo = gdf["COD_POSTAL"].iloc[0][:2]
        df_prov = df_filtrado[df_filtrado["codigo_postal_norm"].str.startswith(prefijo)]

        # Unir datasets
        gdf = gdf.merge(df_prov, left_on="COD_POSTAL", right_on="codigo_postal_norm", how="left")
        gdf = gdf[gdf.geometry.notnull()].copy()

        # Convertir columnas no serializables a string
        for col in gdf.columns:
            if col != "geometry":
                gdf[col] = gdf[col].apply(
                    lambda x: str(x)
                    if not isinstance(x, (int, float, str, type(None)))
                    else x
                )

        # Calcular centro de la provincia
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
        gdf_proj = gdf.to_crs(epsg=3857)
        centroid = gdf_proj.geometry.centroid.to_crs(epsg=4326)
        lat, lon = centroid.y.mean(), centroid.x.mean()

        # Crear mapa base
        m = folium.Map(location=[lat, lon], zoom_start=9, tiles="cartodb positron")

        # Añadir capa coroplética
        folium.Choropleth(
            geo_data=gdf.to_json(),
            name="Coste medio por siniestro (€)",
            data=gdf,
            columns=["COD_POSTAL", "pred_euros"],
            key_on="feature.properties.COD_POSTAL",
            fill_color="YlOrRd",
            fill_opacity=0.8,
            line_opacity=0.3,
            nan_fill_color="white",       # blanco para sin datos
            nan_fill_opacity=1.0,
            legend_name="Coste medio por siniestro (€)",
            threshold_scale=[0, 500, 1000, 1500, 2000, 3000]
        ).add_to(m)

        # Añadir tooltip
        folium.GeoJson(
            gdf.to_json(),
            name="tooltips",
            style_function=lambda feat: {
                "fillColor": "transparent",
                "color": "transparent",
                "weight": 0,
                "fillOpacity": 0,
            },
            highlight_function=lambda feat: {
                "color": "#444444",
                "weight": 1.5,
                "fillOpacity": 0,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["COD_POSTAL", "pred_euros"],  # o prob_ocurre_siniestro
                aliases=["Código postal", "Coste medio (€)"],
                localize=True
            )
        ).add_to(m)

        # Leyenda de "sin datos"
        legend_html = """
        <div style="position: fixed;
             bottom: 50px; left: 50px; width: 160px; height: 40px;
             background-color: white; z-index:9999; font-size:14px;
             border:2px solid grey; border-radius:8px; padding: 6px;">
          <b>⬜ Sin datos</b>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        # Título superpuesto
        title_html = f"""
            <h4 style="position: fixed; top: 10px; left: 50px; z-index:9999;
            background-color: white; border-radius: 8px; padding: 6px;
            border:1px solid grey; font-size:16px;">
            Coste medio por siniestro – {provincia}
            </h4>
        """
        m.get_root().html.add_child(folium.Element(title_html))

        # Guardar el mapa de la provincia
        output_file = Path(ruta_salida) / f"mapa_coste_{provincia}.html"
        m.save(output_file)
        print(f"Mapa de {provincia} guardado en: {output_file}")

    print("Todos los mapas provinciales del modelo de regresion se han generado correctamente.")