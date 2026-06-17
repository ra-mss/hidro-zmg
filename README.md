# HidroZMG

Sistema de inteligencia geoespacial para el análisis de riesgo de inundación
urbana en la Zona Metropolitana de Guadalajara.

## Descripción

HidroZMG cruza el **Mapa Único de Inundaciones 2026 (IMEPLAN)** con variables
topohidrológicas derivadas de DEM satelital (SRTM 30m) para:

- Clasificar zonas de riesgo usando un modelo Random Forest
- Identificar los factores topográficos que explican cada sitio de inundación
- Proponer intervenciones de infraestructura verde (jardines de lluvia,
  bioswales, parques inundables)

## Stack tecnológico

- **Datos:** MUI 2026 IMEPLAN, SRTM GL1 30m (OpenTopography), INEGI
- **Procesamiento:** Python, GeoPandas, pysheds, Rasterio
- **Modelo:** Random Forest (scikit-learn): ROC-AUC 0.761
- **Visualización:** Streamlit, Folium, Plotly

## Metodología

Los umbrales de riesgo siguen la clasificación de peligro por inundación
del CENAPRED (2014). El modelo predictivo usa exclusivamente variables
topohidrológicas (TWI, pendiente, elevación) sin incluir la altura
reportada, evitando circularidad metodológica (evitar que el modelo haga 
trampa memorizando) y permitiendo predicción en sitios no catalogados.

## Estructura del proyecto

hidro-zmg/
├── app/
│   ├── app.py
│   └── model/
├── data/
|   ├── raw/
│   └── processed/
├── notebooks/
└── requirements.txt

## Despliegue

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hidro-zmg.streamlit.app/)
