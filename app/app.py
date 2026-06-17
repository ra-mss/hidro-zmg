import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import DualMap
from streamlit_folium import st_folium
import plotly.express as px
import joblib
import os

st.set_page_config(
    page_title="HidroZMG",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "predicciones_modelo.gpkg")
MODEL_DIR  = os.path.join(BASE_DIR, "app", "model")

COLORES_RIESGO = {
    "Bajo": "#27AE60",
    "Medio": "#F39C12",
    "Alto": "#E74C3C",
    "Crítico": "#8E44AD"
}
ORDEN_RIESGO = ["Bajo", "Medio", "Alto", "Crítico"]

COLORES_INTERVENCION = {
    "Bioswale + rejilla de captación": "#038B4C", 
    "Jardín de lluvia en banqueta": "#BAEC04",
    "Pavimento permeable + árbol de lluvia": "#0198AC",
    "Parque inundable (retention basin)": "#810077",
    "Cuneta verde": "#4000B8"
}

# Carga de datos
@st.cache_data
def load_data():
    gdf = gpd.read_file(DATA_PATH)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    # Simplificar geometrías p acelerar el renderizado
    gdf["geometry"] = gdf["geometry"].simplify(0.0001)
    return gdf

@st.cache_resource
def load_models():
    rf_bin = joblib.load(os.path.join(MODEL_DIR, "rf_binario.pkl"))
    rf_4c = joblib.load(os.path.join(MODEL_DIR, "rf_4clases.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    return rf_bin, rf_4c, le, features

with st.spinner("Cargando datos y modelos..."):
    gdf = load_data()
    rf_bin, rf_4c, le, feature_cols = load_models()

# Header
st.title("HidroZMG")
st.caption(
    "Sistema de inteligencia geoespacial para riesgo de inundación urbana · "
    "ZMG | MUI 2026 IMEPLAN | SRTM 30m | Random Forest"
)
st.divider()

# Sidebar
with st.sidebar:
    st.header("Filtros")
    niveles_sel = st.multiselect(
        "Nivel de riesgo",
        options=ORDEN_RIESGO,
        default=ORDEN_RIESGO
    )
    infras_disponibles = sorted(gdf["infra_norm"].dropna().unique().tolist())
    infras_sel = st.multiselect(
        "Tipo de infraestructura",
        options=infras_disponibles,
        default=infras_disponibles
    )
    basemap_sel = st.radio(
        "Mapa base",
        ["OpenStreetMap", "CartoDB Positron", "CartoDB DarkMatter", "Stamen Terrain"],
        index=2
    )
    st.divider()
    st.markdown("### Acerca del proyecto")
    st.markdown(
        "HidroZMG cruza el catálogo oficial MUI 2026 del IMEPLAN con "
        "variables topohidrológicas derivadas de DEM satelital (SRTM 30m) "
        "para identificar patrones de riesgo y proponer intervenciones de "
        "infraestructura verde urbana."
    )

# Filtrado
mask = (
    gdf["riesgo_final"].isin(niveles_sel) &
    gdf["infra_norm"].isin(infras_sel)
)
gdf_fil = gdf[mask].copy()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "Mapa de riesgo",
    "Análisis del modelo",
    "Motor de intervención"
])

# --
# TAB 1 - Mapa
with tab1:
    total = len(gdf_fil)
    cols_met = st.columns(4)
    for i, nivel in enumerate(ORDEN_RIESGO):
        n = (gdf_fil["riesgo_final"] == nivel).sum()
        cols_met[i].metric(
            label=nivel,
            value=n,
            delta=f"{n/total*100:.0f}% del total" if total > 0 else "—"
        )

    st.markdown("---")
    
    col_izq, col_der = st.columns(2)
    col_izq.markdown("#### Riesgo oficial IMEPLAN")
    col_der.markdown("#### Predicción del modelo topográfico")

    tiles_map = {
        "OpenStreetMap" : "OpenStreetMap",
        "CartoDB Positron" : "CartoDB positron",
        "CartoDB DarkMatter": "CartoDB dark_matter",
        "Stamen Terrain": "stamenterrain"
    }

    # Creación del DualMap
    m = DualMap(
        location=[20.67, -103.35],
        zoom_start=11,
        tiles=tiles_map[basemap_sel]
    )

    for _, row in gdf_fil.iterrows():
        if row.geometry is None:
            continue
            
        # Determinar colores para cada mapa
        nivel_imeplan = row.get("riesgo_final", "Bajo")
        color_imeplan = COLORES_RIESGO.get(str(nivel_imeplan), "#888888")
        
        nivel_modelo = row.get("riesgo_predicho_4c", "Bajo")
        color_modelo = COLORES_RIESGO.get(str(nivel_modelo), "#888888")

        tooltip_html = (
            f"<b>Riesgo IMEPLAN:</b> {row.get('riesgo_final','—')}<br>"
            f"<b>Motivo:</b> {row.get('motivo_norm','—')}<br>"
            f"<b>Infraestructura:</b> {row.get('infra_norm','—')}<br>"
            f"<b>Altura reportada:</b> {row.get('altura_cm','—')} cm<br>"
            f"<b>Predicción modelo:</b> {row.get('riesgo_predicho_4c','—')}<br>"
            f"<b>Prob. riesgo significativo:</b> "
            f"{row.get('prob_significativo', 0):.2f}"
        )

        # Mapa 1 (Izq) IMEPLAN
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x, c=color_imeplan: {
                "fillColor": c,
                "color": c,
                "weight": 1.5,
                "fillOpacity": 0.6,
            },
            tooltip=folium.Tooltip(tooltip_html)
        ).add_to(m.m1)
        
        # Mapa 2 (Der) Predicción del modelo
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x, c=color_modelo: {
                "fillColor": c,
                "color": c,
                "weight": 1.5,
                "fillOpacity": 0.6,
            },
            tooltip=folium.Tooltip(tooltip_html)
        ).add_to(m.m2)

    # leyenda manual 
    leyenda_html = """
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                background:white;padding:10px 14px;border-radius:8px;
                border:1px solid #ccc;font-size:13px;line-height:1.8;
                color:black;">
      <b>Nivel de riesgo</b><br>
      <span style='color:#27AE60;font-size:16px'>■</span> Bajo<br>
      <span style='color:#F39C12;font-size:16px'>■</span> Medio<br>
      <span style='color:#E74C3C;font-size:16px'>■</span> Alto<br>
      <span style='color:#8E44AD;font-size:16px'>■</span> Crítico
    </div>"""
    m.m1.get_root().html.add_child(folium.Element(leyenda_html))

    st_folium(m, use_container_width=True, height=520, returned_objects=[])

# --
# TAB 2 - Análisis
with tab2:
    st.subheader("¿Qué explica el riesgo de inundación en ZMG?")
    st.markdown(
        "El modelo predice el riesgo analizando únicamente el relieve del terreno" 
        "con datos satelitales. Decidimos no incluir los reportes históricos de "
        "inundación durante su entrenamiento para que el sistema aprenda las causas"
        "reales del estancamiento de agua, permitiéndole descubrir nuevos puntos "
        "de riesgo que no están en los catálogos oficiales."
    )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("#### Métricas del modelo")
        st.metric("Accuracy (CV 5-fold)", "67.0%")
        st.metric("ROC-AUC", "0.761")
        st.metric("F1-score", "0.590")
        st.info(
            "El DEM de 30m explica el **67% del riesgo** catalogado por "
            "el IMEPLAN usando únicamente datos satelitales gratuitos. "
            "El 33% restante corresponde a factores urbanos invisibles "
            "al satélite: colectores, rejillas, obstrucciones locales."
        )

    with col_b:
        st.markdown("#### Importancia de variables topohidrológicas")
        nombres_feat = {
            "twi_mean": "TWI medio (acumulación de flujo)",
            "twi_max": "TWI máximo",
            "twi_std": "TWI variabilidad",
            "slope_mean": "Pendiente media (°)",
            "slope_max": "Pendiente máxima (°)",
            "elev_mean": "Elevación media (m)",
            "elev_min": "Elevación mínima (m)",
            "area_m2": "Área del polígono (m²)",
            "infra_cod": "Tipo de infraestructura"
        }
        imp = pd.Series(
            rf_4c.feature_importances_,
            index=[nombres_feat.get(f, f) for f in feature_cols]
        ).sort_values()

        fig = px.bar(
            x=imp.values, y=imp.index,
            orientation="h",
            color=imp.values,
            color_continuous_scale="Blues",
            labels={"x": "Importancia (Gini)", "y": ""},
        )
        fig.update_layout(
            coloraxis_showscale=False,
            height=380,
            margin=dict(l=0, r=10, t=10, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Distribución de riesgo en las zonas catalogadas")
    dist = (
        gdf.groupby("riesgo_final")
        .size()
        .reindex(ORDEN_RIESGO)
        .reset_index(name="zonas")
    )
    fig2 = px.bar(
        dist, x="riesgo_final", y="zonas",
        color="riesgo_final",
        color_discrete_map=COLORES_RIESGO,
        labels={"riesgo_final": "Nivel de riesgo", "zonas": "Nº de zonas"},
        text="zonas"
    )
    fig2.update_layout(showlegend=False, height=300)
    fig2.update_traces(textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

# --
# TAB 3 Intervencion
with tab3:
    st.subheader("Propuestas de infraestructura verde")
    st.markdown(
        "Para cada zona de riesgo Alto o Crítico, el sistema recomienda "
        "la intervención de infraestructura verde con mayor potencial "
        "de captación según el tipo de infraestructura expuesta."
    )

    def recomendar(row):
        infra = str(row.get("infra_norm", "")).lower()
        riesgo = str(row.get("riesgo_final", ""))
        area = float(row.get("area_m2") or 0)

        if "desnivel" in infra:
            return pd.Series({
                "Intervención": "Bioswale + Rejilla de captación",
                "Captación est. (m²)": round(area * 2.5, 1),
                "Co-beneficios": "Reduce velocidad | Filtra contaminantes"
            })
        elif "vivienda" in infra and riesgo in ["Crítico", "Alto"]:
            return pd.Series({
                "Intervención": "Jardín de lluvia en banqueta",
                "Captación est. (m²)": round(area * 1.8, 1),
                "Co-beneficios": "Recarga acuífero | Reduce temperatura"
            })
        elif "vialidad" in infra:
            return pd.Series({
                "Intervención": "Pavimento permeable + Árbol de lluvia",
                "Captación est. (m²)": round(area * 3.0, 1),
                "Co-beneficios": "Reduce escorrentía | Sombra urbana"
            })
        elif riesgo == "Crítico":
            return pd.Series({
                "Intervención": "Parque inundable (Retention basin)",
                "Captación est. (m²)" : round(area * 5.0, 1),
                "Co-beneficios": "Alta retención | Espacio recreativo"
            })
        else:
            return pd.Series({
                "Intervención": "Cuneta verde",
                "Captación est. (m²)": round(area * 1.2, 1),
                "Co-beneficios": "Conduce escorrentía | Mejora agua"
            })

    # Filtrar datos y aplicar recomendaciones conservando la geometría
    gdf_alto = gdf[gdf["riesgo_final"].isin(["Alto", "Crítico"])].copy()
    df_rec   = gdf_alto.apply(recomendar, axis=1)
    
    # Asignar resultados al GeoDataFrame
    gdf_alto["Intervención"] = df_rec["Intervención"]
    gdf_alto["Captación est. (m²)"] = df_rec["Captación est. (m²)"]
    gdf_alto["Co-beneficios"] = df_rec["Co-beneficios"]

    # Preparar el dataframe para la tabla y gráficas
    df_vista = gdf_alto[["motivo_norm", "infra_norm", "riesgo_final", "area_m2", 
                         "Intervención", "Captación est. (m²)", "Co-beneficios"]].copy()
    df_vista = df_vista.rename(columns={
        "motivo_norm": "Motivo",
        "infra_norm": "Infraestructura",
        "riesgo_final": "Riesgo",
        "area_m2": "Área polígono (m²)"
    })

    c1, c2, c3 = st.columns(3)
    c1.metric("Zonas de riesgo Alto + Crítico", len(df_vista))
    c2.metric(
        "Captación potencial total",
        f"{df_vista['Captación est. (m²)'].sum()/10000:.1f} ha"
    )
    c3.metric(
        "Intervención más frecuente",
        df_vista["Intervención"].mode().iloc[0] if len(df_vista) > 0 else "—"
    )

    st.divider()

    # 1. Mapa (para saber dónde se puede intervenir o que se recomiende hacer algo)
    st.markdown("### Ubicación de intervenciones recomendadas")
    
    m_interv = folium.Map(
        location=[20.67, -103.35],
        zoom_start=11,
        tiles=tiles_map[basemap_sel]
    )

    for _, row in gdf_alto.iterrows():
        if row.geometry is None:
            continue
            
        # Centroide del polígono
        centroide = row.geometry.centroid
        interv = row["Intervención"]
        capt = row["Captación est. (m²)"]
        color_int = COLORES_INTERVENCION.get(interv, "#888888")

        tooltip_html = (
            f"<b>Intervención:</b> {interv}<br>"
            f"<b>Captación estimada:</b> {capt} m²"
        )

        folium.CircleMarker(
            location=[centroide.y, centroide.x],
            radius=7,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=color_int,
            fill_opacity=0.9,
            tooltip=folium.Tooltip(tooltip_html)
        ).add_to(m_interv)

    # Leyenda del mapa de intervenciones
    leyenda_interv_html = """
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                background:white;padding:10px 14px;border-radius:8px;
                border:1px solid #ccc;font-size:13px;line-height:1.8;
                color:black;">
      <b>Tipo de Intervención</b><br>
    """
    for key, val in COLORES_INTERVENCION.items():
        leyenda_interv_html += f"<span style='color:{val};font-size:16px'>●</span> {key}<br>"
    leyenda_interv_html += "</div>"
    
    m_interv.get_root().html.add_child(folium.Element(leyenda_interv_html))
    st_folium(m_interv, use_container_width=True, height=520, returned_objects=[])

    st.divider()

    # 2. Tabla (Para saber ué intervenir)
    st.markdown("### Sitios prioritarios por captación estimada")
    st.dataframe(
        df_vista.sort_values("Captación est. (m²)", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=340
    )
    
    st.divider()

    # 3. Gráfica de pastel (Cua´nto se debe de intervenir)
    st.markdown("### Proporción de intervenciones recomendadas")
    fig_pie = px.pie(
        df_vista,
        names="Intervención",
        color="Intervención",
        color_discrete_map=COLORES_INTERVENCION
    )
    fig_pie.update_layout(height=450)
    st.plotly_chart(fig_pie, use_container_width=True)