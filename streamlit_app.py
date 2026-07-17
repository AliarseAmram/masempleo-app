import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import geopandas as gpd
import folium
from branca.element import Template, MacroElement
import io
import os
import Modulo_3 as m3
import streamlit as st
from PIL import Image
import base64
from scipy import stats




# ------------------------
# 🎨 Estilo personalizado
# ------------------------
def aplicar_estilo_personalizado():
    st.markdown(
        """
        <style>
        /* Fondo general */
        .stApp {
            background-color: #FFFFFF;
            font-family: "Segoe UI", sans-serif;
            padding: 1rem;
        }

        /* Contenedor principal debajo del header */
        .block-container {
            padding: 2rem;
            margin-top: 1.5rem;
        }

        /* Títulos */
        h1 {
            font-size: 2.2rem !important;
            color: #412A62;
            font-weight: bold;
            margin-bottom: 1.2rem;
        }

        h2 {
            color: #655681;
            font-size: 1.4rem;
            margin-top: 2rem;
        }

        h3 {
            color: #655681;
            font-size: 1.2rem;
        }


        /* Botones */
        div.stButton > button {
            background-color: #F0836A !important;  /* Morado institucional */
            color: black !important;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        
        /* Hover en botón */
        div.stButton > button:hover {
            background-color: #655681 !important;  /* Lavanda */
            color: white !important;
        }

        /* Efecto hover */
        button[kind="primary"]:hover {
            background-color: #655681 !important;  /* Lavanda */
            color: white !important;
        }


        /* Selectbox, inputs, uploader, sliders */
        .stSelectbox, .stTextInput, .stFileUploader, .stSlider {
            font-size: 1rem !important;
            padding: 0.5rem;
            border-radius: 6px;
        }

        /* Sidebar */
        .css-1d391kg {
            background-color: #F8BCAE;
        }

        /* Métricas */
        div[data-testid="metric-container"] {
            background-color: #f2edf3;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }

        /* Gráficos Plotly: ajustar margen superior */
        .plot-container > div {
            margin-top: -20px;
        }

        /* Texto general */
        body {
            color: #333333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


st.set_page_config(page_title="Análisis Colocación Laboral", layout="wide", initial_sidebar_state="expanded")
aplicar_estilo_personalizado()


# Encabezado con fondo morado, logo blanco y título centrado
with open("aliarse blanco.png", "rb") as img_file:
    logo_encoded = base64.b64encode(img_file.read()).decode()

st.markdown(
    f"""
    <div style='background-color: #655681; padding: 1.2rem 2rem; border-radius: 8px; display: flex; align-items: center;'>
        <img src='data:image/png;base64,{logo_encoded}' width='220' style='margin-right: 2rem;'/>
        <div style='color: white;'>
            <h1 style='font-size: 1.3rem; margin: 0;'>Análisis de Colocación Laboral</h1>
            <h2 style='font-size: 1.05rem; font-weight: normal; margin: 0;'>Programa +Empleo</h2>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


BASE_DIR = os.path.dirname("Actualizacion algoritmo")
MODELO_PATH = os.path.join(BASE_DIR, "modelo_gradientboosting_colocacion.pkl")
COLUMNAS_PATH = os.path.join(BASE_DIR, "columns_modelo.pkl")
SHAPEFILE_PATH = "CRMap/gadm41_CRI_2.shp"

modelo_colocacion = None
columnas_modelo = []
try:
    with open(MODELO_PATH, "rb") as f:
        modelo_colocacion = pickle.load(f)
    with open(COLUMNAS_PATH, "rb") as f:
        columnas_modelo = pickle.load(f)
except Exception as e:
    st.warning(f"Error cargando el modelo: {e}")

st.sidebar.title("Navegación")
menu_options = [    
     "1️⃣ Subir dataset y preparación inicial",
     "2️⃣ Predicción de colocación",
     "3️⃣ Clusterización y mapa",
     "4️⃣ Análisis exploratorio univariado",
     "5️⃣ Análisis exploratorio bivariado"
]

        
selected_option = st.sidebar.selectbox("Selecciona una opción:", menu_options)

if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'df_with_predictions' not in st.session_state:
    st.session_state.df_with_predictions = None
if 'selected_vars_for_clustering' not in st.session_state:
    st.session_state.selected_vars_for_clustering = []

def limpiar_columnas_categoricas(df: pd.DataFrame, normalizar_valores: dict = None, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        if normalizar_valores and col in normalizar_valores:
            df[col] = df[col].replace(normalizar_valores[col])
        df[col] = df[col].str.title()
        if verbose:
            print(f"\n📊 Frecuencia de valores en: {col}")
            print(df[col].value_counts())
    return df

def seccion_carga():
    st.subheader("🗕️ Cargar archivo")
    archivo = st.file_uploader("Sube archivo CSV con variables de entrada", type=["csv"])
    sep = st.selectbox("Separador de columnas", options=[",", ";"], index=0)
    decimal = st.selectbox("Separador decimal", options=[".", ","], index=0)

    if archivo:
        df = pd.read_csv(archivo, sep=sep, decimal=decimal)
        st.dataframe(df.head())
        st.session_state.raw_df = df.copy()  

        if st.button("🩹 Preparar datos"):
            df = limpiar_columnas_categoricas(df, verbose=False)
            df.replace(to_replace=["N/D", "n/d", "Nd", "Na", "Nan", "nan", "None", "Sin Dato", "Desconocido"],
                         value=np.nan, inplace=True)
            df.dropna(inplace=True)

            if "Nivel educativo" in df.columns:
                df["Nivel educativo"] = df["Nivel educativo"].astype("category").cat.codes
            if "Inglés inicial" in df.columns:
                df["Inglés inicial"] = df["Inglés inicial"].astype("category").cat.codes
            #if "Año" in df.columns:
             #   df["Año"] = df["Año"].astype("category").cat.codes

            variables_nominales = ["Sexo", "Tipo de beca", "Región", "Tipo de población", "Provincia", "Cantón"]
            for col in variables_nominales:
                if col in df.columns:
                    df[col] = df[col].astype("category")

            st.session_state.df_clean = df
            st.success("Datos correctamente limpios")
            st.dataframe(df)

# ----------------------------------
# 2. Predicción
# ----------------------------------
def seccion_prediccion():
    st.subheader("Predicción de colocación")

    if st.session_state.df_clean is not None and modelo_colocacion is not None:
        if st.button("🔮 Aplicar predicción"):

            df = st.session_state.df_clean.copy()

            # 🗂️ Copiar columnas auxiliares antes de eliminarlas
            columnas_auxiliares = {}
            for col in ["Nombre de participante", "Año"]:
                if col in df.columns:
                    columnas_auxiliares[col] = df[col].copy()

            # ✅ Guardar versión legible de variables antes de codificación
            columnas_visuales = ["Nivel educativo", "Inglés inicial"]
            df_visual = df[columnas_visuales].copy()

            # 🔻 Eliminar columnas que no van al modelo
            for col in ["Nombre de participante", "Año", "Estado"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)

            # 🔢 Codificación y predicción
            dummies = pd.get_dummies(df, drop_first=False)
            dummies = dummies.reindex(columns=columnas_modelo, fill_value=0)
            preds = modelo_colocacion.predict(dummies)

            # 🧩 Reconstruir df_result con predicción + columnas originales
            df_result = st.session_state.df_clean.copy()
            df_result = df_result.loc[dummies.index]
            df_result["Colocación"] = ["Sí" if p == 1 else "No" for p in preds]

            # Reintegrar columnas auxiliares copiadas (Nombre, Año)
            for col, serie in columnas_auxiliares.items():
                df_result[col] = serie.values

            # Reintegrar columnas visuales (nivel educativo, inglés)
            df_result[columnas_visuales] = df_visual

            # Guardar resultado en sesión
            st.session_state.df_with_predictions = df_result

            # Mostrar resultados
            st.success("✅ Predicciones generadas")
            st.dataframe(df_result, use_container_width=True)

            # 📊 Mostrar resumen
            st.metric("Total de registros con predicción", len(df_result))
            df_info = pd.DataFrame({
                "Descripción": ["Registros en df_with_predictions"],
                "Cantidad": [len(df_result)]
            })
            #st.table(df_info)

            # 📊 Gráfico de pastel

            st.subheader("Distribución de colocación laboral (predicción)")

            # Preparar datos
            conteo_colocacion = df_result["Colocación"].value_counts().reset_index()
            conteo_colocacion.columns = ["Colocación", "Cantidad"]
            
            # Crear gráfico
            fig = px.pie(
                conteo_colocacion,
                names="Colocación",
                values="Cantidad",
                color="Colocación",
                color_discrete_map={
                    "Sí": "#F0836A",   # naranja institucional
                    "No": "#655681"    # morado lavanda
                },
                hole=0.4  # Donut
            )
            
            fig.update_traces(
                textinfo='percent+label',
                textfont_size=14
            )
            
            fig.update_layout(
                title="Porcentaje de Colocación Laboral",
                title_font_size=18,
                showlegend=True,
                legend=dict(orientation="h", y=-0.2),
                margin=dict(t=50, b=20, l=0, r=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Debes cargar y limpiar datos primero")

def seccion_cluster():
    import Modulo_3 as m3

    st.subheader("Clusterización y visualización geográfica")

    if 'raw_df' not in st.session_state or st.session_state.raw_df is None:
        st.warning("Primero debes cargar los datos en la sección 1.")
        return

    df = st.session_state.raw_df.copy()
    df.replace(
        to_replace=["N/D", "n/d", "Nd", "Na", "Nan", "nan", "None", "Sin Dato", "Desconocido"],
        value=np.nan,
        inplace=True
    )
    df = limpiar_columnas_categoricas(df, verbose=False)

    ubicacion = df[["Cantón", "Provincia"]].copy() if "Cantón" in df.columns and "Provincia" in df.columns else None

    mapa_nivel_educativo = {
        "Primaria Completa": 0, "Secundaria Incompleta": 1, "Secundaria Completa": 2,
        "Cursando Una Carrera Tecnica": 3, "Carrera Tecnica Completa": 4,
        "Cursando La Universidad": 5, "Universidad Incompleta": 6, "Universidad Completa": 7
    }
    mapa_ingles = {
        "No Proficiency": 1, "A1": 2, "A2": 3, "B1": 4, "B1+": 5, "B2": 6, "B2+": 7, "C1": 11
    }
    
    if "Nivel educativo" in df.columns:
        df["Nivel educativo"] = df["Nivel educativo"].map(mapa_nivel_educativo).fillna(-1).astype(int)

    if "Inglés inicial" in df.columns:
        df["Inglés inicial"] = df["Inglés inicial"].map(mapa_ingles).fillna(-1).astype(int)

    df.dropna(inplace=True)

    # Lista de variables que NO deben aparecer
    excluir = ["Año", "Nombre de participante"]

    # Obtener columnas y quitar las excluidas
    variables_disponibles = [
        col for col in df.select_dtypes(include=["int64", "float64", "object", "category"]).columns
        if col not in excluir
    ]

    # Mostrar selector sin las columnas excluidas
    seleccionadas = st.multiselect(
        "Selecciona variables para aplicar clustering:", 
        options=variables_disponibles, 
        default=variables_disponibles
    )

    if not seleccionadas:
        st.warning("Debes seleccionar al menos una variable para continuar.")
        return

    st.session_state.selected_vars_for_clustering = seleccionadas

    df_modelo = df[seleccionadas].copy()
    df_dummies = pd.get_dummies(df_modelo, drop_first=False)
    scaler = StandardScaler()
    X_cluster = pd.DataFrame(scaler.fit_transform(df_dummies), columns=df_dummies.columns)

    k = st.slider("Selecciona número de clusters (k):", 2, 10, 3)

    if st.button("🏷️ Ejecutar clusterización"):
        kmeans = KMeans(n_clusters=k, n_init=100, max_iter=500, random_state=42)
        clusters = kmeans.fit_predict(X_cluster)

        df_clusterizado = df.copy()
        df_clusterizado["Cluster"] = clusters
        
        if st.session_state.df_with_predictions is not None and "Colocación" in st.session_state.df_with_predictions.columns:
            df_colocacion = st.session_state.df_with_predictions[["Colocación"]]
            df_clusterizado = df_clusterizado.merge(df_colocacion, left_index=True, right_index=True, how="left")
                   
# --------------------------
# 📊 Tabla y barra: colocación por clúster
# --------------------------
        st.subheader("📋 Tabla de colocación por clúster")
        
        # Agrupar y calcular métricas
        tabla_colocacion_cluster = (
            df_clusterizado.groupby("Cluster")["Colocación"]
            .agg(
                Total_personas='count',
                Personas_colocadas=lambda x: (x == 'Sí').sum()
            )
            .reset_index()
        )
        tabla_colocacion_cluster["Porcentaje colocación (%)"] = (
            tabla_colocacion_cluster["Personas_colocadas"] / tabla_colocacion_cluster["Total_personas"] * 100
        ).round(2)
        
        # Mostrar tabla
        tabla_colocacion_cluster = tabla_colocacion_cluster.sort_values(by="Cluster")
        st.dataframe(tabla_colocacion_cluster, use_container_width=True)
        
        # Gráfico de barras
        st.subheader("📊 Porcentaje de colocación por clúster")
        porcentaje_colocacion_cluster = tabla_colocacion_cluster[["Cluster", "Porcentaje colocación (%)"]].copy()
        porcentaje_colocacion_cluster["Cluster Label"] = porcentaje_colocacion_cluster["Cluster"].apply(lambda x: f"Clúster {x}")
            
        fig_bar = px.bar(
            porcentaje_colocacion_cluster,
            x="Cluster Label",
            y="Porcentaje colocación (%)",
            text="Porcentaje colocación (%)",
            color="Cluster Label",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig_bar.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            marker_line_color='white',
            marker_line_width=1.5
        )
        
        fig_bar.update_layout(
            title="Porcentaje de colocación por clúster",
            yaxis=dict(title="Porcentaje de colocación (%)", range=[0, 100]),
            xaxis_title=None,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI", size=14, color="#333333")
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
        st.subheader("🌐 Visualización PCA")
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(X_cluster)
        centros = pca.transform(kmeans.cluster_centers_)

        df_pca = pd.DataFrame(componentes, columns=["PC1", "PC2"])
        df_pca["Cluster"] = clusters.astype(str)
        # Convertir centros a DataFrame
        df_centros = pd.DataFrame(centros, columns=["PC1", "PC2"])
        df_centros["Cluster"] = [f"Centro {i}" for i in range(len(centros))]
        
        # Gráfico PCA interactivo
        fig_pca = px.scatter(
            df_pca,
            x="PC1",
            y="PC2",
            color="Cluster",
            title=f"Visualización PCA con k={k}",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"PC1": "Componente Principal 1", "PC2": "Componente Principal 2"},
            opacity=0.8,
        )
        
        # Agregar centros como símbolos X en negro
        fig_pca.add_trace(
            go.Scatter(
                x=df_centros["PC1"],
                y=df_centros["PC2"],
                mode="markers+text",
                name="Centros",
                marker=dict(color="black", size=12, symbol="x"),
                text=[f"C{i}" for i in range(len(centros))],
                textposition="top center"
            )
        )
        
        fig_pca.update_layout(
            font=dict(family="Segoe UI", size=13),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend_title="Clúster"
        )
        
        st.plotly_chart(fig_pca, use_container_width=True)
          
        def radar_plot_plotly(centros_filtrados, labels, k):
        
            # Normalizar valores al rango 0-100
            min_val = centros_filtrados.min()
            max_val = centros_filtrados.max()
            centros_escalados = 100 * (centros_filtrados - min_val) / (max_val - min_val)
        
            # Crear DataFrame
            df_centros = pd.DataFrame(centros_escalados, columns=labels)
            df_centros["Cluster"] = [f"Clúster {i}" for i in range(len(centros_escalados))]
        
            # Transponer para formato de radar
            df_melt = df_centros.melt(id_vars="Cluster", var_name="Variable", value_name="Valor")
        
            # Crear gráfico radar
            fig = go.Figure()
        
            for cluster in df_centros["Cluster"]:
                valores = df_melt[df_melt["Cluster"] == cluster]["Valor"].tolist()
                fig.add_trace(go.Scatterpolar(
                    r=valores + [valores[0]],  # cerrar el loop
                    theta=labels.tolist() + [labels[0]],  # cerrar el loop
                    name=cluster,
                    line=dict(width=2),
                    opacity=0.8
                ))
        
            # Estética: mostrar de 0 a 100 en pasos de 20
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickvals=[0, 20, 40, 60, 80, 100],
                        tickfont=dict(size=10),
                        gridcolor="#d8d8d8"
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=11)
                    )
                ),
                title=f"Perfil de variables normalizadas por clúster (k={k})",
                showlegend=True,
                height=650,
                font=dict(family="Segoe UI", size=13),
                margin=dict(t=100, b=40)
            )
        
            return fig

        st.subheader("📊 Perfil de clústeres (Radar)")
        columnas_validas = [c for c in X_cluster.columns if not c.startswith("Cantón_") and c != "Año"]
        centros_df = pd.DataFrame(kmeans.cluster_centers_, columns=X_cluster.columns)
        centros_filtrados = centros_df[columnas_validas].values
      
        fig_radar = radar_plot_plotly(centros_filtrados, np.array(columnas_validas), k)
        st.plotly_chart(fig_radar, use_container_width=True)
               
    
        st.subheader("📊 Comparación de variables por clúster (Barra)")
        fig = plt.figure(figsize=(15, 8), dpi=100)
        m3.bar_plot(centros_filtrados, labels=np.array(columnas_validas))
        plt.suptitle(f"Comparación de variables por clúster (k={k}, sin cantón ni año)", fontsize=15, weight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        

        st.subheader("📍 Mapa por cantón y clúster dominante")
        try:
            gdf = gpd.read_file(SHAPEFILE_PATH)
            gdf = gdf.rename(columns={"NAME_2": "Cantón"})
            gdf["Cantón"] = gdf["Cantón"].astype(str).str.upper().str.strip()
            df_clusterizado["Cantón"] = df_clusterizado["Cantón"].astype(str).str.upper().str.strip()

            resumen = df_clusterizado.groupby("Cantón").agg(
                Cluster_Dominante=('Cluster', lambda x: x.value_counts().idxmax()),
                Porc_Colocación_Cantón=('Colocación', lambda x: (x == "Sí").mean() * 100 if x.dtype == object else np.nan),
                Total_Individuos=('Cluster', 'count')
            ).reset_index()

            gdf_map = gdf.merge(resumen, on='Cantón', how='left')
            centroides = gdf_map.copy()
            centroides['centroide'] = centroides.geometry.centroid
            centroides['lat'] = centroides.centroide.y
            centroides['lon'] = centroides.centroide.x

            df_clusterizado = df_clusterizado.merge(centroides[['Cantón', 'lat', 'lon']], on='Cantón', how='left')
            np.random.seed(42)
            df_clusterizado['lat_jit'] = df_clusterizado['lat'] + np.random.normal(0, 0.02, len(df_clusterizado))
            df_clusterizado['lon_jit'] = df_clusterizado['lon'] + np.random.normal(0, 0.02, len(df_clusterizado))
            st.session_state.df_clusterizado = df_clusterizado.copy()

            m = folium.Map(location=[9.75, -84], zoom_start=7, tiles='CartoDB positron')

            folium.GeoJson(
                gdf_map,
                style_function=lambda feature: {
                    'fillColor': '#e6e6e6' if pd.isna(feature['properties']['Cluster_Dominante']) else '#ffffff',
                    'color': '#444444', 'weight': 0.8, 'fillOpacity': 0.5
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['Cantón', 'Cluster_Dominante', 'Porc_Colocación_Cantón', 'Total_Individuos'],
                    aliases=['Cantón', 'Cluster Dominante', '% Colocación', 'Total'], localize=True
                )
            ).add_to(m)

            for _, row in df_clusterizado.dropna(subset=['lat_jit', 'lon_jit']).iterrows():
                color = '#F0836A' if row.get('Colocación') == 'Sí' else '#655681'
                folium.CircleMarker(
                    location=[row['lat_jit'], row['lon_jit']],
                    radius=5, color=color, fill=True, fill_opacity=0.8,
                    tooltip=f"{row['Cantón']} | {row.get('Colocación', 'N/D')} | Clúster: {row['Cluster']}"
                ).add_to(m)

            st.components.v1.html(m._repr_html_(), height=700)
        except Exception as e:
            st.error(f"Error al generar el mapa: {e}")
                       
            
#-----------------------------------------------------------
#Sección de cálculos de análisis exploratorio univariado 
#-----------------------------------------------------------
            
def seccion_analisis_exploratorio():
    if 'raw_df' not in st.session_state or st.session_state.raw_df is None:
        st.warning("⚠️ Primero debes subir y preparar el dataset en la sección 1.")
        return

    # Copia del dataset original, sin transformaciones
    df = st.session_state.raw_df.copy()

    # Limpieza de valores nulos y ruidosos
    df.replace(
        to_replace=["N/D", "n/d", "Nd", "Na", "Nan", "nan", "None", "Sin Dato", "Desconocido"],
        value=np.nan, inplace=True
    )

    # Limpieza general de strings y espacios
    df = limpiar_columnas_categoricas(df, verbose=False)

    # Orden definido para variables ordinales
    orden_educativo = [
        "Primaria Completa",
        "Secundaria Incompleta",
        "Secundaria Completa",
        "Cursando Una Carrera Tecnica",
        "Carrera Tecnica Completa",
        "Cursando La Universidad",
        "Universidad Incompleta",
        "Universidad Completa"
    ]
    orden_ingles = ["A1", "A2", "B1", "B1+", "B2", "B2+", "C1", "No Proficiency"]

    if "Nivel educativo" in df.columns:
        df["Nivel educativo"] = df["Nivel educativo"].astype(str).str.strip()
        df["Nivel educativo"] = df["Nivel educativo"].str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
        df["Nivel educativo"] = pd.Categorical(df["Nivel educativo"], categories=orden_educativo, ordered=True)

    if "Inglés inicial" in df.columns:
        df["Inglés inicial"] = df["Inglés inicial"].astype(str).str.strip()
        df["Inglés inicial"] = df["Inglés inicial"].str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
        df["Inglés inicial"] = pd.Categorical(df["Inglés inicial"], categories=orden_ingles, ordered=True)

    # Tabla descriptiva
    st.subheader("Análisis Univariado")
    variables_categoricas = ['Sexo', 'Tipo de beca', 'Región', 'Nivel educativo', 'Inglés inicial', 'Provincia', 'Cantón']
    total = len(df)

    for col in variables_categoricas:
        if col in df.columns:
            conteo = df[col].value_counts(sort=False).reset_index()
            conteo.columns = [col, "Cantidad"]
            conteo["Porcentaje"] = (conteo["Cantidad"] / total * 100).round(1).astype(str) + "%"
            st.write(f"**📋 Distribución de {col}:**")
            st.dataframe(conteo)

    # Ahora toda la sección de gráficos usa df (no df_with_predictions)
    datos = df.copy()


    # Histograma con curva KDE usando Plotly
    st.subheader("🟣 Distribución de la Edad")
    
    # Crear histograma
    fig_hist = px.histogram(
        datos,
        x="Edad",
        nbins=20,
        title="Distribución de la Edad",
        labels={"Edad": "Edad"},
        opacity=0.7,
        color_discrete_sequence=["#655681"]
    )
    
    # Agregar curva de densidad
    fig_hist.add_trace(
        go.Scatter(
            x=np.linspace(datos["Edad"].min(), datos["Edad"].max(), 200),
            y=len(datos) * np.diff(np.histogram(datos["Edad"], bins=20)[1])[0] *
              stats.gaussian_kde(datos["Edad"])(np.linspace(datos["Edad"].min(), datos["Edad"].max(), 200)),
            mode='lines',
            line=dict(color='#412A62', width=2),
            name='Curva KDE'
        )
    )
    
    # Ajustar layout
    fig_hist.update_layout(
        font=dict(family="Segoe UI", size=13),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Edad",
        yaxis_title="Cantidad",
        bargap=0.1
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)


    color_degradado = ["#C4A9D9", "#F0836A", "#A684C7", "#F5B09E", "#655681", "#F6A489"]

    def grafico_horizontal(titulo, variable, orden=None):
        st.subheader(f"🟣 {titulo}")
        conteo = datos[variable].value_counts().reset_index()
        conteo.columns = [variable, "Cantidad"]
        conteo["Porcentaje"] = (conteo["Cantidad"] / conteo["Cantidad"].sum()) * 100
    
        if orden:
            conteo[variable] = pd.Categorical(conteo[variable], categories=orden, ordered=True)
            conteo = conteo.sort_values(variable)
    
        fig = px.bar(
            conteo,
            x="Cantidad",
            y=variable,
            orientation='h',
            text="Cantidad",
            title=titulo,
            color=variable,  # 🎨 Diferentes colores por categoría
            color_discrete_sequence=color_degradado,
            hover_data={"Porcentaje": ':.1f'}
        )
    
        fig.update_traces(textposition='outside')
        fig.update_layout(
            font=dict(family="Segoe UI", size=13),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Cantidad",
            yaxis_title=None,
            legend_title=None
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
    
    # Función auxiliar para gráficos de barras verticales con Plotly
    def grafico_vertical(titulo, variable):
        st.subheader(f"🟣 {titulo}")
        conteo = datos[variable].value_counts().reset_index()
        conteo.columns = [variable, "Cantidad"]
        conteo["Porcentaje"] = (conteo["Cantidad"] / conteo["Cantidad"].sum()) * 100
    
        fig = px.bar(
            conteo,
            x=variable,
            y="Cantidad",
            text="Cantidad",
            title=titulo,
            color=variable,  # 🎨 Diferentes colores por categoría
            color_discrete_sequence=color_degradado,
            hover_data={"Porcentaje": ':.1f'}
        )
    
        fig.update_traces(textposition='auto')
        fig.update_layout(
            font=dict(family="Segoe UI", size=13),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title=None,
            yaxis_title="Cantidad",
            legend_title=None,
            margin=dict(t=80)
        )
    
        st.plotly_chart(fig, use_container_width=True)

    
    # 📊 Gráficos categóricos
    grafico_vertical("Distribución por Región", "Región")
    grafico_vertical("Distribución por Tipo de población", "Tipo de población")
    grafico_vertical("Distribución por Sexo", "Sexo")
    grafico_horizontal("Distribución por Nivel Educativo", "Nivel educativo", orden=orden_educativo)
    grafico_horizontal("Distribución por Tipo de beca", "Tipo de beca")
    grafico_horizontal("Distribución por Inglés inicial", "Inglés inicial", orden=orden_ingles)
    

    if "Edad" in datos.columns:
        st.subheader("🟣 Distribución de Edad")
        
        fig = px.box(
            datos,
            x="Edad",
            points="outliers",  # Muestra los puntos atípicos
            color_discrete_sequence=["#655681"],  # Color institucional
            title="Boxplot de Edad"
        )
        
        fig.update_layout(
            font=dict(family="Segoe UI", size=13),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Edad",
            yaxis_title=None,
        )
        
        st.plotly_chart(fig, use_container_width=True)

#---------------------------------------------------------------
# Sección análisis de relación de variables con la colocación
#---------------------------------------------------------------

colores_personalizados = ["#655681", "#F0836A", "#F79B88", "#F8BCAE", "#A197B2"]

# Diccionarios inversos con claves como enteros (no strings)
mapa_nivel_educativo_inv = {
    0: "Primaria Completa",
    1: "Secundaria Incompleta",
    2: "Secundaria Completa",
    3: "Cursando Una Carrera Tecnica",
    4: "Carrera Tecnica Completa",
    5: "Cursando La Universidad",
    6: "Universidad Incompleta",
    7: "Universidad Completa"
}

mapa_ingles_inv = {
    0: "A1",
    1: "A2",
    2: "B1",
    3: "B1+",
    4: "B2",
    5: "B2+",
    6: "C1",
    7: "No Proficiency"
}

def grafico_comparativo(df, variable):
    import plotly.express as px
    import pandas as pd

    orden_educativo = [
        "Primaria Completa", "Secundaria Incompleta", "Secundaria Completa",
        "Cursando Una Carrera Tecnica", "Carrera Tecnica Completa",
        "Cursando La Universidad", "Universidad Incompleta", "Universidad Completa"
    ]
    orden_ingles = ["A1", "A2", "B1", "B1+", "B2", "B2+", "C1", "No Proficiency"]

    if variable not in df.columns or "Colocación" not in df.columns:
        st.warning("⚠️ La variable seleccionada no está disponible en el dataframe.")
        return

    # Copia y limpieza
    df_filtrado = df[[variable, "Colocación"]].dropna().copy()

    # Asegurar tipo correcto
    if variable == "Nivel educativo":
        df_filtrado[variable] = pd.to_numeric(df_filtrado[variable], errors='coerce')
        df_filtrado[variable] = df_filtrado[variable].map(mapa_nivel_educativo_inv)
        df_filtrado[variable] = pd.Categorical(df_filtrado[variable], categories=orden_educativo, ordered=True)
        category_orders = {variable: orden_educativo}

    elif variable == "Inglés inicial":
        df_filtrado[variable] = pd.to_numeric(df_filtrado[variable], errors='coerce')
        df_filtrado[variable] = df_filtrado[variable].map(mapa_ingles_inv)
        df_filtrado[variable] = pd.Categorical(df_filtrado[variable], categories=orden_ingles, ordered=True)
        category_orders = {variable: orden_ingles}

    else:
        df_filtrado[variable] = df_filtrado[variable].astype(str).str.strip()
        category_orders = None

    # Validación final
    if df_filtrado.empty or df_filtrado[variable].isna().all():
        st.warning(f"⚠️ No se encontraron datos válidos para la variable {variable}.")
        return

    # Agrupar datos
    conteo = df_filtrado.groupby([variable, 'Colocación']).size().reset_index(name='Conteo')
    total = conteo.groupby(variable)['Conteo'].transform('sum')
    conteo['Porcentaje'] = (conteo['Conteo'] / total * 100).round(1)
    conteo['Etiqueta'] = conteo['Conteo'].astype(str) + " (" + conteo['Porcentaje'].astype(str) + "%)"

    # Generar gráfico
    fig = px.bar(
        conteo,
        x=variable,
        y="Porcentaje",
        color="Colocación",
        barmode="group",
        text='Etiqueta',
        category_orders=category_orders,
        title=f"Distribución de colocación por {variable}",
        color_discrete_sequence=colores_personalizados,
        custom_data=['Colocación', 'Conteo']
    )

    fig.update_traces(
        textposition='outside',
        hovertemplate="%{x}<br>Colocación=%{customdata[0]}<br>Cantidad=%{customdata[1]}<br>Porcentaje=%{y}%"
    )

    fig.update_layout(
        xaxis_title=variable,
        yaxis_title="Porcentaje",
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        margin=dict(t=80),  # ⬅️ Este margen evita que el número se corte arriba
        font=dict(family="Segoe UI", size=13),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend_title="Colocación"
    )
        
    st.plotly_chart(fig, use_container_width=True)

#---------------------------------------------------------------
# Función principal para análisis gráfico comparativo
#---------------------------------------------------------------
def seccion_graficos_distribucion():
    import plotly.express as px

    st.markdown('<h2 class="section-header">Análisis bivariado</h2>', unsafe_allow_html=True)

    if "df_with_predictions" not in st.session_state or st.session_state.df_with_predictions is None:
        st.warning("⚠️ Primero debes cargar los datos y generar predicciones.")
        return

    datos = st.session_state.df_with_predictions.copy()

    # Asegurar limpieza mínima

    if datos["Inglés inicial"].dtype == object:
        datos["Inglés inicial"] = datos["Inglés inicial"].str.strip()

    # 🔹 Gráfico general de colocación
    st.subheader("🟣 Distribución general de colocación")
    distribucion_colocacion = datos["Colocación"].value_counts(normalize=True).round(3) * 100
    conteo_colocacion = datos["Colocación"].value_counts()

    fig_general = px.bar(
        x=conteo_colocacion.index,
        y=distribucion_colocacion.values,
        text=conteo_colocacion.values,
        labels={"x": "Colocación", "y": "Porcentaje"},
        title="Distribución general de colocación (Sí / No)",
        color=conteo_colocacion.index,
        color_discrete_sequence=colores_personalizados
    )
    fig_general.update_traces(textposition="outside")
    fig_general.update_layout(
    margin=dict(t=80),  # 🟣 Aumenta espacio en la parte superior
    font=dict(family="Segoe UI", size=13),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis_title="Colocación",
    yaxis_title="Porcentaje",
    legend_title="Colocación"
)
    st.plotly_chart(fig_general, use_container_width=True)

    # 🔹 Análisis comparativo por variable
    st.subheader("🟣 Análisis comparativo por variable")
    posibles_vars = ['Sexo', 'Tipo de beca', 'Región', 'Inglés inicial', 'Provincia', 'Cantón', 'Tipo de población']
    categorical_cols = [col for col in posibles_vars if col in datos.columns]

    if categorical_cols:
        variable_seleccionada = st.selectbox("Selecciona la variable para análisis:", categorical_cols)
        if variable_seleccionada:
            grafico_comparativo(datos, variable_seleccionada)

    df_export = st.session_state.df_clusterizado.copy()
    #st.dataframe(df_export.head())
    
    if 'df_clusterizado' in st.session_state and 'Cluster' in st.session_state.df_clusterizado.columns:
        df_export = st.session_state.df_clusterizado.copy()
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df_export.to_excel(writer, index=False, sheet_name="Resultados")
        buffer.seek(0)
    
        st.subheader("📤 Descargar resultados")
        st.download_button(
            label="📥 Descargar archivo Excel",
            data=buffer,
            file_name="cluster_con_colocacion.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("⚠️ Aún no se han generado resultados de clusterización.")

#---------------------------------------------------------------
# Menú principal de navegación en la app
#---------------------------------------------------------------

if selected_option == menu_options[0]:
    seccion_carga()
elif selected_option == menu_options[1]:
    seccion_prediccion()
elif selected_option == menu_options[2]:
    seccion_cluster()
elif selected_option == menu_options[3]:
    seccion_analisis_exploratorio()
elif selected_option == menu_options[4]:
    seccion_graficos_distribucion()
    
