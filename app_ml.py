# -*- coding: utf-8 -*-
"""App_ML.ipynb"""

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler

# Configura la página de Streamlit
st.set_page_config(page_title="ALGORITMO DE RECOMENDACION", layout="wide")

# Estilos personalizados
st.markdown("""
    <style>
        .main {background-color: #f5f5f5;}
        .stSidebar {background-color: #fff;}
        .stSidebar .st-c1 {
            font-family: Arial, sans-serif;
            font-weight: bold;
            color: #000;
        }
        .stSidebar .st-c4 .stMarkdownContainer p {
            color: #000;
        }
    </style>
""", unsafe_allow_html=True)

# Capturando el Origen de datos de Google Drive
url = r"C:\Users\falabella\Desktop\Algoritmo de Recomendacion\Productos_ML.csv"

# Carga y Lectura de la Base de datos
data = pd.read_csv(url)

# Función para el cálculo de similitud del coseno
@st.cache_data
def calcular_similitud(data):
    tfidf = TfidfVectorizer(max_features=2000)
    tfidf_matrix = tfidf.fit_transform(data['nombre'])
    scaler = MinMaxScaler()
    numerical_features = data[['precio', 'calificacion', 'cantidad_calificaciones']]
    numerical_features_scaled = scaler.fit_transform(numerical_features)
    combined_features = hstack([tfidf_matrix, numerical_features_scaled])
    cosine_sim = cosine_similarity(combined_features)
    return cosine_sim

# Calcula la similitud del coseno (solo una vez)
cosine_sim = calcular_similitud(data)

# Función para obtener las recomendaciones
def get_recommendations(nombre_producto, cosine_sim, df):
    idx = df.index[df['nombre'] == nombre_producto][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Se excluye el producto seleccionado
    product_indices = [i[0] for i in sim_scores]
    recommended_products = df.iloc[product_indices]
    return recommended_products[['nombre', 'imagen', 'calificacion', 'precio']]

# Diseño de la aplicación
def main():
    # Encabezado principal
    st.title("🛍️ Recomendacion de Ofertas Mercado Libre")
    st.image("https://d1.awsstatic.com/r2018/r/seer/customers/600x400_Mercado- Libre_Horizontal_Logo.591869641e3359d8ad7c938d4413445babdacda7.png")
    st.markdown("Descubre las mejores ofertas y productos recomendados para ti.")

    # Barra lateral - Selección de producto
    st.sidebar.header("Filtros de Búsqueda")
    producto_seleccionado = st.sidebar.selectbox("Selecciona un producto:", data['nombre'].unique())

    if producto_seleccionado:
        # Mostrar detalles del producto seleccionado en la barra lateral
        producto_info = data.loc[data['nombre'] == producto_seleccionado, ['nombre', 'imagen', 'calificacion']]
        if not producto_info.empty:
            st.sidebar.image(producto_info['imagen'].values[0], use_column_width=True)
            st.sidebar.subheader(producto_info['nombre'].values[0])
            st.sidebar.write(f"Calificación: {producto_info['calificacion'].values[0]} ⭐")

    # Contenedor principal
    st.header("Recomendaciones")
    recomendaciones = get_recommendations(producto_seleccionado, cosine_sim, data)
    if not recomendaciones.empty:
        for _, row in recomendaciones.iterrows():
            st.image(row['imagen'], width=150)
            st.write(row['nombre'])
            st.write(f"Calificación: {row['calificacion']} ⭐")
            st.write(f"Precio: ${row['precio']}")
    else:
        st.write("No hay recomendaciones disponibles.")

# Ejecutar la aplicación principal
if __name__ == "__main__":
    main()