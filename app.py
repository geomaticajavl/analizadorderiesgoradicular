import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# --- CONFIGURACIÓN ESTÉTICA ---
st.set_page_config(page_title="Analizador Multivariable Raíz", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    div[data-testid="stSidebar"] { background-color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 Sistema de Predicción Fitopatológica Multivariable")
st.write("Esta versión analiza Longitud, Hojas, Humedad del suelo y Días desde la siembra para generar mapas predictivos dinámicos.")

# --- 1. GENERADOR DE DATOS Y PLANTILLA ---
def generar_datos_completos():
    np.random.seed(42)
    n = 350
    longitud = np.random.normal(45, 18, n)
    hojas = np.random.normal(18, 7, n)
    humedad = np.random.normal(65, 12, n)
    dias = np.random.normal(60, 15, n)
    
    logit = -0.04 * longitud + 0.12 * hojas + 0.03 * humedad - 0.01 * dias - 2.5
    prob = 1 / (1 + np.exp(-logit))
    dano = [np.random.choice([True, False], p=[p, 1-p]) for p in prob]
    
    return pd.DataFrame({
        'id': range(1, n + 1),
        'Longitud': np.clip(longitud, 0, 150).astype(int),
        'Hojas': np.clip(hojas, 0, 80).astype(int),
        'Humedad': np.clip(humedad, 10, 100).astype(int),
        'Dias': np.clip(dias, 5, 120).astype(int),
        'Dano': dano # True = Con daño, False = Sana
    })

if 'df_multi' not in st.session_state:
    st.session_state.df_multi = generar_datos_completos()

# Tu plantilla exacta (Intacta)
@st.cache_data
def generar_plantilla_csv():
    df_plantilla = pd.DataFrame({
        'Planta N° ': [1, 2],
        'Longitud (cm) ': [45, 60],
        'Hojas ': [20, 15],
        'Humedad ': [65, 60],
        'Dias desde la Siembra ': [60, 50],
        'Daño en Raices ': ['FALSO', 'VERDADERO']
    })
    return df_plantilla.to_csv(index=False).encode('utf-8')

# --- 2. PANEL LATERAL (CARGA, DESCARGA Y SIMULACIÓN) ---
st.sidebar.header("📁 Gestión de Archivos")

st.sidebar.write("1. Descarga la plantilla y llénala con tus datos:")
st.sidebar.download_button(
    label="📥 Descargar Plantilla CSV",
    data=generar_plantilla_csv(),
    file_name='plantilla_cultivos.csv',
    mime='text/csv',
)

st.sidebar.write("2. Sube tu archivo lleno aquí:")
archivo_cargado = st.sidebar.file_uploader("Arrastra tu Excel o CSV", type=["csv", "xlsx"])

if archivo_cargado is not None:
    try:
        if archivo_cargado.name.endswith('.csv'):
            df_nuevo = pd.read_csv(archivo_cargado)
        else:
            df_nuevo = pd.read_excel(archivo_cargado)
            
        # Limpiamos los nombres de las columnas quitando espacios en blanco extra
        df_nuevo.columns = df_nuevo.columns.str.strip()
        
        # TRADUCTOR: Convierte los nombres de tu Excel a los nombres que usa el código
        df_nuevo = df_nuevo.rename(columns={
            'Planta N°': 'id',
            'Longitud (cm)': 'Longitud',
            'Dias desde la Siembra': 'Dias',
            'Daño en Raices': 'Dano'
        })
            
        # Traductor para que Python entienda el VERDADERO/FALSO de Excel
        if 'Dano' in df_nuevo.columns:
            df_nuevo['Dano'] = df_nuevo['Dano'].replace({'VERDADERO': True, 'FALSO': False, 'Verdadero': True, 'Falso': False, 'Si': True, 'No': False})
            df_nuevo['Dano'] = df_nuevo['Dano'].astype(bool)
            
        st.session_state.df_multi = df_nuevo
        st.sidebar.success("¡Datos cargados con éxito!")
    except Exception as e:
        st.sidebar.error(f"Error al leer el archivo. Revisa que tenga las columnas correctas.")

st.sidebar.divider()
st.sidebar.header("🎛️ Simulación de Entorno")
st.sidebar.write("Ajusta estas variables para ver cómo se comporta el mapa:")

humedad_simulada = st.sidebar.slider("Humedad actual del lote (%)", 10, 100, 65)
dias_simulados = st.sidebar.slider("Días transcurridos desde siembra", 5, 120, 60)

# --- 3. DISEÑO DE LA INTERFAZ PRINCIPAL ---
col_tabla, col_grafica = st.columns([4, 5])

with col_tabla:
    st.subheader("📓 Registro y Edición de Muestras")
    
    # Configuración adaptada a tus nuevas columnas
    config_columnas = {
        "id": st.column_config.NumberColumn("ID", disabled=True), 
        "Dano": st.column_config.CheckboxColumn(
            "Daño Radicular",
            help="Marca la casilla si la planta está enferma.",
            default=False,
        )
    }
    
    datos_vivos = st.data_editor(
        st.session_state.df_multi,
        column_config=config_columnas,
        num_rows="dynamic",
        use_container_width=True
    )

with col_grafica:
    st.subheader("🖼️ Visualización Dinámica del Modelo")
    
    # AQUÍ ESTÁ EL ERROR CORREGIDO ('Dano' en lugar de 'Daño en Raices')
    if len(datos_vivos['Dano'].unique()) < 2:
        st.warning("El modelo requiere al menos una planta sana y una con daño.")
    else:
        # Usamos tus nombres de columnas simplificados
        X = datos_vivos[['Longitud', 'Hojas', 'Humedad', 'Dias']]
        y = datos_vivos['Dano'].astype(int)
        
        modelo_completo = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('logistic', LogisticRegression(max_iter=1000))
        ])
        modelo_completo.fit(X, y)
        
        # --- GENERACIÓN DE LA GRÁFICA ---
        fig, ax = plt.subplots(figsize=(10, 9))
        
        x_space = np.linspace(0, 150, 150)
        y_space = np.linspace(0, 80, 80)
        xx, yy = np.meshgrid(x_space, y_space)
        
        malla_puntos = np.c_[
            xx.ravel(), 
            yy.ravel(), 
            np.full_like(xx.ravel(), humedad_simulada), 
            np.full_like(xx.ravel(), dias_simulados)
        ]
        
        Z = modelo_completo.predict_proba(malla_puntos)[:, 1].reshape(xx.shape)
        
        mapa_calor = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlGn_r', alpha=0.6)
        
        cbar = fig.colorbar(mapa_calor, ax=ax)
        cbar.set_label('Probabilidad de daño radicular', fontsize=11, fontweight='bold')
        
        linea_limite = ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5)
        etiquetas = ax.clabel(linea_limite, inline=True, fontsize=12, fmt="0.50")
        for texto in etiquetas:
            texto.set_weight('bold')
            
        # Puntos usando tus nuevas columnas
        sanas = datos_vivos[datos_vivos['Dano'] == False]
        enfermas = datos_vivos[datos_vivos['Dano'] == True]
        
        ax.scatter(sanas['Longitud'], sanas['Hojas'], 
                   c='#4CAF50', marker='o', s=45, edgecolors='white', linewidth=0.5, label='Sin daño')
        
        ax.scatter(enfermas['Longitud'], enfermas['Hojas'], 
                   c='#F44336', marker='x', s=45, linewidth=1.5, label='Con daño radicular')
        
        ax.set_title("Riesgo de anomalías en raíz pivotante y secundaria\nen función del desarrollo foliar y altura aérea", 
                     fontsize=13, fontweight='bold', pad=20)
        ax.set_xlabel("Longitud aérea (cm)", fontsize=11)
        ax.set_ylabel("Número de hojas", fontsize=11)
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 80)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        
        st.pyplot(fig)

# --- 4. CALCULADORA PREDICTIVA EXPRESO ---
st.divider()
st.subheader("🔮 Diagnóstico Rápido Individual")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    v_lon = st.number_input("Longitud Planta (cm)", 0, 150, 45)
with c2:
    v_hoj = st.number_input("Número de Hojas", 0, 80, 18)
with c3:
    v_hum = st.number_input("Humedad Suelo (%)", 10, 100, 60)
with c4:
    v_dia = st.number_input("Días de Cultivo", 1, 150, 45)
with c5:
    if 'modelo_completo' in locals():
        prediccion_p = modelo_completo.predict_proba([[v_lon, v_hoj, v_hum, v_dia]])[0][1]
        st.metric("Riesgo Estimado", f"{prediccion_p*100:.1f}%")
    else:
        st.metric("Riesgo Estimado", "N/A")
