## Paqueterias necesarias:
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import Funciones_MCF as MCF
from scipy.stats import kurtosis, skew, norm, t

# Algunos parámetros del programa:
stock = '^MXX' #Activo financiero elegido
n_sims = 100000 #Número de Simulaciones para Monte Carlo
ventana = 252 #Tamaño de la ventana de días para Rolling Window

## Proyecto 01 - Encabezado :
st.title("Proyecto 1. Métricas de Riesgo")
st.header(":blue[Métodos Cuantitativos en Finanzas 2025-2]")

# Información del equipo 
st.markdown("""
    <h3 style='color:#111827;'>Integrantes del equipo:</h3>
    <ul>
        <li><b style='color:#2D3748;'>Alix Sue Rangel Mondragón</b> - No. de cuenta: 320219515</li>
        <li><b style='color:#2D3748;'>Edgar Giovanny Caravantes Román</b> - No. de cuenta: 421015887</li>
        <li><b style='color:#2D3748;'>Huitzil Sánchez Martínez</b> - No. de cuenta: 317318681  </li>
        <li><b style='color:#2D3748;'>Martín Sierra González </b> - No. de cuenta: 316027308  </li>
    </ul>
""", unsafe_allow_html=True)

## Breve Descripción del activo elegido
st.write("En esta ocación estaremos trabajando con los datos históricos del IPC Méxcico, cuya clave en Yahoo Finance es: ^MXX. \
         El Índice de Precios y Cotizaciones (IPC) es el índice accionario más representativo del mercado de valores en México \
         y está compuesto por 35 empresas.")

## Descarga de los datos y cálculo de los rendimientos diarios
with st.spinner('Descargando datos...'):
    df_precios = MCF.obtener_datos_2010(stock)
    df_rendimientos = MCF.calcular_rendimientos(df_precios)

## Cálculo de métricas estadísticas: media, sesgo y curtosis.
st.subheader(f'Algunas Metricas Estadísticas: {stock}')
promedio_rendi_diario = df_rendimientos[stock].mean()
curtosis = kurtosis(df_rendimientos[stock])
sesgo = skew(df_rendimientos[stock])

col1,col2,col3 = st.columns(3)

col1.metric("Rendimiento Medio Diario", f"{promedio_rendi_diario:.4%}")
col2.metric("Exceso de Curtosis",f"{curtosis:.4}")
col3.metric("Sesgo",f"{sesgo:.4}")

## Cálculo de Métricas de Riesgo
st.subheader(f'Métricas de Riesgo : {stock}')
metricas = ["VaR", "CVaR / ES"]
metodos = ["Paramétrico (normal)", "Paramétrico (t-Student)", "Histórico", "Monte Carlo (normal)", "Monte Carlo (t-Student)"]
alphas = [0.95, 0.975, 0.99]

# Seleccionadores de opciones:  
metrica_selec = st.selectbox('Selecciona la métrica que deseas consultar', metricas)
metodo_selec = st.selectbox('Selecciona bajo qué método deseas calcular la métrica', metodos)
alpha_selec = st.selectbox('Selecciona un nivel de confianza.', alphas)


if (metodo_selec == metodos[0]):
    VaR = MCF.calcular_VaR_normal(alpha_selec,df_rendimientos[stock])
    if(metrica_selec == metricas[0]):
        col4, col5 = st.columns(2)
        col4.metric("VaR (normal)", f"{VaR:.4%}")
    else:
        ES = MCF.calcular_ES(VaR, df_rendimientos[stock])
        col4, col5 = st.columns(2)
        col4.metric("CVaR o ES (normal)", f"{ES:.4%}")
elif (metodo_selec == metodos[1]):
    VaR = MCF.calcular_VaR_t(alpha_selec,df_rendimientos[stock])
    if(metrica_selec == metricas[0]):
        col4, col5 = st.columns(2)
        col4.metric("VaR (t-Student)", f"{VaR:.4%}")
    else:
        ES = MCF.calcular_ES(VaR, df_rendimientos[stock])
        col4, col5 = st.columns(2)
        col4.metric("CVaR o ES (t-Student)", f"{ES:.4%}")
elif (metodo_selec == metodos[2]):
    VaR = MCF.calcular_VaR_historico(alpha_selec,df_rendimientos[stock])
    if(metrica_selec == metricas[0]):
        col4, col5 = st.columns(2)
        col4.metric("VaR (Histórico)", f"{VaR:.4%}")
    else:
        ES = MCF.calcular_ES(VaR, df_rendimientos[stock])
        col4, col5 = st.columns(2)
        col4.metric("CVaR o ES (Histórico)", f"{ES:.4%}")
elif (metodo_selec == metodos[3]):
    VaR = MCF.calcular_VaR_MC_normal(alpha_selec,df_rendimientos[stock], n_sims)
    if(metrica_selec == metricas[0]):
        col4, col5 = st.columns(2)
        col4.metric("VaR Monte Carlo (normal)", f"{VaR:.4%}")
    else:
        ES = MCF.calcular_ES(VaR, df_rendimientos[stock])
        col4, col5 = st.columns(2)
        col4.metric("CVaR o ES Monte Carlo (normal)", f"{ES:.4%}")
else:
    VaR = MCF.calcular_VaR_MC_t(alpha_selec,df_rendimientos[stock], n_sims)
    if(metrica_selec == metricas[0]):
        col4, col5 = st.columns(2)
        col4.metric("VaR Monte Carlo (t-Student)", f"{VaR:.4%}")
    else:
        ES = MCF.calcular_ES(VaR, df_rendimientos[stock])
        col4, col5 = st.columns(2)
        col4.metric("CVaR o ES Monte Carlo (t-Student)", f"{ES:.4%}")

## Rolling window 
st.subheader(f'Rolling Windows : {stock} ')

tipos =["Normal", "Histórico"] #Tipos de método para calcular VaR y ES en Rolling Window

## Generación de los Rolling Windows:

#Bajo Distribución Normal
VaR_95_normal, ES_95_normal = MCF.generar_RW(ventana,0.95,df_rendimientos[stock], tipo = tipos[0])
VaR_99_normal, ES_99_normal = MCF.generar_RW(ventana,0.99,df_rendimientos[stock], tipo = tipos[0])

#Bajo Aproximación histórica
VaR_95_historico, ES_95_historico = MCF.generar_RW(ventana,0.95,df_rendimientos[stock], tipo = tipos[1])
VaR_99_historico, ES_99_historico = MCF.generar_RW(ventana,0.99,df_rendimientos[stock], tipo = tipos[1])

#Unión en un mismo Data Frame:
df_rend_original = df_rendimientos.copy()
df_rendimientos[stock] = df_rendimientos[stock]*100

df_rendimientos['VaR_95_normal'] = VaR_95_normal*100
df_rendimientos['ES_95_normal'] = ES_95_normal*100
df_rendimientos['VaR_99_normal'] = VaR_99_normal*100
df_rendimientos['ES_99_normal'] = ES_99_normal*100
df_rendimientos['VaR_95_historico'] = VaR_95_historico*100
df_rendimientos['ES_95_historico'] = ES_95_historico*100
df_rendimientos['VaR_99_historico'] = VaR_99_historico*100
df_rendimientos['ES_99_historico'] = ES_99_historico*100

## Gráfica
fig = px.line(df_rendimientos, x = df_rendimientos.index, \
              y = [stock, 'VaR_95_normal', 'ES_95_normal', 'VaR_99_normal', 'ES_99_normal', 'VaR_95_historico', 'ES_95_historico', 'VaR_99_historico', 'ES_99_historico'],\
               title = f'Gráfico de los Rendimientos Diarios : {stock}')
fig.update_xaxes(title = "Fecha")
fig.update_yaxes(title = "Rendimiento(%)")
fig.update_traces(hovertemplate=None)

st.plotly_chart(fig)

## Cálculo y presentación de Violaciones

st.subheader(f'Violaciones : {stock} ')

# Número de violaciones:
n_viol_95_normal =len(df_rendimientos[stock][df_rendimientos[stock]<df_rendimientos['VaR_95_normal']])
n_viol_99_normal = len(df_rendimientos[stock][df_rendimientos[stock]<df_rendimientos['VaR_99_normal']])
n_viol_95_historico = len(df_rendimientos[stock][df_rendimientos[stock]<df_rendimientos['VaR_95_historico']])
n_viol_99_historico = len(df_rendimientos[stock][df_rendimientos[stock]<df_rendimientos['VaR_99_historico']])

# Porcentaje de violaciones:
tamaño = len(df_rendimientos)
p_viol_95_normal = n_viol_95_normal / tamaño
p_viol_99_normal = n_viol_99_normal / tamaño
p_viol_95_historico = n_viol_95_historico / tamaño
p_viol_99_historico = n_viol_99_historico / tamaño

col6, col7= st.columns(2)
col6.metric("Número de Violaciones (0.95 normal)", f'{n_viol_95_normal}')
col6.metric("Número de Violaciones (0.99 normal)", f'{n_viol_99_normal}')
col6.metric("Número de Violaciones (0.95 histórico)", f'{n_viol_95_historico}')
col6.metric("Número de Violaciones (0.99 historico)", f'{n_viol_99_historico}')
col7.metric("Porcentaje de Violaciones (0.95 normal)", f'{p_viol_95_normal:.4%}')
col7.metric("Porcentaje de Violaciones (0.99 normal)", f'{p_viol_99_normal:.4%}')
col7.metric("Porcentaje de Violaciones (0.95 histórico)", f'{p_viol_95_historico:.4%}')
col7.metric("Porcentaje de Violaciones (0.99 historico)", f'{p_viol_99_historico:.4%}')

## Roling Window del VaR con volatitilidad móvil bajo una distribución normal
st.subheader(f'VaR con volatilidad móvil bajo una distribución normal : {stock} ')
# Generación de los rolling windows
VaR_movil_95 = MCF.generar_RW_VaR_movil(ventana, 0.95, df_rend_original[stock])
VaR_movil_99 = MCF.generar_RW_VaR_movil(ventana, 0.99, df_rend_original[stock])

# Unión en el mismo Data Frame:
df_rendimientos['VaR_movil_95'] = VaR_movil_95*100
df_rendimientos['VaR_movil_99'] = VaR_movil_99*100

## Gráfica
fig1 = px.line(df_rendimientos, x = df_rendimientos.index, y = [stock, 'VaR_movil_95', 'VaR_movil_99'],\
               title = f'Gráfico de los Rendimientos Diarios y VaR con volatilidad móvil : {stock}')
fig1.update_xaxes(title = "Fecha")
fig1.update_yaxes(title = "Rendimiento(%)")
fig1.update_traces(hovertemplate=None)

st.plotly_chart(fig1)

st.subheader(f'Violaciones del VaR móvil: {stock} ')
# Número de violaciones:
n_viol_95 =len(df_rendimientos[stock][df_rendimientos[stock]<df_rendimientos['VaR_movil_95']])
n_viol_99 = len(df_rendimientos[stock][df_rendimientos[stock]<df_rendimientos['VaR_movil_99']])

# Porcentaje de violaciones:
tamaño = len(df_rendimientos)
p_viol_95 = n_viol_95 / tamaño
p_viol_99= n_viol_99 / tamaño

col8, col9= st.columns(2)
col8.metric("Número de Violaciones VaR móvil (0.95 normal)", f'{n_viol_95}')
col8.metric("Número de Violaciones VaR móvil (0.99 normal)", f'{n_viol_99}')
col9.metric("Porcentaje de Violaciones VaR móvil (0.95 normal)", f'{p_viol_95:.4%}')
col9.metric("Porcentaje de Violaciones VaR móvil (0.99 normal)", f'{p_viol_99:.4%}')


## Fin del programa
