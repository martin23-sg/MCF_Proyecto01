## Paqueterias necesarias:
import yfinance as yf # Api de Yahoo Finance
import datetime
import numpy as np
from scipy.stats import norm, t

#-------------------------------------------------------------------------------------

def obtener_datos_2010(stock):
    '''
    Funcion para descargar el precio de cierre de un activo financiero desde 2010.

    Input = Ticker del activo en string 

    Output = DataFrame del precio del activo

    '''
    start_date = datetime.date(2010,1,1)  #Inicio del periodo
    today = datetime.date.today().strftime('%Y-%m-%d')  #Fin del periodo
    return yf.download(stock, start = start_date, end = today )['Close']

def calcular_rendimientos(df):
    '''
    Funcion de calcula los rendimientos de un activo financiero

    Input = Data Frame de precios del activo

    Output = Data Frame de los rendimientos del activo

    '''
    return df.pct_change().dropna()

def calcular_VaR_normal(nconf, datos):
    '''
    Función para calcular el VaR bajo una distribución normal de un activo financiero

    Input  = nivel de confianza y serie histórica del activo

    Output = VaR normal, media y desviación estándar

    '''
    mu = np.mean(datos)
    sigma = np.std(datos)
    return (norm.ppf(1-nconf,mu,sigma))

def calcular_VaR_t(nconf, datos):
    '''
    Función para calcular el VaR bajo una distribución t-Student de un activo financiero

    Input  = nivel de confianza y serie histórica del activo

    Output = VaR t-Student
    '''
    dof, loc, scale = t.fit(datos)
    return (t.ppf(1-nconf, dof, loc, scale))

def calcular_VaR_historico(nconf, datos):
    '''
    Función para calcular el VaR bajo una aproximación histórica de un activo financiero

    Input  = nivel de confianza y serie histórica del activo

    Output = VaR histórico
    '''
    return (datos.quantile(1-nconf))

def calcular_VaR_MC_normal(nconf, datos, n_sims):
    '''
    Función para calcular el VaR bajo una aproximación Monte Carlo (normal) de un activo financiero

    Input  = nivel de confianza, serie histórica del activo y número de simulaciones

    Output = VaR Monte Carlo
    '''
    mu = np.mean(datos)
    sigma = np.std(datos)
    sim_rend = np.array(norm.rvs(mu, sigma, size = n_sims))
    return (np.percentile(sim_rend, (1-nconf)*100))

def calcular_VaR_MC_t(nconf, datos, n_sims):
    '''
    Función para calcular el VaR bajo una aproximación Monte Carlo (t-Student) de un activo financiero

    Input  = nivel de confianza, serie histórica del activo y número de simulaciones

    Output = VaR Monte Carlo
    '''
    dof, loc, scale = t.fit(datos)
    sim_rend= np.array(t.rvs(dof, loc, scale, size = n_sims))
    return (np.percentile(sim_rend, (1-nconf)*100))

def calcular_ES(VaR, datos):
    '''
    Función para calcular el CVaR o ES de un activo financiero

    Input  = VaR y serie histórica del activo

    Output = CVaR o ES
    '''
    return datos[datos <= VaR].mean()

def generar_RW(tam_ventana, nconf, datos, tipo):
    '''
    Función para generar una Rolling Window de VaR y ES de un activo financiero
    Input = Tamaño de la ventana, nivel de confianza, serie histórico del activo y tipo de método a utilizar.
    Output = Numpy array con la rolling window de VaR y Numpy array con la rolling window de ES
    '''
    n = len(datos) 
    VaR = np.zeros(n)
    ES = np.zeros(n)
    for i in range(0, tam_ventana):
        VaR[i] = None
        ES[i] = None
    for i in range(0, n-tam_ventana):
        if(tipo == "Normal"):
            VaR[i+tam_ventana] = calcular_VaR_normal(nconf, datos[i: i + tam_ventana])
            ES[i+tam_ventana] = calcular_ES(VaR[i+tam_ventana], datos[i: i + tam_ventana])
        elif(tipo == "Histórico"):
            VaR[i+tam_ventana] = calcular_VaR_historico(nconf, datos[i: i + tam_ventana])
            ES[i+tam_ventana] = calcular_ES(VaR[i+tam_ventana], datos[i: i + tam_ventana])
    return VaR, ES

def generar_RW_VaR_movil(tam_ventana, nconf, datos):
    '''
    Función para generar una Rolling Window de VaRcon volatilidad móvil bajo distribución normal de un activo financiero
    Input = Tamaño de la ventana, nivel de confianza y serie histórico del activo.
    Output = Numpy array con la rolling window de VaR móvil
    '''
    n = len(datos)
    VaR = np.zeros(n)
    for i in range(0, tam_ventana):
        VaR[i] = None
    for i in range(0, n-tam_ventana):
        sigma = np.std(datos[i: i + tam_ventana])
        VaR[i+tam_ventana] = (norm.ppf(1-nconf, 0, 1))*sigma
    return VaR