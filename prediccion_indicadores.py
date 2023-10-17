import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from typing import Union
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.simplefilter("ignore", UserWarning)

def prepare_data(excel_file: str, tipo: Union['escuela','departamento']):
    """
    Prepara los datos cargados desde un archivo Excel para el análisis.

    Args:
        excel_file (str): Ruta del archivo Excel que contiene los datos.
        tipo (str): Tipo de entidad a analizar ('escuela' o 'departamento').

    Returns:
        None
    """
    global df
    global tipaso
    tipaso = tipo
    df = pd.read_excel(excel_file, sheet_name=0)
    df.drop(['fecha_inicio','fecha_fin','total'], axis=1, inplace=True)

    grouped = df.groupby(['semestre',tipo,'proceso'], as_index=False)
    df = grouped[['a_tiempo','fuera_tiempo']].sum()
    date = pd.to_datetime(df.semestre)
    date[date.dt.month == 2] += pd.DateOffset(months=5)
    df['date'] = date
    df['predecido'] = False

def get_timeline(nombre: str, proceso: str, n_predictions: int=0):
    """
    Obtiene una línea de tiempo de indicadores de cumplimiento para una entidad específica.

    Args:
        nombre (str): Nombre de la escuela o departamento.
        proceso (str): Nombre del proceso.
        n_predictions (int): Número de predicciones a generar.

    Returns:
        pd.DataFrame: DataFrame con la línea de tiempo de indicadores.
    """
    timeline = df[(df.proceso == proceso) & (df[tipaso] == nombre)]
    if n_predictions == 0:
        return timeline
    predictions = get_predictions(nombre, proceso, n_predictions)
    return pd.concat([timeline, predictions], ignore_index=True)

def plot_timeline(nombre: str, proceso: str, n_predictions: int=0):
    """
    Genera un gráfico de la línea de tiempo de indicadores de cumplimiento.

    Args:
        nombre (str): Nombre de la escuela o departamento.
        proceso (str): Nombre del proceso.
        n_predictions (int): Número de predicciones a mostrar en el gráfico.

    Returns:
        None
    """
    timeline = get_timeline(nombre, proceso, n_predictions)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f'Linea de tiempo de {tipaso} {nombre} en el proceso {proceso}')
    ax.set_xlabel('Semestre')
    if n_predictions == 0:
        ax.plot(timeline.semestre, timeline.a_tiempo, label='A tiempo', color='#22dd22')
        ax.plot(timeline.semestre, timeline.fuera_tiempo, label='Fuera de tiempo', color='#cc7777')
    else:
        i = timeline[timeline.predecido == False].index[-1]
        sin_predecir = timeline.loc[:i]
        predecidos = timeline.loc[i:]

        ax.axvline(i, ls='--', lw=1.5, c='k', alpha=.2)
        ax.plot(sin_predecir.semestre, sin_predecir.a_tiempo, '.-', predecidos.semestre, predecidos.a_tiempo, '.--', label='A tiempo', color='#22dd22')
        ax.plot(sin_predecir.semestre, sin_predecir.fuera_tiempo, '.-', predecidos.semestre, predecidos.fuera_tiempo, '.--', label='Fuera de tiempo', color='#cc7777')
    ax.legend()
    plt.show()
    
def plot_partial_acf(nombre: str, proceso: str, a_tiempo: bool=True):
    """
    Genera un gráfico de la función de autocorrelación parcial.

    Args:
        nombre (str): Nombre de la entidad.
        proceso (str): Nombre del proceso.
        a_tiempo (bool): True si se va a analizar 'a_tiempo', False si se analiza 'fuera_tiempo'.

    Returns:
        None
    """
    tipo = 'a_tiempo' if a_tiempo else 'fuera_tiempo'
    timeline = get_timeline(nombre=nombre, proceso=proceso)
    _ = plot_pacf(timeline[tipo],lags=4 , zero=False, alpha=.1)

def convert_timeline_to_timeseries(timeline, a_tiempo: bool=True):
    """
    Convierte una línea de tiempo en un objeto de series de tiempo.

    Args:
        timeline (pd.DataFrame): DataFrame de la línea de tiempo.
        a_tiempo (bool): True si se va a analizar 'a_tiempo', False si se analiza 'fuera_tiempo'.

    Returns:
        pd.Series: Objeto de series de tiempo con fechas como índice.
    """
    tipo = 'a_tiempo' if a_tiempo else 'fuera_tiempo'
    timeseries = timeline[['date',tipo]]
    timeseries.set_index('date', inplace=True)
    timeseries = timeseries.squeeze()
    timeseries = timeseries.asfreq('2QS-JAN')
    return timeseries

def convert_predictions_to_semester_format(predictions, a_tiempo=True):
    """
    Convierte las predicciones en formato semestral.

    Args:
        predictions (pd.DataFrame): DataFrame de predicciones.
        a_tiempo (bool): True si se va a analizar 'a_tiempo', False si se analiza 'fuera_tiempo'.

    Returns:
        pd.DataFrame: DataFrame con las predicciones en formato semestral.
    """
    tipo = 'a_tiempo' if a_tiempo else 'fuera_tiempo'
    x = predictions.reset_index(name=tipo)
    x.loc[x['index'].dt.month == 7, 'index'] -= pd.DateOffset(months=5)
    x['semestre'] = x['index'].dt.year.astype('str') + '-' +x['index'].dt.month.astype('str')
    x.drop('index', axis=1, inplace=True)
    return x
    
def get_model(nombre: str, proceso: str, a_tiempo: bool=True):
    """
    Obtiene un modelo ARIMA para una entidad y un proceso específicos.

    Args:
        nombre (str): Nombre de la escuela o departamento.
        proceso (str): Nombre del proceso.
        a_tiempo (bool): True si se va a analizar 'a_tiempo', False si se analiza 'fuera_tiempo'.

    Returns:
        statsmodels.tsa.arima.model.ARIMAResultsWrapper: Modelo ARIMA ajustado.
    """
    timeline = get_timeline(nombre=nombre, proceso=proceso)

    nlags = int((timeline.shape[0] - 1) / 2)
    train_data = convert_timeline_to_timeseries(timeline, a_tiempo)

    model = ARIMA(train_data, order=(nlags, 0, 0))
    model.initialize_approximate_diffuse()
    model_fit = model.fit()
    return model_fit

def get_predictions(nombre: str, proceso: str, n_predictions: int):
    """
    Obtiene las predicciones futuras de indicadores de cumplimiento.

    Args:
        nombre (str): Nombre de la escuela o departamento.
        proceso (str): Nombre del proceso.
        n_predictions (int): Número de predicciones a generar.

    Returns:
        pd.DataFrame: DataFrame con las predicciones futuras.
    """
    if n_predictions <= 0:
        print('n_predictions tiene que ser mayor que 0')
        return None
    
    timeline = get_timeline(nombre=nombre, proceso=proceso)
    n_training = timeline.shape[0]
    
    if n_training < 4:
        print('Se requieren al menos 4 datos para hacer la prediccion')
        return None

    model_fit_a_tiempo = get_model(nombre, proceso, a_tiempo=True)

    predictions_a_tiempo = model_fit_a_tiempo.predict(start=n_training, end=n_training+n_predictions-1)
    predictions_a_tiempo[predictions_a_tiempo < 0] = 0
    predictions_a_tiempo = predictions_a_tiempo.round().astype('int64')

    model_fit_fuera_tiempo = get_model(nombre, proceso, a_tiempo=False)

    predictions_fuera_tiempo = model_fit_fuera_tiempo.predict(start=n_training, end=n_training+n_predictions-1)
    predictions_fuera_tiempo[predictions_fuera_tiempo < 0] = 0
    predictions_fuera_tiempo = predictions_fuera_tiempo.round().astype('int64')
    
    predictions_a_tiempo.index.rename(None, inplace=True)
    predictions_fuera_tiempo.index.rename(None, inplace=True)

    predictions_a_tiempo = convert_predictions_to_semester_format(predictions_a_tiempo, a_tiempo=True)
    predictions_fuera_tiempo = convert_predictions_to_semester_format(predictions_fuera_tiempo, a_tiempo=False)
    predictions = predictions_a_tiempo.merge(predictions_fuera_tiempo, 'inner', 'semestre')
    
    date = pd.to_datetime(predictions.semestre)
    date[date.dt.month == 2] += pd.DateOffset(months=5)
    predictions['date'] = date
    predictions[tipaso] = nombre
    predictions['proceso'] = proceso
    predictions['predecido'] = True
    
    return predictions

#n_predictions = 2
#new_df = df.copy()
#predictions = get_predictions_escuela_proceso(escuela, proceso, n_predictions)
#pd.concat([new_df, predictions], ignore_index=True)
#
#escuelas_y_procesos = df.drop_duplicates(['escuela','proceso'])[['escuela', 'proceso']].copy()
#for index, row in escuelas_y_procesos.iterrows():
#    print(row.escuela, row.proceso)
#    get_predictions_escuela_proceso(escuela=row.escuela, proceso=row.proceso, n_predictions=1)

prepare_data('escuela_indicadores.xlsx', tipo='escuela')
plot_timeline('ADMINISTRACIÓN', 'Carga', 2)