import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
from sqlalchemy import create_engine
from mpl_toolkits import mplot3d
from surprise import Reader, Dataset, SVD, accuracy, KNNBasic, BaselineOnly, NormalPredictor, KNNWithMeans, SVDpp, Trainset
from surprise.model_selection import cross_validate, train_test_split, KFold
from surprise.accuracy import rmse, mae, mse
from surprise.similarities import cosine
from surprise.prediction_algorithms.predictions import Prediction

SERVER = 'DESKTOP-4AI1QI3'
DATABASE = 'Itunasam1404'
USER = 'sa'
PASSWORD = '011235813213455'
connection_string = f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server"
con = create_engine(connection_string)

Alumno = pd.read_sql_table('Alumno', con, 'dbo')

Curso = pd.read_sql_table('Curso', con, 'dbo')
Curso.drop(['CreditosAprobados','Horario','Sumilla','Tipo','Distancia','codigo','resumen','formato_virtual','prediccion_matriculados','DepartamentoAcademico','DocentePrincipal','DocenteAuxiliar'], axis=1, inplace=True)
Curso.Ciclo, Curso.Curso = Curso.Ciclo.str.strip(), Curso.Curso.str.strip()

Escuela = pd.read_sql_table('Escuela', con, 'dbo')
Escuela.drop(['FechaCreacion','FechaFuncionamiento','NombreGrado','NombreTituloVaron','NombreTituloDama','ubigeo',
              'ResolucionEscuela','ConsiderarGrados','DispositivoFuncionamiento','DuracionAnios','Creditos',
              'NivelGrado','DuracionSemestres','Descripcion'], axis=1, inplace=True)
Escuela.drop(Escuela[Escuela.Escuela == '33'].index, axis=0, inplace=True)

Curricula = pd.read_sql_table('Curricula', con, 'dbo')
Curricula.drop(['Ciclo'], axis=1, inplace=True)
Curricula.head()

Semestre = pd.read_sql_table('Semestre', con, 'dbo')
Semestre.drop(['nombre','Activo','vigente','especial','cuenta_veces_desaprobadas','conectado_tesoreria'], axis=1, inplace=True)
Semestre.sort_values('orden', inplace=True)
Semestre.head()

DepartamentoAcademico = pd.read_sql_table('DepartamentoAcademico', con, 'dbo')

Trabajador = pd.read_sql_table('Trabajador', con, 'dbo')
Trabajador.drop(['Telefono','TelefonoEmergencia','Celular','Direccion','PaginaWeb',
                 'Email','FechaIngreso','FechaEgreso','FechaNacimiento','Observaciones',
                 'Dni','Foto','password','PASSMD5','remember_token','categoria_original',
                 'correo_institucional','id','ApellidoPaterno','ApellidoMaterno','Nombre'], axis=1, inplace=True)
Trabajador.Trabajador, Trabajador.Profesion = Trabajador.Trabajador.str.strip(), Trabajador.Profesion.str.strip()
Trabajador.drop_duplicates('Trabajador', inplace=True, keep='last')

CursoProgramado = pd.read_sql_table('CursoProgramado', con, 'dbo',)
CursoProgramado.drop(CursoProgramado[CursoProgramado.Sede == '1 '].index, inplace=True)
CursoProgramado.drop(['created_at','updated_at','FechaRegistro','EstacionRegistro','UsuarioRegistro',
                      'observaciones','programador_id','no_presencial','justificacion_no_presencial',
                      'ad_honorem','modificable','Silabo','CursoProgramado','Nombredocente'], axis=1, inplace=True)
                                    #    columns=['Sede','Escuela','Curricula','Curso','Semestre','Seccion','Grupo','CuentaCarga','Capacidad','Trabajador'])
CursoProgramado.Curso, CursoProgramado.Trabajador, CursoProgramado.TrabajadorTeoria = CursoProgramado.Curso.str.strip(), CursoProgramado.Trabajador.str.strip(), CursoProgramado.TrabajadorTeoria.str.strip()
CursoProgramado = CursoProgramado.merge(Escuela[['Escuela','Abreviatura']], 'left', 'Escuela').merge(Curso[['Escuela','Curricula','Curso','Nombre']], 'left', ['Escuela','Curricula','Curso']).merge(Trabajador[['Trabajador','NombreCompleto']], 'left','Trabajador')

CursoProgramadoGrupo = pd.read_sql_table('CursoProgramadoGrupo', con, 'dbo',)
CursoProgramadoGrupo.drop(CursoProgramadoGrupo[CursoProgramadoGrupo.Sede == '1 '].index, inplace=True)
CursoProgramadoGrupo.drop(['created_at','updated_at','UsuarioRegistro','FechaRegistro','EstacionRegistro','programador_id','CursoProgramadoGrupo'], axis=1, inplace=True)
                                    #    columns=['Sede','Escuela','Curricula','Curso','Semestre','Seccion','Grupo','CuentaCarga','Capacidad','Trabajador'])
CursoProgramadoGrupo.Curso, CursoProgramadoGrupo.Trabajador = CursoProgramadoGrupo.Curso.str.strip(), CursoProgramadoGrupo.Trabajador.str.strip()
CursoProgramadoGrupo = CursoProgramadoGrupo.merge(Escuela[['Escuela','Abreviatura']], 'left', 'Escuela').merge(Curso[['Escuela','Curricula','Curso','Nombre']], 'left', ['Escuela','Curricula','Curso']).merge(Trabajador[['Trabajador','NombreCompleto']], 'left','Trabajador')

CargaLectivaZet = pd.read_sql_table('CargaLectivaZet', con, 'dbo')
CargaLectivaZet.drop(['created_at','updated_at','programador_id','programador_type','modificable','Ingreso','Tipito','CargaLectiva','autogenerado'], axis=1, inplace=True)
CargaLectivaZet.drop_duplicates(inplace=True)
CargaLectivaZet.Curso, CargaLectivaZet.Trabajador = CargaLectivaZet.Curso.str.strip(), CargaLectivaZet.Trabajador.str.strip()
CargaLectivaZet = CargaLectivaZet.merge(Escuela[['Escuela','Abreviatura']], 'left', 'Escuela').merge(Curso[['Escuela','Curricula','Curso','Nombre']], 'left', ['Escuela','Curricula','Curso']).merge(Trabajador[['Trabajador','NombreCompleto','Profesion']], 'left','Trabajador')

CargaLectivaZet_Eliminados = pd.read_sql_table('CargaLectivaZet_Eliminados', con, 'dbo')
CargaLectivaZet_Eliminados.drop(['created_at','updated_at','programador_id','programador_type','modificable','Sede','Ingreso','Tipito','CargaLectiva',
                                 'eliminador_id','eliminador_type','fecha_eliminacion'], axis=1, inplace=True)
CargaLectivaZet_Eliminados.Curso = CargaLectivaZet_Eliminados.Curso.str.strip()

CargaLectiva = pd.read_sql_table('CargaLectiva', con, 'dbo')

Docentes = pd.read_sql_table('docentes', con, 'dbo')
Docentes.trabajador, Docentes.titulo_profesional = Docentes.trabajador.str.strip(), Docentes.titulo_profesional.str.strip()
Docentes['NombreCompleto'] = Docentes.apellido_paterno + ' ' + Docentes.apellido_materno + ' ' + Docentes.nombres
Docentes.drop(['tipo_documento','dni','apellido_paterno','apellido_materno','nombres','fecha_ingreso_unasam','universidad_titulo','cargo',
               'tipo_dependencia','semestre','tiene_proyecto','pais_titulo'], axis=1, inplace=True)
Docentes.drop_duplicates('trabajador', inplace=True)

index_columns = ['Sede','Escuela','Curricula','Curso','Semestre','Seccion','Grupo']

def logistic_function(x, k, x_0, L):
    """
    k: factor de crecimiento
    x_0: Valor de x donde el resultado sera L/2
    L: valor maximo de la funcion
    """
    return L / (1 + np.exp(-k * (x - x_0)))

trabajadores_no_ensenan = np.setdiff1d(Trabajador.Trabajador.unique(), CargaLectivaZet.Trabajador.unique(), assume_unique=True)

# Eliminamos a los trabajadores que no enseñan, ya que solo nos interesa los docentes
Trabajador.drop(Trabajador[Trabajador.Trabajador.isin(trabajadores_no_ensenan)].index, inplace=True)

Trabajador_Docentes = Trabajador.merge(Docentes, 'left', left_on='Trabajador', right_on='trabajador')
Trabajador_Docentes.Profesion.replace('', np.nan, inplace=True)
Trabajador_Docentes.Profesion.fillna(Trabajador_Docentes.titulo_profesional, inplace=True)
Trabajador_Docentes.drop(['trabajador','NombreCompleto_y','titulo_profesional'], axis=1, inplace=True)
Trabajador_Docentes.shape

# Numero de cursos programados que tienen a 2 docentes
grouped = CargaLectivaZet.groupby(index_columns)
carga_2_trabajadores = grouped.filter(lambda group: group['Trabajador'].nunique() == 2)
frac = carga_2_trabajadores.drop_duplicates(['Sede','Escuela','Curricula','Curso','Semestre','Seccion','Grupo']).sort_values('Curso').shape[0] / CargaLectivaZet.drop_duplicates(['Sede','Escuela','Curricula','Curso','Semestre','Seccion','Grupo']).sort_values('Curso').shape[0]
print(f'El {frac*100:.2f}% de los cursos programados tiene a 2 docentes simultaneos')

merged = CursoProgramadoGrupo.merge(CargaLectivaZet.drop_duplicates(index_columns)[index_columns+['Trabajador']], how='left', on=index_columns)
merged.Trabajador_x.fillna(merged.Trabajador_y, inplace=True)
merged.drop('Trabajador_y', axis=1, inplace=True)
merged.sort_values(index_columns, inplace=True)
merged = merged.merge(CursoProgramado[index_columns[:-1]+['Trabajador']], 'left', index_columns[:-1])
merged.Trabajador_x.fillna(merged.Trabajador, inplace=True)
merged.drop('Trabajador', axis=1, inplace=True)
merged.dropna(axis=0, subset='Trabajador_x', inplace=True)
df = merged.copy()
df.drop(['CuentaCarga','Capacidad','modificable'], axis=1, inplace=True)
df = df.merge(Semestre[['Semestre','orden']], 'left', 'Semestre') 
df.orden = df.orden.astype('int')
df.sort_values(['Curso','orden','Seccion','Grupo','Curricula','Escuela'], inplace=True)
df = df.merge(Curso[['Escuela','Curricula','Curso','NumeroCreditos']], 'left', ['Escuela','Curricula','Curso'])

orden = 139
trabajador_orden = df.groupby('Trabajador_x').agg(max_orden=('orden','max'))
trabajadores_inactivos = trabajador_orden[trabajador_orden.max_orden <= orden].index
Trabajador_Docentes.loc[Trabajador_Docentes.Trabajador.isin(trabajadores_inactivos), 'Activo'] = False

df_train, df_test = df[df.Semestre != '2022-2'].copy(), df[df.Semestre == '2022-2'].copy()
Trabajador_Docentes.loc[Trabajador_Docentes.Trabajador.isin(df_test.Trabajador_x.unique()), 'Activo'] = True
Trabajador_Docentes.drop(Trabajador_Docentes[Trabajador_Docentes.Activo == False].index, inplace=True)

k = 0.05
x_0 = 90

df_train['rating'] = df_train.orden.apply(lambda x: logistic_function(x, k, x_0, 10))

grouped = df_train.groupby(['Curso','Trabajador_x'], as_index=False)
values = grouped.agg(rating = ('rating', 'sum'))
values.sort_values(['Curso','rating'], ascending=False, inplace=True)

def get_score(k, x_0):
    df_train['rating'] = df_train.orden.apply(lambda x: logistic_function(x, k, x_0, 10))

    grouped = df_train.groupby(['Curso','Trabajador_x'], as_index=False)
    values = grouped.agg(rating = ('rating', 'sum'))
    values.sort_values(['Curso','rating'], ascending=False, inplace=True)

    semestre_2022_2 = df_test.loc[:, ['Escuela','Curricula','Curso','Semestre']]
    semestre_2022_2 = semestre_2022_2.merge(values.groupby('Curso').first()['Trabajador_x'], 'left', left_on='Curso', right_index=True)
    trabajadores_correctos = (df_test.Trabajador_x == semestre_2022_2.Trabajador_x).sum()
    total = df_test.shape[0]
    return trabajadores_correctos*100 / total

func = np.vectorize(get_score)

k_array = np.linspace(0, 2, num=50)
x_0_array = np.linspace(80, 200, num=20)
K, X_0 = np.meshgrid(k_array, x_0_array)
Z = func(K,X_0)

indices = np.unravel_index(Z.argmax(), Z.shape)
best_k = K[indices]
best_x_0 = X_0[indices]
print(f'{best_k=} {best_x_0=}')

df_train['rating'] = df_train.orden.apply(lambda x: logistic_function(x, best_k, best_x_0, 10))
grouped = df_train.groupby(['Curso','Trabajador_x'], as_index=False)
values = grouped.agg(rating = ('rating', 'sum'))
values.drop(values[values.Trabajador_x.isin(Trabajador_Docentes.loc[Trabajador_Docentes.Activo == False, 'Trabajador'])].index, inplace=True)
values.sort_values(['Curso','rating'], ascending=False, inplace=True)

semestre_2022_2 = df_test.loc[:, ['Escuela','Curricula','Curso','Semestre']].copy()
semestre_2022_2 = semestre_2022_2.merge(values.groupby('Curso').first()['Trabajador_x'], 'left', left_on='Curso', right_index=True)

trabajadores_correctos = (df_test.Trabajador_x == semestre_2022_2.Trabajador_x).sum()
total = df_test.shape[0]
trabajadores_correctos*100 / total

grouped = df_train.groupby(['Escuela','Trabajador_x'], as_index=False)
values_escuela = grouped.agg(rating = ('Escuela', 'count'))
values_escuela.sort_values(['Trabajador_x','rating'], inplace=True, ascending=False)
values_escuela = values_escuela.merge(values_escuela.groupby('Trabajador_x').rating.sum(), 'inner', left_on='Trabajador_x', right_index=True)
values_escuela.rating_x = values_escuela.rating_x / values_escuela.rating_y
values_escuela.drop('rating_y', axis=1, inplace=True)

def trabajador_tiene_cursos_nuevos(trabajador):
    cursos_2022 = df_test[df_test.Trabajador_x == trabajador].Curso.unique()
    cursos_antiguos = df_train[df_train.Trabajador_x == trabajador].Curso.unique()
    cursos_nuevos = np.setdiff1d(cursos_2022, cursos_antiguos)
    return cursos_nuevos.size != 0

n_trabajadores_cursos_nuevos = np.vectorize(trabajador_tiene_cursos_nuevos)(df_test.Trabajador_x.unique()).sum()

# Docentes que han enseñado en escuelas nuevas en el semestre 2022-2
def trabajador_tiene_escuelas_nuevas(trabajador):
    escuelas_2022 = df_test[df_test.Trabajador_x == trabajador].Escuela.unique()
    escuelas_antiguos = df_train[df_train.Trabajador_x == trabajador].Escuela.unique()
    escuelas_nuevos = np.setdiff1d(escuelas_2022, escuelas_antiguos)
    return escuelas_nuevos.size != 0

n_trabajadores_escuelas_nuevas = np.vectorize(trabajador_tiene_escuelas_nuevas)(df_test.Trabajador_x.unique()).sum()

def get_score(k, x_0):
    df_train['rating'] = df_train.orden.apply(lambda x: logistic_function(x, k, x_0, 100))

    values_both = df_train.groupby(['Escuela','Curso','Trabajador_x'], as_index=False).agg(rating = ('rating', 'sum'))
    values_both.sort_values(['Escuela','Curso','rating'], inplace=True, ascending=False)
    values_both = values_both[values_both.Trabajador_x.isin(Trabajador_Docentes.Trabajador)]

    semestre_2022_2 = df_test.loc[:, ['Escuela','Curricula','Curso','Semestre']].copy()
    semestre_2022_2 = semestre_2022_2.merge(values_both.groupby(['Escuela','Curso'])['Trabajador_x'].first(), 'left', left_on=['Escuela','Curso'], right_index=True)
    trabajadores_correctos = (df_test.Trabajador_x == semestre_2022_2.Trabajador_x).sum()
    total = df_test.shape[0]
    return trabajadores_correctos*100 / total

func = np.vectorize(get_score)

k_array = np.linspace(0.05, 1, num=50)
x_0_array = np.linspace(80, 185, num=30)
K, X_0 = np.meshgrid(k_array, x_0_array)
Z = func(K,X_0)

indices = np.unravel_index(Z.argmax(), Z.shape)
best_k = K[indices]
best_x_0 = X_0[indices]
print(f'{best_k=} {best_x_0=}')

df_train['rating'] = df_train.orden.apply(lambda x: logistic_function(x, best_k, best_x_0, 100))

values_both = df_train.groupby(['Escuela','Curso','Trabajador_x'], as_index=False).agg(rating = ('rating', 'sum'))
values_both.sort_values(['Escuela','Curso','rating'], inplace=True, ascending=False)
values_both = values_both[values_both.Trabajador_x.isin(Trabajador_Docentes.Trabajador)]

semestre_2022_2 = df_test.loc[:, ['Escuela','Curricula','Curso','Semestre']].copy()
semestre_2022_2 = semestre_2022_2.merge(values_both.groupby(['Escuela','Curso'])['Trabajador_x'].first(), 'left', left_on=['Escuela','Curso'], right_index=True)
trabajadores_correctos = (df_test.Trabajador_x == semestre_2022_2.Trabajador_x).sum()
total = df_test.shape[0]
print(f'Se ha predecido {trabajadores_correctos*100 / total:.2f}% de la carga academica correctamente')

escuela = '36'
curso = 'C78'
#profesor = 'WJML01'
profesor = 'ACL529'
profesor = 'SZM001'

def predecir_profesor(escuela, curso):
    return values_both[(values_both.Escuela == escuela) & (values_both.Curso == curso)]['Trabajador_x'].iloc[0]

def predecir_carga_academica(profesor):
    predecido = semestre_2022_2[semestre_2022_2.Trabajador_x == profesor]
    return predecido.merge(Curso.drop_duplicates(['Curso'])[['Curso', 'Nombre']], on='Curso', how='left').drop_duplicates('Curso')

