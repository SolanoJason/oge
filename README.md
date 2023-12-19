# Prediccion de los indicadores de las escuelas y departamentos
Este un sistema que permite predecir indicadores de cumplimiento de procesos en una universidad para diferentes entidades, ya sea escuelas o departamentos. Utiliza modelos ARIMA para realizar predicciones basadas en datos históricos.

## Instalacion en local
### Requerimientos
* Python 3.10
* Git

### Procedimiento
1. Clone este proyecto, usando el comando:
> git clone <https://github.com/SolanoJason/oge.git>
2. Instalar los packages necesarios usando el comando:
> pip install -r requirements.txt
### Prediccion de indicadores
3. Importamos las 3 funciones más importantes
```python
from prediccion_indicadores import prepare_data, get_timeline, plot_timeline
```
4. Cargar los datos desde un archivo excel, indicando si es de escuelas o departamentos
```python
prepare_data('escuela_indicadores.xlsx', tipo='escuela')
```
5. Graficamos especificando la escuela, proceso y el numero de semestres a predecir
```python
nombre = 'ADMINISTRACIÓN'
proceso = 'Carga'
n_predicciones = 3
plot_timeline(nombre, proceso, n_predicciones)
```
![alt text](https://github.com/SolanoJason/oge/blob/main/escuela_grafica.png?raw=true)
6. Si queremos los numeros exactos usamos la funcion 'get_timeline' para obtener la tabla de datos que se graficó
```python
get_timeline(nombre, proceso, n_predicciones)
```
![alt text](https://github.com/SolanoJason/oge/blob/main/escuela_timeline.png?raw=true)

### Prediccion de la carga academica
7. Importamos las funciones
```python
from prediccion_carga_academica import predecir_carga_academica, predecir_profesor, semestre_2022_2, df_test
```

8. Predecimos la carga academica de un profesor
```python
profesor = 'SZM001'
predecir_carga_academica(profesor)
```
![alt text](https://github.com/SolanoJason/oge/blob/main/prediccion_profesor.png?raw=true)

9. Asignamos un profesor a una escuela y un curso
```python
escuela = '36'
curso = 'C78'
predecir_profesor(escuela, curso)
```
![alt text](https://github.com/SolanoJason/oge/blob/main/asignar_profesor.png?raw=true)