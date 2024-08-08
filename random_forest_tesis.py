
# Tratamiento de datos
# =================================================================
import numpy as np
import pandas as pd

# Graficos
# =================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento y modelado de los datos
# =================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedKFold, GridSearchCV, ParameterGrid
from sklearn.inspection import permutation_importance
import multiprocessing

from sklearn.svm import SVC         # Support Vector Clasification
from sklearn.metrics import f1_score, jaccard_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder

import os
import pickle

# Leer el data

# Ruta al archivo Excel (o la ruta correcta a tu conjunto de datos)
archivo_excel = 'Datavf_Modificado.xlsx'

# Leer el conjunto de datos con pandas
Data = pd.read_excel(archivo_excel)

# Convertir la columna 'Satisfaccion' a numérica
label_encoder = LabelEncoder()
Data['Satisfaccion'] = label_encoder.fit_transform(Data['Satisfaccion'])

# Preprocesamiento de los datos para la distribución
estudiantes_CEPRE = Data.drop(['PROM'], axis=1)  # Eliminar la columna 'PROM'
Columnas = estudiantes_CEPRE.columns.to_list()
nColumnas = ['P-1', 'P-2', 'P-3', 'P-4', 'P-5', 'P-6', 'P-7']
dfColumnas = dict(zip(Columnas, nColumnas))
estudiantes_CEPRE.rename(columns=dfColumnas, inplace=True)

# Ajuste del modelo y optimización de hiperparámetros
X_train, X_test, y_train, y_test = train_test_split(
    estudiantes_CEPRE.drop('Satisfaccion', axis=1),  # Eliminar la columna 'Satisfaccion' para los features
    estudiantes_CEPRE['Satisfaccion'],  # Utilizar la columna 'Satisfaccion' como objetivo
    random_state=123
)
print(len(X_train), len(X_test), len(y_train), len(y_test))

# Grid Search basado en validación cruzada
# =================================================================================================
param_grid = {'n_estimators': [150], #la data que utiliza
               'max_features': [3, 5, 7], #las preguntas
               'max_depth': [None, 3, 10, 20],
               'criterion': ['gini', 'entropy']
               }
# Búsqueda por grid search con validación cruzada
grid = GridSearchCV(
    estimator  = RandomForestClassifier(random_state = 123),
    param_grid = param_grid,
    scoring    = 'accuracy',
    n_jobs     = multiprocessing.cpu_count() - 1,
    cv         = RepeatedKFold(n_splits=5, n_repeats=3, random_state=123),
    refit      = True,
    verbose    = 0,
    return_train_score = True
)
m_rf = grid.fit(X = X_train, y = y_train)

# Resultados
resultados = pd.DataFrame(m_rf.cv_results_)
print(resultados.filter(regex = '(param*|mean_t|std_t)').drop(columns = 'params').sort_values('mean_test_score', ascending = False).head(10)) #Encontrar las 10 mejores combinaciones de hiperparametrp en funcion de putaje de prueba

# Obtener la mitad del rango de mean_test_score
mean_test_score_midpoint = resultados['mean_test_score'].min() + (resultados['mean_test_score'].max() - resultados['mean_test_score'].min()) / 2

# Crear un gráfico para mostrar la relación entre n_estimators y mean_test_score
for criterion in param_grid['criterion']:
    subset = resultados[resultados['param_criterion'] == criterion]
    color = 'green' if criterion == 'entropy' else 'red'  # Colorear 'entropy' en verde y 'gini' en rojo
    label = f'{criterion.capitalize()} criterion'
    plt.plot(subset['param_n_estimators'], subset['mean_test_score'], marker='o', color=color, label=label)

plt.axhline(y=mean_test_score_midpoint, color='gray', linestyle='--')  # Línea horizontal en la mitad
plt.xlabel('n_estimators')
plt.ylabel('mean_test_score')
plt.title('Relación entre n_estimators y mean_test_score en RandomForest según el criterio')
plt.legend()
plt.show()

# Mejores hiperparámetros por validación cruzada
# ****************************************************************
print(grid.best_params_, ":", grid.best_score_, grid.scoring)

modelo_final = grid.best_estimator_

# =================================================================================================
# Predicción y evaluación del modelo
# =================================================================================================
predicciones = modelo_final.predict(X = X_test)
print(predicciones[:20])

# Medir el rendimiento del modelo
mat_confusion = confusion_matrix(
    y_true    = y_test,
    y_pred    = predicciones
)

# Calcular la precision
accuracy = accuracy_score(
    y_true    = y_test,
    y_pred    = predicciones,
    normalize = True
)
print(f'Precisión del modelo: {accuracy:.2f}')

print("Matriz de confusión")
print("-------------------")
print(mat_confusion)
print("")
print(f"El accuracy de test es: {100 * accuracy} %")

print(
    classification_report(
        y_true = y_test,
        y_pred = predicciones
    )
)

plt.figure(figsize=(8, 6))
sns.heatmap(mat_confusion, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
plt.xlabel('Etiqueta Real')
plt.ylabel('Etiqueta Predicha')
plt.title('Matriz de Confusión')
plt.show()

# Predicción de probabilidades
# =================================================================================================
predicciones = modelo_final.predict_proba(X = X_test)
print(predicciones[:5, :])

nuevo_modelo = RandomForestClassifier(random_state=123)
nuevo_modelo.fit(X_train, y_train)

# Importancia por permutación
importancia = permutation_importance(
   estimator    = nuevo_modelo,
    X            = X_train,
    y            = y_train,
    n_repeats    = 5,
    scoring      = 'neg_root_mean_squared_error',
    n_jobs       = multiprocessing.cpu_count() - 1,
    random_state = 123
)

# Se almacenan los resultados (media y desviación) en un dataframe
df_importancia = pd.DataFrame(
    { k: importancia[k] for k in ['importances_mean', 'importances_std'] }
)
df_importancia['predictor'] = X_train.columns
print(df_importancia.sort_values('importances_mean', ascending=False))

# Graficamos los resultados
# ************************************************************************************************
color = ['y','y','y','y','g','g','g']
fig, ax = plt.subplots(figsize=(5, 6))
df_importancia = df_importancia.sort_values('importances_mean', ascending=True)
ax.barh(
    df_importancia['predictor'],
    df_importancia['importances_mean'],
    xerr=df_importancia['importances_std'],
    align='center',
    alpha=1,
    color=color
)
ax.plot(
    df_importancia['importances_mean'],
    df_importancia['predictor'],
    marker="o",
    linestyle="",
    alpha=0.8,
    color="r"
)
ax.tick_params(axis='x', labelsize=9)
ax.tick_params(axis='y', labelsize=8)
plt.grid(alpha=.5)
plt.show()

"""###**MODELO DE MAQUINAS DE VECTORES DE SOPORTE**
Support Vector Machine
"""

# Para el modelo SVM
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    estudiantes_CEPRE.drop('Satisfaccion', axis=1),  # Eliminar la columna 'Satisfaccion' para los features
    estudiantes_CEPRE['Satisfaccion'],  # Utilizar la columna 'Satisfaccion' como objetivo
    random_state=5
)

modelo_svm = SVC(kernel='rbf', C=1)  # rbf, linear, poly, sigmoid
m_svm = modelo_svm.fit(X_train_svm, y_train_svm)

print('Precision SVM:', m_svm.score(X_test_svm, y_test_svm))

# Predicción inicial
predict_svm = m_svm.predict(X_test_svm)
matriz_confussion = confusion_matrix(y_test_svm, predict_svm)

print('\nMATRIZ DE CONFUSION SVM \n***************************************************')
print(matriz_confussion)

f1_score_svm = f1_score(y_test_svm, predict_svm, average='weighted')
jac_score_svm = jaccard_score(y_test_svm, predict_svm, pos_label=1)

print(f1_score_svm, jac_score_svm)

"""###**MODELO KNN**

Predicción utilizando el modelo de clasificación KNN
"""

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    estudiantes_CEPRE.drop('Satisfaccion', axis=1),  # Eliminar la columna 'Satisfaccion' para los features
    estudiantes_CEPRE['Satisfaccion'],  # Utilizar la columna 'Satisfaccion' como objetivo
    random_state=5
)

modelo_KNN = KNeighborsClassifier(n_neighbors=3)
m_KNN = modelo_KNN.fit(X_train_knn, y_train_knn)

print('Precision KNN:', m_KNN.score(X_train_knn, y_train_knn))

predict_KNN = m_KNN.predict(X_test_knn)
matriz_confussion_knn = confusion_matrix(y_test_knn, predict_KNN)

print('PREDICCIONES: ', predict_KNN[0:30])
print('\nMATRIZ DE CONFUSION KNN \n***************************************************')
print(matriz_confussion_knn)

# Guardar los modelos utilizando pickle
with open('m_rf.pkl', 'wb') as rf:
    pickle.dump(m_rf, rf)
    
with open('m_svm.pkl', 'wb') as svm_file:
    pickle.dump(m_svm, svm_file)
    
with open('m_KNN.pkl', 'wb') as knn_file:
    pickle.dump(m_KNN, knn_file)
