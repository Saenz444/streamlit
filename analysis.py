#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, mean_squared_error
from sklearn.svm import SVC
from sklearn.metrics import f1_score, jaccard_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os

# Ruta al archivo Excel (ajusta la ruta según sea necesario)
file_path = os.path.join(os.getcwd(), 'Datavf.xlsx')

# Imprime la ruta del archivo para verificar
print(f"Looking for the file at: {file_path}")

# Verifica si el archivo existe
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Lee el archivo de Excel
Data = pd.read_excel(file_path)

# Ahora puedes trabajar con el DataFrame en tu código
print(Data.head())

# Preprocesamiento de los datos para la distribución
estudiantes_CEPRE = Data.drop(['estudiante', 'si_no'], axis=1)
Columnas = estudiantes_CEPRE.columns.to_list()
nColumnas = ['P-1', 'P-2', 'P-3', 'P-4', 'P-5', 'P-6', 'P-7']
dfColumnas = dict(zip(Columnas, nColumnas))
estudiantes_CEPRE.rename(columns=dfColumnas, inplace=True)

Data_tesis = estudiantes_CEPRE[['P-3', 'P-2', 'P-7', 'P-4', 'P-1', 'P-6', 'P-5']]

def Calcular_Promedio(Dt):
    p3, p2, p7, p4, p1, p6, p5 = Dt
    return (p3 + p2 + p7 + p4 + p1) - (p6 + p5)

# Calcular el promedio de satisfacción
Data_tesis['PROM'] = Data_tesis[['P-3', 'P-2', 'P-7', 'P-4', 'P-1', 'P-6', 'P-5']].apply(Calcular_Promedio, axis=1)

# Definir el criterio de satisfacción
satisfaction_criteria = 4  # Ajusta este valor según sea necesario

# Crear la nueva columna 'Satisfecho/No Satisfecho'
Data_tesis['Satisfecho/No Satisfecho'] = Data_tesis['PROM'].apply(lambda x: 'Satisfecho' if x >= satisfaction_criteria else 'No Satisfecho')

# Verificar los datos nulos en las variables
print(Data_tesis.isnull().sum())

# Mostrar el DataFrame resultante
print(Data_tesis.head())

# Aplicar operaciones de rolling solo en las columnas numéricas
data_rolling = Data_tesis[['P-3', 'P-2', 'P-7', 'P-4', 'P-1', 'P-6', 'P-5']].rolling(30).mean()

# Visualización
fig, ax = plt.subplots(figsize=(16, 4))
sns.lineplot(data=data_rolling, palette="tab10", linewidth=1.5).set_title("EVOLUCIÓN DE LAS ENCUESTAS POR PREGUNTAS")

# Rolling mean del promedio de satisfacción
data_prom_rolling = Data_tesis['PROM'].rolling(80).mean()
fig, ax = plt.subplots(figsize=(16, 4))
sns.lineplot(data=data_prom_rolling, palette="tab10", linewidth=1.5).set_title("PROMEDIO DE LAS ENCUESTAS - CALCULADOS EN BASE A LA IMPORTANCIA POR PREGUNTAS")

t_Data = estudiantes_CEPRE[['P-1', 'P-2', 'P-3', 'P-4', 'P-5', 'P-6', 'P-7']].head(100).transpose()
t_Data = np.asarray(t_Data)

fig, ax = plt.subplots(figsize=(16, 4))
Graph = ax.imshow(t_Data, cmap='RdBu')
cbar = ax.figure.colorbar(Graph, ax=ax)
cbar.ax.set_ylabel("Escala de Likert", rotation=-90, va="bottom")
plt.show()

graph_Data = estudiantes_CEPRE[['P-1', 'P-2', 'P-3', 'P-4', 'P-5', 'P-6', 'P-7']].transpose()

metric = ['russellrao', 'rogerstanimoto', 'sokalmichener', 'chebyshev', 'kulsinski', 'cityblock', 'minkowski', 'euclidean', 'hamming', 'jaccard', 'matching', 'sqeuclidean', 'yule']
Grafico = sns.clustermap(graph_Data, metric=metric[3], cmap='vlag_r', figsize=(16, 3), row_cluster=False, dendrogram_ratio=(.1, .2), cbar_pos=(0, .2, .03, .4))

ok_Data_Cols = graph_Data.columns[Grafico.dendrogram_col.reordered_ind]
ok_Data = graph_Data[ok_Data_Cols]
ok_Data = ok_Data.transpose()
idx_List = ok_Data.index
ok_Data.reset_index(inplace=True, drop=True)
ok_Data = ok_Data.transpose()
ok_Data

fig, ax = plt.subplots(figsize=(16, 4))
Graph = ax.imshow(ok_Data, cmap='RdBu')
cbar = ax.figure.colorbar(Graph, ax=ax)
cbar.ax.set_ylabel("Escala de Likert", rotation=-90, va="bottom")
plt.show()

ok_Data.describe()
Data_tesis[['P-3', 'P-2', 'P-7', 'P-4', 'P-1', 'P-6', 'P-5']].describe()

# Segmentación de la escala de Likert
sum_Cols = ok_Data.apply(lambda x: sum(ok_Data[x]))
Max, Min = max(sum_Cols), min(sum_Cols)
print(Max, Min)

# Calculamos los Quintiles
Qt_1 = [7.0,  12.6]     # Totalmente en desacuerdo
Qt_2 = [12.6, 18.2]     # En desacuerdo
Qt_3 = [18.2, 23.8]     # Ni de acuerdo ni en desacuerdo
Qt_4 = [23.8, 29.4]     # De acuerdo
Qt_5 = [29.4, 35.0]     # Totalmente de acuerdo

data_lk_1 = sum_Cols[(sum_Cols >= Qt_1[0]) & (sum_Cols < Qt_1[1])]
data_lk_2 = sum_Cols[(sum_Cols >= Qt_2[0]) & (sum_Cols < Qt_2[1])]
data_lk_3 = sum_Cols[(sum_Cols >= Qt_3[0]) & (sum_Cols < Qt_3[1])]
data_lk_4 = sum_Cols[(sum_Cols >= Qt_4[0]) & (sum_Cols < Qt_4[1])]
data_lk_5 = sum_Cols[(sum_Cols >= Qt_5[0]) & (sum_Cols <= Qt_5[1])]

cn1, cn2, cn3, cn4, cn5 = len(data_lk_1), len(data_lk_2), len(data_lk_3), len(data_lk_4), len(data_lk_5)

ls_1 = list(data_lk_1) + list([0]*(len(sum_Cols)-cn1))
ls_2 = list([0]*(cn1 + 1)) + list(data_lk_2) + list([0]*(len(sum_Cols)-(cn1 + cn2)))
ls_3 = list([0]*(cn1 + cn2 + 1)) + list(data_lk_3) + list([0]*(len(sum_Cols) - (cn1 + cn2 + cn3)))
ls_4 = list([0]*(cn1 + cn2 + cn3 + 1)) + list(data_lk_4) + list([0] * (len(sum_Cols) - (cn1 + cn2 + cn3 + cn4)))
ls_5 = list([0]*(cn1 + cn2 + cn3 + cn4 + 1)) + list(data_lk_5) + list([0]*1)

Columnas = ['Muy en desacuerdo', 'En desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'De acuerdo', 'Totalmente de acuerdo']
Colors = ['brown', 'red', 'orange', 'limegreen', 'darkgreen']
Percentage = ['{0:.2f}%'.format(cn1/len(sum_Cols)*100),
              '{0:.2f}%'.format(cn2/len(sum_Cols)*100),
              '{0:.2f}%'.format(cn3/len(sum_Cols)*100),
              '{0:.2f}%'.format(cn4/len(sum_Cols)*100),
              '{0:.2f}%'.format(cn5/len(sum_Cols)*100)]

fig, ax = plt.subplots(figsize=(16, 4))
sns.lineplot(data=ls_1, color=Colors[0], linewidth=1.5, label=Columnas[0] + ' ' + Percentage[0])
sns.lineplot(data=ls_2, color=Colors[1], linewidth=1.5, label=Columnas[1] + ' ' + Percentage[1])
sns.lineplot(data=ls_3, color=Colors[2], linewidth=1.5, label=Columnas[2] + ' ' + Percentage[2])
sns.lineplot(data=ls_4, color=Colors[3], linewidth=1.5, label=Columnas[3] + ' ' + Percentage[3])
sns.lineplot(data=ls_5, color=Colors[4], linewidth=1.5, label=Columnas[4] + ' ' + Percentage[4])
plt.legend(title="SEGMENTACIÓN DE LA ESCALA DE LIKERT")
plt.show()

print(cn1 + cn2 + cn3 + cn4 + cn5)
print(min(data_lk_5), max(data_lk_5))
Percentage

Data_tesis_model = Data_tesis[['P-3', 'P-2', 'P-7', 'P-4', 'P-1', 'P-6', 'P-5']]
Scaler = MinMaxScaler().fit(Data_tesis_model.values)

Data_tesis_model = pd.DataFrame(Scaler.transform(Data_tesis_model.values), columns=['P-3', 'P-2', 'P-7', 'P-4', 'P-1', 'P-6', 'P-5'])

kmeans = KMeans(n_clusters=5).fit(Data_tesis_model.values)
Data_tesis_model['Cluster'] = kmeans.labels_

print(kmeans.cluster_centers_, kmeans.inertia_)
Data_tesis_model

plt.figure(figsize=(8, 6), dpi=100)
colores = ['red', 'b', 'orange', 'b', 'purple', 'pink', 'brown']
for cluster in range(kmeans.n_clusters):
    plt.scatter(Data_tesis_model[Data_tesis_model['Cluster'] == cluster]['P-3'],
                Data_tesis_model[Data_tesis_model['Cluster'] == cluster]['P-2'],
                marker='o', s=50, color=colores[cluster], alpha=.5)
    plt.scatter(kmeans.cluster_centers_[cluster][0],
                kmeans.cluster_centers_[cluster][1],
                marker='P', s=100, color=colores[cluster])

plt.title('NIVEL DE SATISFACCIÓN')
plt.show()

# Guardar el DataFrame modificado en un nuevo archivo Excel
output_file_path = os.path.join(os.getcwd(), 'Data_Modificado.xlsx')
Data_tesis.to_excel(output_file_path, index=False)
print(f"El archivo modificado ha sido guardado en: {output_file_path}")

def generate_plots(data):
    # Ejemplo de generación de gráficos
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20)
    plt.xlabel('Valores')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de Datos')
    plt.savefig('histogram.png')  # Guarda el gráfico como un archivo de imagen
