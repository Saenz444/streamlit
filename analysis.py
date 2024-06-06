#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedKFold, GridSearchCV, ParameterGrid
from sklearn.inspection import permutation_importance
import multiprocessing

from sklearn.svm import SVC         # Support Vector Clasification
from sklearn.metrics import f1_score, jaccard_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

import os


# In[2]:

# Ruta al archivo Excel (o la ruta correcta a tu conjunto de datos)
file_path = os.path.join(os.getcwd(), 'Data.xlsx')

# Imprime la ruta del archivo para verificar
print(f"Looking for the file at: {file_path}")

# Verifica si el archivo existe
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")


# Lee el archivo de Excel
Data = pd.read_excel(file_path)

# Ahora puedes trabajar con el DataFrame en tu código
print(Data.head())
#Data
#Data = pd.read_excel(os.getcwd() + '/data//Data.xlsx')
#Data

c_ingreso = Data['ingreso'].value_counts()
print(c_ingreso)


# In[3]:


# Preprocesamiento de los datos para la distribucion
estudiantes_CEPRE = Data.drop(['estudiante','si_no'], axis=1)
Columnas = estudiantes_CEPRE.columns.to_list()
nColumnas = ['P-1','P-2','P-3','P-4','P-5','P-6','P-7']
dfColumnas = dict(zip(Columnas, nColumnas))
estudiantes_CEPRE.rename(columns=dfColumnas, inplace=True)


# In[4]:


Data_tesis = estudiantes_CEPRE[['P-3','P-2','P-7','P-4','P-1','P-6','P-5']]
Data_tesis


# In[5]:


def Calcular_Promedio(Dt):
    p3,p2,p7,p4,p1,p6,p5 = Dt

    return (p3+p2+p7+p4+p1)-(p6+p5)

def extract_clustered_table(Graphic, gData):
    if Graphic.dendrogram_row is None:
        print("Aparentemete, Las columnas no estan agrupadas")
        return -1
    if Graphic.dendrogram_col is not None:
        new_cols = gData.columns[Graphic.dendrogram_col.reordered_ind]
        new_ind = gData.index[Graphic.dendrogram_row.reordered_ind]
        return gData.loc[new_ind, new_cols]
    else:
        new_ind = gData.index[Graphic.dendrogram_row.reordered_ind]
        return gData.loc[new_ind,:]


# In[84]:


Data_tesis['PROM'] = Data_tesis[['P-3','P-2','P-7','P-4','P-1','P-6','P-5']].apply(Calcular_Promedio, axis=1)
# Data_tesis.sort_values

# Verificamos si hay datos nulos en las variables
Data_tesis.isnull().sum()


# In[7]:


X = np.arange(Data_tesis['PROM'].size)

data = Data_tesis.rolling(150).mean()
fig, ax = plt.subplots(figsize = (16, 4))
sns.lineplot(data=Data_tesis[['P-3','P-2','P-7','P-4','P-1','P-6','P-5']].rolling(30).mean(), palette="tab10", linewidth=1.5).set_title("EVOLUCIÓN DE LAS ENCUESTAS POR PREGUNTAS")
# sns.lineplot(data=estudiantes_CEPRE[['P-1','P-2','P-3','P-4','P-5','P-6','P-7']].rolling(30).mean(), palette="tab10", linewidth=1.5).set_title("EVOLUCIÓN DE LAS ENCUESTAS POR PREGUNTAS")


# In[8]:


data_prom = Data_tesis['PROM'].rolling(80).mean()
fig, ax = plt.subplots(figsize = (16, 4))
sns.lineplot(data=data_prom, palette="tab10", linewidth=1.5).set_title("PROMEDIO DE LAS ENCUESTAS - CALCULADOS EN BASE A LA IMPORTANCIA POR PREGUNTAS")


# In[9]:


t_Data = estudiantes_CEPRE[['P-1','P-2','P-3','P-4','P-5','P-6','P-7']].head(100).transpose()
t_Data = np.asarray(t_Data)

fig, ax = plt.subplots(figsize = (16, 4))
Graph = ax.imshow(t_Data, cmap='RdBu')
cbar = ax.figure.colorbar(Graph, ax = ax)
cbar.ax.set_ylabel("Escala de Likert", rotation = -90, va = "bottom")
plt.show()
# t_Data


# In[65]:


graph_Data = estudiantes_CEPRE[['P-1','P-2','P-3','P-4','P-5','P-6','P-7']].transpose()

# metric: russellrao, rogerstanimoto, sokalmichener, chebyshev, kulsinski, ..., cityblock, minkowski, euclidean, hamming, jaccard, matching, sqeuclidean, yule
metric = ['russellrao', 'rogerstanimoto', 'sokalmichener', 'chebyshev', 'kulsinski', 'cityblock', 'minkowski', 'euclidean', 'hamming', 'jaccard', 'matching', 'sqeuclidean', 'yule']
Grafico = sns.clustermap(graph_Data, metric=metric[3], cmap='vlag_r', figsize=(16, 3), 
               row_cluster=False, dendrogram_ratio=(.1, .2),  cbar_pos=(0, .2, .03, .4))


# In[69]:


ok_Data_Cols = graph_Data.columns[Grafico.dendrogram_col.reordered_ind]
ok_Data = graph_Data[ok_Data_Cols]
ok_Data = ok_Data.transpose()
idx_List = ok_Data.index
ok_Data.reset_index(inplace=True, drop=True)
# ok_Data.reset_index(inplace=True)
ok_Data = ok_Data.transpose()
ok_Data


# In[178]:


# Sumando las columnas de la Matriz
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

Columnas   = ['Muy en desacuerdo', 'En desacuerdo', 'Ni de acuerdo ni en desacuerdo', 'De acuerdo', 'Totalmente de acuerdo']
Colors     = ['brown','red','orange','limegreen','darkgreen']
Percentage = ['{0:.2f}%'.format(cn1/len(sum_Cols)*100),
              '{0:.2f}%'.format(cn2/len(sum_Cols)*100),
              '{0:.2f}%'.format(cn3/len(sum_Cols)*100),
              '{0:.2f}%'.format(cn4/len(sum_Cols)*100),
              '{0:.2f}%'.format(cn5/len(sum_Cols)*100)]

fig, ax = plt.subplots(figsize = (16, 4))
# sns.lineplot(data=[[data_lk_1, data_lk_2]], palette="tab10", linewidth=1.5).set_title("SEGMENTACION DE LA ESCALA DE LIKERT")
# Likert_graph = sns.lineplot(data=[ls_1, ls_2, ls_3], palette="tab10", linewidth=1.5)
sns.lineplot(data=ls_1, color=Colors[0], linewidth=1.5, label=Columnas[0] + ' ' + Percentage[0])
sns.lineplot(data=ls_2, color=Colors[1], linewidth=1.5, label=Columnas[1] + ' ' + Percentage[1])
sns.lineplot(data=ls_3, color=Colors[2], linewidth=1.5, label=Columnas[2] + ' ' + Percentage[2])
sns.lineplot(data=ls_4, color=Colors[3], linewidth=1.5, label=Columnas[3] + ' ' + Percentage[3])
sns.lineplot(data=ls_5, color=Colors[4], linewidth=1.5, label=Columnas[4] + ' ' + Percentage[4])
# for t, l in zip(Likert_graph.legend, Columnas):
#     t.set_text(l)
plt.legend(title="SEGMENTACION DE LA ESCALA DE LIKERT")
plt.show()

sum_Cols.unique()
# Mat_likert
# ls_1
print(cn1 + cn2 + cn3 + cn4 + cn5)
print(min(data_lk_5), max(data_lk_5))
# print(Likert_graph.legend)
Percentage


# In[66]:


Dendogramas = Grafico.dendrogram_col.dendrogram
I = np.array(Dendogramas['icoord'])
D = np.array(Dendogramas['dcoord'])
Dendogramas['color_list'][:5]     # ['C0', 'C0', 'C0', 'C0', 'C0']
Dendogramas['dcoord'][:5]
Dendogramas['icoord'][:5]
# Dendogramas['ivl'][:5]
# Dendogramas['leaves'][:5]



# print(D[0],D[2],D[3],D[4],D[5])

Linkage = Grafico.dendrogram_col.linkage
Linkage


# In[67]:


from collections import defaultdict

cluster_idxs = defaultdict(list)
for c, pi in zip(Dendogramas['color_list'], Dendogramas['icoord']):
    for leg in pi[1:3]:
        i = (leg - 5.0) / 10.0
        if abs(i - int(i)) < 1e-5:
            cluster_idxs[c].append(int(i))

cluster_idxs


# In[13]:


fig, ax = plt.subplots(figsize = (16, 4))
Graph = ax.imshow(ok_Data, cmap='RdBu')
cbar = ax.figure.colorbar(Graph, ax = ax)
cbar.ax.set_ylabel("Escala de Likert", rotation = -90, va="bottom")
plt.show()


# In[61]:


ok_Data.describe()
Data_tesis[['P-3','P-2','P-7','P-4','P-1','P-6','P-5']].describe()


# In[71]:


# Creamos el modelo kMEANS
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

Data_tesis_model = Data_tesis[['P-3','P-2','P-7','P-4','P-1','P-6','P-5']]
Scaler = MinMaxScaler().fit(Data_tesis_model.values)

Data_tesis_model = pd.DataFrame(Scaler.transform(Data_tesis_model.values), columns=['P-3','P-2','P-7','P-4','P-1','P-6','P-5'])

kmeans = KMeans(n_clusters=5).fit(Data_tesis_model.values)
Data_tesis_model['Cluster'] = kmeans.labels_

# Mostrando los centroides y su inercia (Clasifica que tan bueno son las agrupaciones)
print(kmeans.cluster_centers_, kmeans.inertia_)
Data_tesis_model


# In[75]:


plt.figure(figsize=(8,6), dpi=100)
colores = ['red','b','orange','b','purple','pink','brown']
print(kmeans.n_clusters)
for cluster in range(kmeans.n_clusters):
    plt.scatter(Data_tesis_model[Data_tesis_model['Cluster']==cluster]['P-3'],
                Data_tesis_model[Data_tesis_model['Cluster']==cluster]['P-2'],
                # Data_tesis_model[Data_tesis_model['Cluster']==cluster]['P-7'],
                # Data_tesis_model[Data_tesis_model['Cluster']==cluster]['P-4'],
                # Data_tesis_model[Data_tesis_model['Cluster']==cluster]['P-1'],
                # Data_tesis_model[Data_tesis_model['Cluster']==cluster]['P-6'],
                # Data_tesis_model[Data_tesis_model['Cluster']==cluster]['P-5'],
                marker='o', s=50, color=colores[cluster], alpha=.5)
    plt.scatter(kmeans.cluster_centers_[cluster][0],
                kmeans.cluster_centers_[cluster][1],
                # kmeans.cluster_centers_[cluster][2],
                # kmeans.cluster_centers_[cluster][3],
                # kmeans.cluster_centers_[cluster][4],
                marker='P', s=100, color=colores[cluster])
    
plt.title('NIVEL DE SATISFACCIÓN')
plt.show()

def generate_plots(data):
    # Ejemplo de generación de gráficos
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20)
    plt.xlabel('Valores')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de Datos')
    plt.savefig('histogram.png')  # Guarda el gráfico como un archivo de imag