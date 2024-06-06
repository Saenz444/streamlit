import streamlit as st

from analysis import generate_plots
from sklearn import metrics
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, mean_squared_error
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedKFold, GridSearchCV, ParameterGrid
#from sklearn.model_selection import train_test_splitfrom 
#from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.inspection import permutation_importance
import multiprocessing
import matplotlib.pyplot as plt


from sklearn.svm import SVC         # Support Vector Clasification
from sklearn.metrics import f1_score, jaccard_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

import os

    # Función para generar un gráfico de líneas
def line_chart():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    st.pyplot(fig)

# Función para generar un gráfico de dispersión
def scatter_chart():
    x = np.random.rand(100)
    y = np.random.rand(100)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    st.pyplot(fig)

#LA DATA DEL MODELO
Data = pd.read_excel("Data.xlsx")


# Preprocesamiento de los datos para la distribucion
estudiantes_CEPRE = Data.drop(['estudiante','si_no'], axis=1)
Columnas = estudiantes_CEPRE.columns.to_list()
nColumnas = ['P-1','P-2','P-3','P-4','P-5','P-6','P-7']
dfColumnas = dict(zip(Columnas, nColumnas))
estudiantes_CEPRE.rename(columns=dfColumnas, inplace=True)

######################################################################################################
#ANALYSIS
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
#Data_tesis


# In[5]:
graph_Data = estudiantes_CEPRE[['P-1','P-2','P-3','P-4','P-5','P-6','P-7']].transpose()

# metric: russellrao, rogerstanimoto, sokalmichener, chebyshev, kulsinski, ..., cityblock, minkowski, euclidean, hamming, jaccard, matching, sqeuclidean, yule
metric = ['russellrao', 'rogerstanimoto', 'sokalmichener', 'chebyshev', 'kulsinski', 'cityblock', 'minkowski', 'euclidean', 'hamming', 'jaccard', 'matching', 'sqeuclidean', 'yule']
Grafico = sns.clustermap(graph_Data, metric=metric[3], cmap='vlag_r', figsize=(16, 3), 
               row_cluster=False, dendrogram_ratio=(.1, .2),  cbar_pos=(0, .2, .03, .4))

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


Data_tesis = estudiantes_CEPRE[['P-3', 'P-2', 'P-7', 'P-4', 'P-1', 'P-6', 'P-5']]

Data_tesis['PROM'] = Data_tesis[['P-3', 'P-2', 'P-7', 'P-4', 'P-1', 'P-6', 'P-5']].apply(Calcular_Promedio, axis=1)




###########################################################################################################   
   # Ajuste del modelo y optimización de hiperparámetros 
X_train, X_test, y_train, y_test = train_test_split(
estudiantes_CEPRE.drop(columns = 'ingreso'),estudiantes_CEPRE['ingreso'],random_state = 123)
    
# Grid Search basado en validación cruzada
# =================================================================================================
param_grid = {'n_estimators': [150], #la data que utiliza
                    'max_features': [3, 5, 7], #las preguntas
                    'max_depth'   : [None, 3, 10, 20],
                    'criterion'   : ['gini', 'entropy']
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
    
# with tf.device('/device:GPU:0'):
grid.fit(X = X_train, y = y_train)

#Calcular la accuracy
modelo_final = grid.best_estimator_
predicciones = modelo_final.predict(X = X_test) 
    
#Medir el rendimmiento del modelo
mat_confusion = confusion_matrix(
        y_true    = y_test,
        y_pred    = predicciones
    )
    
#Calcular la precision
accuracy = accuracy_score(
        y_true    = y_test,
        y_pred    = predicciones,
        normalize = True
    )

###################################################################################################################
#ENTRENO DE SVM
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
estudiantes_CEPRE.drop(columns = 'ingreso'),
estudiantes_CEPRE['ingreso'],
random_state = 5    # 4 rondas
    )

estudiantes_CEPRE.drop(columns = 'ingreso')  

modelo_svm = SVC(kernel='rbf', C=1)      # rbf, linear, poly, sigmoid
modelo_svm.fit(X_train_svm, y_train_svm)
    
accuracy_rf = modelo_svm.score(X_test_svm, y_test_svm)
y_pred = modelo_svm.predict(X_test_svm)

######################################################################################################################
#ENTRENO DE KNN
modelo_KNN = KNeighborsClassifier(n_neighbors=3)
modelo_KNN.fit(X_train_svm, y_train_svm)
m= modelo_KNN.fit(X_train_svm, y_train_svm)
    
acurracy_k = m.score(X_train_svm, y_train_svm)
predict_KNN = m.predict(X_test_svm)
    

#######################################################################################################################
#ENTRENO RL - SCIKIT-LEARN
matriz_unos = np.array(np.ones((X_train_svm.shape[0], 1)))
matriz_X = np.append(matriz_unos, X_train_svm.values, axis=1)
    
matriz_X_Transpuesta = matriz_X.transpose()
matriz_X_inversa = np.linalg.inv(np.matmul(matriz_X_Transpuesta, matriz_X))

matriz_X_TranspuestaY = np.matmul(matriz_X_Transpuesta, y_train_svm)
    
matriz_Beta = np.matmul(matriz_X_inversa, matriz_X_TranspuestaY)
# Prueba inicial
#                          P1, P2, P3, P4, P5, P6, P7
datos_Prueba = np.array([[1, 1,  5,  5,  5,  1,  1, 5], [1, 4,3,4,5,2,3,5]])
media_Ingresa = np.dot(datos_Prueba, matriz_Beta)

    
# Utilizando la librería Scikit-Learn
modelo_RL = LinearRegression()
modelo_RL.fit(X_train_svm, y_train_svm)
coeficientes_RL = modelo_RL.coef_
interceptor_RL = modelo_RL.intercept_

# Coeficiente de determinarion R2
# Coeficiente RL mas hacia 1, el modelo es adecuado de lo contrario no es el adecuado
#                y si es negativo, se considera CERO
print('Coeficiente de determinacion RL:', modelo_RL.score(X_test_svm, y_test_svm))

prediccion_RL = modelo_RL.predict(X_test_svm)
print('RMSE:', (np.sqrt(mean_squared_error(y_test_svm, prediccion_RL))))
#########################################################################

    
 ###########################################################################################   
 # Función para mostrar el gráfico de importancia de características para RandomForest
def mostrar_grafico_importancia_preguntas(modelo, X_train, y_train):
    # Importancia por permutación
    importancia = permutation_importance(
        estimator=modelo,
        X=X_train,
        y=y_train,
        n_repeats=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=multiprocessing.cpu_count() - 1,
        random_state=123
    ) 
        # Almacenar los resultados (media y desviación) en un DataFrame
    df_importancia = pd.DataFrame({
        'importances_mean': importancia['importances_mean'],
        'importances_std': importancia['importances_std'],
        'predictor': X_train.columns
    })

    # Graficar los resultados
    color = ['r', 'r', 'r', 'y', 'g', 'g', 'g']
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

    # Mostrar la figura en Streamlit
    st.pyplot(fig)
######################################################################################


######################################################################################################################
#STREAMLIT - WEB
def main():

#TITULO GENERAL DE LA PAGINA
    st.title('SATISFACCIÓN ACADÉMICA')   
    st.markdown("Utilizaremos algoritmos y medición de grado satisfacción")

    
# Imagen inicial
    if 'show_image' not in st.session_state:
        st.session_state.show_image = True

    if  st.session_state.show_image:
        st.image("machine.png", caption=" ", use_column_width=True)

#BARRA IZQUIERDA TITULO GENERAL
    st.sidebar.header('PARÁMETROS DE ENTRADA') 
    
    
#DARLE CLICK PARA MOSTRAR EL DATA FRAMA
    st.sidebar.subheader("Mostrar la DATA completa")
    if st.sidebar.checkbox('Data'):
        st.write(Data)
  
    st.sidebar.subheader("Mostrar la DATA completa")
    if st.sidebar.checkbox('Satisfacción - Likert'):
        st.write(Data_tesis)
    
    #ESCOGER CLASIFICACION DE MIS MODELOS
    st.sidebar.subheader("Escoger Clasificación")
    option1 = ['Ninguno','Random Forest','Support Vector Machine (SVM)', 'KNN', 'Regresion Lineal']
    classifier = st.sidebar.selectbox("Escoger el Modelo",option1, key="unique_key_1")
    
###############################################################################################################
    if classifier == 'Ninguno':
        st.session_state.show_image = True
    else:
        st.session_state.show_image = False
   
###############################################################################################################
    
   
#SI ELIJO RANDOMFOREST   

        if classifier == 'Random Forest':
            st.header("Modelo Random Forest")
            st.write("Accuracy: {:.2f}%".format(100 * accuracy))
        
        
    # Matriz de confusión y gráfico para Random Forest
            st.subheader("Matriz de Confusión")
            conf_matrix_rf = confusion_matrix(y_test, grid.predict(X_test))
        #st.write("Matriz de Confusión:", conf_matrix_rf )
    

# Código para mostrar el gráfico de la matriz de confusión con seaborn
        #st.write("Gráfico de Matriz de Confusión:")
       # disp_rf = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf, display_labels=np.unique(y_test))

    # Configuración del gráfico
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax)

    # Ajustes adicionales (puedes personalizar según tus preferencias)
            plt.title("Matriz de Confusión - Random Forest")
            plt.xlabel("Predicciones")
            plt.ylabel("Valores reales")

    # Mostrar el gráfico en Streamlit
            st.pyplot(fig)
            plt.close(fig)

#########################################################################################
        elif classifier == 'Support Vector Machine (SVM)':
            st.header("Modelo SVM")
            st.write("Accuracy: {:.2f}%".format(100 * accuracy_rf))
            cm_svm = confusion_matrix(y_test_svm, y_pred)
        #st.write('Confusion matrix: ', cm_svm)
    
            st.subheader('Confusion Matrix')
     
    # Código para mostrar el gráfico de la matriz de confusión con seaborn
        #st.write("Gráfico de Matriz de Confusión:")
        #disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=np.unique(y_test_svm))
    
    # Configuración del gráfico
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_svm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax)

    # Ajustes adicionales (puedes personalizar según tus preferencias)
            plt.title("Matriz de Confusión - SVM")
            plt.xlabel("Predicciones")
            plt.ylabel("Valores reales")

    # Mostrar el gráfico en Streamlit
            st.pyplot(fig)
            plt.close(fig)

        elif classifier == 'KNN':
            st.header("Modelo KNN")
            st.write("Accuracy: {:.2f}%".format(100 * acurracy_k))
            matriz_confussion = confusion_matrix(y_test_svm, predict_KNN)
        #st.write('Confusion matrix: ', matriz_confussion)
    
    # Mostrar la matriz de confusión como un gráfico
            st.subheader('Confusion Matrix')
     
    # Código para mostrar el gráfico de la matriz de confusión con seaborn
       # st.write("Gráfico de Matriz de Confusión:")
        #disp_knn = ConfusionMatrixDisplay(confusion_matrix=matriz_confussion, display_labels=np.unique(y_test_svm))
    
    # Configuración del gráfico
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(matriz_confussion, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax)

    # Ajustes adicionales (puedes personalizar según tus preferencias)
            plt.title("Matriz de Confusión - KNN")
            plt.xlabel("Predicciones")
            plt.ylabel("Valores reales")

    # Mostrar el gráfico en Streamlit
            st.pyplot(fig)
            plt.close(fig)
    
    ##############################################################################################################
    # regresión lineal
    
        elif classifier == 'Regresion Lineal':
            st.header("Modelo Regresión Lineal")
            st.write("Coeficiente de Determinación RL : {:.2f}%".format(100 * modelo_RL.score(X_test_svm, y_test_svm)))
            st.write(f'RMSE: {np.sqrt(mean_squared_error(y_test_svm, prediccion_RL))}')

        
    #GRAFICO  
    
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test_svm, prediccion_RL)
            plt.plot([min(y_test_svm), max(y_test_svm)], [min(y_test_svm), max(y_test_svm)], linestyle='--', color='red', linewidth=2)
            plt.xlabel('Valores Reales')
            plt.ylabel('Predicciones')
            plt.title('Comparación entre Valores Reales y Predicciones (Regresión Lineal)')
            st.pyplot(plt)
                

        
            # Volver a mostrar la imagen si no se ha seleccionado ningún modelo
        if st.session_state.show_image:
            st.image("machine.png", caption="Descripción de la imagen", use_column_width=True)
###############################################################################################################
    #MEDICION DE SATISFACCION
    st.sidebar.subheader("Medición de Satisfacción Académica")
    option2 = ['Ninguno','Metrica','Satisfaccion']
    classifier1 = st.sidebar.selectbox("Escoger el Modelo",option2, key="unique_key_2")
    

    if classifier1 == 'Ninguno':
        st.session_state.show_image = True
    else:
        st.session_state.show_image = False
        
#SI ELIJO RANDOMFOREST   
        if classifier1 == 'Metrica':
            st.header("Cluster")
            t_Data = estudiantes_CEPRE[['P-1','P-2','P-3','P-4','P-5','P-6','P-7']].head(100).transpose()
            t_Data = np.asarray(t_Data)

            fig, ax = plt.subplots(figsize = (16, 4))
            Graph = ax.imshow(t_Data, cmap='RdBu')
            cbar = ax.figure.colorbar(Graph, ax = ax)
            cbar.ax.set_ylabel("Escala de Likert", rotation = -90, va = "bottom")
                #plt.show()
            graph_Data = estudiantes_CEPRE[['P-1','P-2','P-3','P-4','P-5','P-6','P-7']].transpose()

# metric: russellrao, rogerstanimoto, sokalmichener, chebyshev, kulsinski, ..., cityblock, minkowski, euclidean, hamming, jaccard, matching, sqeuclidean, yule
            metric = ['russellrao', 'rogerstanimoto', 'sokalmichener', 'chebyshev', 'kulsinski', 'cityblock', 'minkowski', 'euclidean', 'hamming', 'jaccard', 'matching', 'sqeuclidean', 'yule']
            Grafico = sns.clustermap(graph_Data, metric=metric[3], cmap='vlag_r', figsize=(16, 3), 
                            row_cluster=False, dendrogram_ratio=(.1, .2),  cbar_pos=(0, .2, .03, .4))
            st.pyplot(Grafico)
                
# t_Data              
                
        elif classifier1 == 'Satisfaccion':
            st.header("Grado de Satisfacción")
            graph_Data = estudiantes_CEPRE[['P-1', 'P-2', 'P-3', 'P-4', 'P-5', 'P-6', 'P-7']].transpose()

# metric: russellrao, rogerstanimoto, sokalmichener, chebyshev, kulsinski, ..., cityblock, minkowski, euclidean, hamming, jaccard, matching, sqeuclidean, yule
            metric = ['russellrao', 'rogerstanimoto', 'sokalmichener', 'chebyshev', 'kulsinski', 'cityblock', 'minkowski', 'euclidean', 'hamming', 'jaccard', 'matching', 'sqeuclidean', 'yule']
            Grafico = sns.clustermap(graph_Data, metric=metric[3], cmap='vlag_r', figsize=(16, 3), row_cluster=False, dendrogram_ratio=(.1, .2), cbar_pos=(0, .2, .03, .4))

            ok_Data_Cols = graph_Data.columns[Grafico.dendrogram_col.reordered_ind]
            ok_Data = graph_Data[ok_Data_Cols]
            ok_Data = ok_Data.transpose()
            idx_List = ok_Data.index
            ok_Data.reset_index(inplace=True, drop=True)
            ok_Data = ok_Data.transpose()

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

            Columnas   = ['Muy insatisfecho', 'Poco satisfecho', 'Ni satisfecho ni en instisfecho', 'Satisfecho', 'Totalmente satisfecho']
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
            st.pyplot(fig)
        
# Volver a mostrar la imagen si no se ha seleccionado ningún modelo
        if st.session_state.show_image:
            st.image("machine.png", caption="Descripción de la imagen", use_column_width=True)
        
        
        
            
   ######################################################################################################
    
# Llamada a la función para mostrar el gráfico de importancia de preguntas
    if st.sidebar.checkbox('Grafico de barras'):
        st.header("Influencia de Satisfacción de las Preguntas")
        mostrar_grafico_importancia_preguntas(modelo_final, X_train, y_train)
        
        

if __name__=='__main__':
    main()
    