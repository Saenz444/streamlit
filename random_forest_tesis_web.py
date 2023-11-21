import streamlit as st
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

 

#LA DATA DEL MODELO
Data = pd.read_excel("Data.xlsx")


# Preprocesamiento de los datos para la distribucion
estudiantes_CEPRE = Data.drop(['estudiante','si_no'], axis=1)
Columnas = estudiantes_CEPRE.columns.to_list()
nColumnas = ['P-1','P-2','P-3','P-4','P-5','P-6','P-7']
dfColumnas = dict(zip(Columnas, nColumnas))
estudiantes_CEPRE.rename(columns=dfColumnas, inplace=True)
   
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
#STREAMLIT - WEB
def main():
#TITULO GENERAL DE LA PAGINA
    st.title('SATISFACCIÓN ACADÉMICA')   
    st.markdown("Utilizaremos modelos como Random Forest, SVM, KNN y Rgresion Lineal.")
    
    
    st.image("machine.png", caption="Descripción de la imagen", use_column_width=True)
    
    
#BARRA IZQUIERDA TITULO GENERAL
    st.sidebar.header('PARÁMETROS DE ENTRADA') 
    
    
#DARLE CLICK PARA MOSTRAR EL DATA FRAMA
    st.sidebar.subheader("Mostrar la DATA completa")
    if st.sidebar.checkbox('Data'):
        st.write(Data)
 
    
    #ESCOGER CLASIFICACION DE MIS MODELOS
    st.sidebar.subheader("Escoger Clasificación")
    option = ['Ninguno','Random Forest','Support Vector Machine (SVM)', 'KNN', 'Regresion Lineal']
    classifier = st.sidebar.selectbox("Escoger el Modelo",option)
    
    
   
###############################################################################################################
    if classifier is None:
        st.image("machine.png", caption="Descripción de la imagen", use_column_width=True)
    else:
        if classifier != 'Ninguno':
   
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

    #SI ES SVM
    
            elif classifier == 'Support Vector Machine (SVM)':
                st.header("Modelo SVM")
                st.write("Accuracy: {:.2f}%".format(100 * accuracy_rf))
                cm_svm = confusion_matrix(y_test_svm, y_pred)
        #st.write('Confusion matrix: ', cm_svm)
    
    # Mostrar la matriz de confusión como un gráfico
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

    
    ###########################################################################################################
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
                

        
            else:
    # Si no se selecciona un gráfico
                st.write("Selecciona un gráfico en el menú desplegable.")
        
        
   ######################################################################################################
    
# Llamada a la función para mostrar el gráfico de importancia de preguntas
    if st.sidebar.checkbox('Grafico de barras'):
        st.header("Determinación de Preguntas")
        mostrar_grafico_importancia_preguntas(modelo_final, X_train, y_train)

if __name__=='__main__':
    main()
    