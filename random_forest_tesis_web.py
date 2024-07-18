import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.inspection import permutation_importance
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import f1_score, jaccard_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

# Función para mostrar el gráfico de importancia de características para RandomForest
def mostrar_grafico_importancia_preguntas(nuevo_modelo, X_train, y_train):
    importancia = permutation_importance(
        estimator=nuevo_modelo,
        X=X_train,
        y=y_train,
        n_repeats=5,
        scoring='accuracy',
        n_jobs=multiprocessing.cpu_count() - 1,
        random_state=123
    )
    
    df_importancia = pd.DataFrame(
        {k: importancia[k] for k in ['importances_mean', 'importances_std']}
    )
    df_importancia['predictor'] = X_train.columns

    fig, ax = plt.subplots(figsize=(5, 6))
    color = ['y','y','y','y','y','g','g']
    fig, ax = plt.subplots(figsize=(5, 6))
    df_importancia = df_importancia.sort_values('importances_mean', ascending=True)
    ax.barh(df_importancia['predictor'],
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
    st.pyplot(fig)

def main():
    st.title('SATISFACCIÓN ACADÉMICA')
    st.markdown("Utilizaremos algoritmos y medición de grado satisfacción")

    if 'show_image' not in st.session_state:
       st.session_state.show_image = True

    if st.session_state.show_image:
        st.image("machine.jpg", caption=" ", use_column_width=True)
    

    st.sidebar.header('PARÁMETROS DE ENTRADA')

    archivo1 = 'Datavf_Modificado.xlsx'
    Data1 = pd.read_excel(archivo1)
    
    estudiantes_CEPRE = Data1.drop(['Satisfaccion', 'PROM'], axis=1)
    Columnas = estudiantes_CEPRE.columns.to_list()
    nColumnas = ['P-1', 'P-2', 'P-3', 'P-4', 'P-5', 'P-6', 'P-7']
    dfColumnas = dict(zip(Columnas, nColumnas))
    estudiantes_CEPRE.rename(columns=dfColumnas, inplace=True)

    # Ajuste del modelo y optimización de hiperparámetros
    X_train, X_test, y_train, y_test = train_test_split(
        estudiantes_CEPRE,
        Data1['Satisfaccion'],  # Utilizar la columna 'Satisfaccion' como objetivo
        random_state=123
    )
    st.write("Tamaño del conjunto de entrenamiento: ", len(X_train))
    st.write("Tamaño del conjunto de prueba: ", len(X_test))

    param_grid = {'n_estimators': [150], 'max_features': [3, 5, 7], 'max_depth': [None, 3, 10, 20], 'criterion': ['gini', 'entropy']}
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=123),
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=multiprocessing.cpu_count() - 1,
        cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=123),
        refit=True,
        verbose=0,
        return_train_score=True
    )
    grid.fit(X=X_train, y=y_train)

    modelo_final = grid.best_estimator_
    predicciones = modelo_final.predict(X=X_test)
    mat_confusion = confusion_matrix(y_true=y_test, y_pred=predicciones)
    accuracy = accuracy_score(y_true=y_test, y_pred=predicciones, normalize=True)

    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
        estudiantes_CEPRE,
        Data1['Satisfaccion'],
        random_state=5
    )
    modelo_svm = SVC(kernel='rbf', C=1)
    modelo_svm.fit(X_train_svm, y_train_svm)
    accuracy_svm = modelo_svm.score(X_test_svm, y_test_svm)
    y_pred_svm = modelo_svm.predict(X_test_svm)

    modelo_KNN = KNeighborsClassifier(n_neighbors=3)
    modelo_KNN.fit(X_train_svm, y_train_svm)
    accuracy_knn = modelo_KNN.score(X_test_svm, y_test_svm)
    y_pred_knn = modelo_KNN.predict(X_test_svm)

    if st.sidebar.checkbox('Satisfacción - Likert'):
        st.write(Data1)

    option1 = ['Ninguno', 'Random Forest', 'Support Vector Machine (SVM)', 'KNN']
    classifier = st.sidebar.selectbox("Escoger el Modelo", option1, key="unique_key_1")
    
    image_container = st.empty()  # Contenedor vacío para la imagen

    if classifier == 'Ninguno':
        st.session_state.show_image = True
    else:
        st.session_state.show_image = False

    if classifier == 'Random Forest':
        st.header("Modelo Random Forest")
        st.write("Accuracy: {:.2f}%".format(100 * accuracy))

        st.subheader("Matriz de Confusión")
        conf_matrix_rf = confusion_matrix(y_test, grid.predict(X_test))

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax)
        plt.title("Matriz de Confusión - Random Forest")
        plt.xlabel("Predicciones")
        plt.ylabel("Valores reales")
        st.pyplot(fig)
        plt.close(fig)

    elif classifier == 'Support Vector Machine (SVM)':
        st.header("Modelo SVM")
        st.write("Accuracy: {:.2f}%".format(100 * accuracy_svm))
        cm_svm = confusion_matrix(y_test_svm, y_pred_svm)

        st.subheader('Matriz de Confusión')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_svm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax)
        plt.title("Matriz de Confusión - SVM")
        plt.xlabel("Predicciones")
        plt.ylabel("Valores reales")
        st.pyplot(fig)
        plt.close(fig)

    elif classifier == 'KNN':
        st.header("Modelo KNN")
        st.write("Accuracy: {:.2f}%".format(100 * accuracy_knn))
        cm_knn = confusion_matrix(y_test_svm, y_pred_knn)

        st.subheader('Matriz de Confusión')
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_knn, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax)
        plt.title("Matriz de Confusión - KNN")
        plt.xlabel("Predicciones")
        plt.ylabel("Valores reales")
        st.pyplot(fig)
        plt.close(fig)

    if st.sidebar.checkbox('Grafico de barras'):
        st.header("Influencia de Satisfacción de las Preguntas")
        nuevo_modelo = RandomForestClassifier(random_state=123)
        nuevo_modelo.fit(X_train, y_train)
        mostrar_grafico_importancia_preguntas(nuevo_modelo, X_train, y_train)

    st.sidebar.subheader("Medición de Satisfacción Académica")
    option2 = ['Ninguno', 'Metrica', 'Satisfaccion']
    classifier1 = st.sidebar.selectbox("Escoger el Modelo", option2, key="unique_key_2")
    
    image_container = st.empty()  # Contenedor vacío para la imagen

    if classifier1 == 'Ninguno':
        st.session_state.show_image = True
    else:
        st.session_state.show_image = False

    if classifier1 == 'Metrica':
        st.header("Cluster")
        t_Data = estudiantes_CEPRE[['P-1', 'P-2', 'P-3', 'P-4', 'P-5', 'P-6', 'P-7']].head(100).transpose()
        t_Data = np.asarray(t_Data)
        fig, ax = plt.subplots(figsize=(16, 4))
        Graph = ax.imshow(t_Data, cmap='RdBu')
        cbar = ax.figure.colorbar(Graph, ax=ax)
        cbar.ax.set_ylabel("Escala de Likert", rotation=-90, va="bottom")
        st.pyplot(fig)

    elif classifier1 == 'Satisfaccion':
        st.header("Grado de Satisfacción")
        graph_Data = estudiantes_CEPRE[['P-1', 'P-2', 'P-3', 'P-4', 'P-5', 'P-6', 'P-7']].transpose()
        metric = ['russellrao', 'rogerstanimoto', 'sokalmichener', 'chebyshev', 'kulsinski', 'cityblock', 'minkowski', 'euclidean', 'hamming', 'jaccard', 'matching', 'sqeuclidean', 'yule']
        Grafico = sns.clustermap(graph_Data, metric=metric[3], cmap='vlag_r', figsize=(16, 3), row_cluster=False, dendrogram_ratio=(.1, .2), cbar_pos=(0, .2, .03, .4))
        st.pyplot(Grafico)

        ok_Data_Cols = graph_Data.columns[Grafico.dendrogram_col.reordered_ind]
        ok_Data = graph_Data[ok_Data_Cols]
        ok_Data = ok_Data.transpose()
        idx_List = ok_Data.index
        ok_Data.reset_index(inplace=True, drop=True)
        ok_Data = ok_Data.transpose()

        sum_Cols = ok_Data.apply(lambda x: sum(ok_Data[x]))
        Max, Min = max(sum_Cols), min(sum_Cols)

        Qt_1 = [7.0,  12.6]
        Qt_2 = [12.6, 18.2]
        Qt_3 = [18.2, 23.8]
        Qt_4 = [23.8, 29.4]
        Qt_5 = [29.4, 35.0]

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

        Columnas = ['Completamente insatisfecho', 'Insatisfecho', 'Neutrales', 'Satisfechos', 'Completamente satisfecho']
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
        plt.legend(title="SEGMENTACION DE LA ESCALA DE LIKERT")
        st.pyplot(fig)

if __name__ == '__main__':
    main()
    

