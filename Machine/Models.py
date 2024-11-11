from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np

class Models:
    def __init__(self,df):
        self.df = df
        self.df_kmeans = None  # Guardará el estado del DataFrame después de K-means
        self.df_pca = None     # Guardará el estado del DataFrame después de PCA
        self.kmeans_model = None
        self.pca_model = None
        
        
    def apply_kmeans(self, n_clusters=5, random_state=42, use_pca=False, test_size=0.3):
        """
        Aplica K-means al DataFrame original o al DataFrame transformado por PCA,
        dividiendo los datos en conjuntos de entrenamiento y prueba, y calcula
        estadísticas de resultados.

        Parameters:
        - n_clusters (int): Número de clusters para K-means.
        - random_state (int): Semilla para asegurar reproducibilidad.
        - use_pca (bool): Si es True, aplica K-means al DataFrame de PCA; si es False, lo aplica al DataFrame original.
        - test_size (float): Proporción de los datos para el conjunto de prueba (entre 0 y 1).

        Returns:
        - tuple: Una tupla con el DataFrame con los clusters asignados y un diccionario de métricas.
        """
        # Seleccionar los datos y escalar si es necesario
        if use_pca and self.df_pca is not None:
            data = self.df_pca.copy()  # Usar una copia del DataFrame transformado por PCA
        else:
            data = self.df.select_dtypes(include=['float64', 'int64']).fillna(self.df.mean())
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            data = pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(data.shape[1])])

        # Dividir en conjunto de entrenamiento y prueba
        X_train, X_test = train_test_split(data, test_size=test_size, random_state=random_state)

        # Aplicar K-means en el conjunto de entrenamiento
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        train_clusters = self.kmeans_model.fit_predict(X_train)

        # Predecir los clusters en el conjunto de prueba
        test_clusters = self.kmeans_model.predict(X_test)

        # Calcular métricas en el conjunto de entrenamiento
        train_silhouette = silhouette_score(X_train, train_clusters)
        train_inertia = self.kmeans_model.inertia_

        # Calcular métricas en el conjunto de prueba
        test_silhouette = silhouette_score(X_test, test_clusters)
        test_inertia = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X_test).inertia_


        results = pd.DataFrame({
            'Silhouette': [train_silhouette, test_silhouette],
            'Inertia': [train_inertia, test_inertia]
        }, index=['Train', 'Test'])
                
        # Asignar los clusters al DataFrame seleccionado (PCA o escalado)
        if use_pca and self.df_pca is not None:
            data['Cluster'] = self.kmeans_model.predict(self.df_pca)
            self.df_pca['Cluster'] = data['Cluster']
        else:
            data['Cluster'] = self.kmeans_model.predict(data)
            self.df_kmeans = data

        # Retornar el DataFrame con los clusters y el diccionario de métricas
        return data, results
        
   
    def eval_categoricas_objetivo(self, columnas, variable_objetivo):
        """
        Evalúa la relación entre variables categóricas y una variable objetivo mediante un análisis ANOVA.
    
        Este método convierte columnas categóricas especificadas en variables numéricas mediante codificación de etiquetas (Label Encoding) y luego 
        realiza un análisis ANOVA para evaluar la asociación entre cada columna categórica y la variable objetivo. 
        El resultado es un DataFrame que contiene las estadísticas F y los valores p correspondientes para cada columna categórica.
    
        Parámetros
        ----------
        columnas : list of str
            Lista de nombres de columnas categóricas en `self.df` que se desean evaluar.
            
        variable_objetivo : str
            Nombre de la columna en `self.df` que representa la variable objetivo para el análisis ANOVA.
    
        Retorna
        -------
        pd.DataFrame
            Un DataFrame que muestra la estadística F y el valor p de cada columna categórica en relación con la variable objetivo.
            Las columnas del DataFrame resultante son:
                - 'F-Statistic': Valor F del análisis ANOVA.
                - 'P-Valor': Valor p asociado al análisis, indicando la significancia estadística.
    
        Ejemplo de uso
        --------------
        # Crear instancia del modelo con un DataFrame
        model = Models(df)
    
        # Evaluar columnas categóricas respecto a la variable objetivo
        resultado = model.eval_categoricas_objetivo(columnas=['categoria1', 'categoria2'], variable_objetivo='objetivo')
    
        # Visualizar resultados
        print(resultado)
        
        Notas
        -----
        - Las columnas categóricas deben ser de tipo `object`. Las que ya están codificadas numéricamente serán ignoradas.
        - En caso de que una columna contenga valores no numéricos después de la codificación, el método emitirá un mensaje de error y omitirá dicha columna en el análisis ANOVA.
        """
        # Crear una copia del DataFrame original y diccionario para almacenar encoders
        data_encoders = self.df.copy()
        label_encoders = {}
    
        # Codificar cada columna categórica
        for col in columnas:
            if data_encoders[col].dtype == 'object':  # Verificar que sea categórica
                le = LabelEncoder()
                data_encoders[col] = le.fit_transform(data_encoders[col])
                label_encoders[col] = le  # Guardar cada encoder para referencia futura
            else:
                print(f"Advertencia: La columna {col} ya está codificada o no es categórica.")
    
        # Realizar el análisis ANOVA
        anova_resultado = {}
        target = data_encoders[variable_objetivo]
        
        for col in columnas:
            # Asegurarse de que los datos sean numéricos para cada grupo
            grupos = [target[data_encoders[col] == val] for val in data_encoders[col].unique()]
            
            # Verificar que cada grupo contenga solo valores numéricos
            if all(grupo.dtype.kind in 'iuf' for grupo in grupos):  # Tipos int, unsigned int, o float
                f_stat, p_val = f_oneway(*grupos)
                anova_resultado[col] = {'F-Statistic': f_stat, 'P-Valor': p_val}
            else:
                print(f"Error: La columna {col} contiene valores no numéricos después de la codificación.")
    
        # Convertir los resultados a un DataFrame y transponerlo para facilitar la lectura
        resultado = pd.DataFrame(anova_resultado).T
        resultado = resultado.applymap(lambda x: f"{x:.2f}")
        return resultado
            
    def informacion_mutua(self,columnas,variable_objetivo):

        """
        Calcula la información mutua entre un conjunto de columnas categóricas y una variable objetivo.
        
        Este método convierte las columnas especificadas y la variable objetivo en etiquetas numéricas mediante Label Encoding,
        luego calcula la información mutua entre cada columna y la variable objetivo, devolviendo un DataFrame ordenado
        según la información mutua en orden descendente.
        
        Parámetros
        ----------
        columnas : list of str
            Lista de nombres de columnas en `self.df` para las cuales se calculará la información mutua respecto a la variable objetivo.
        
        variable_objetivo : str
            Nombre de la columna en `self.df` que representa la variable objetivo para el cálculo de información mutua.
        
        Retorna
        -------
        pd.DataFrame
            Un DataFrame que contiene la información mutua de cada columna con respecto a la variable objetivo.
            La columna 'Información Mutua' representa el valor de información mutua de cada característica, ordenada de mayor a menor.
        
        Ejemplo de uso
        --------------
        # Crear instancia del modelo con un DataFrame
        model = Models(df)
        
        # Calcular la información mutua respecto a una variable objetivo
        resultados = model.informacion_mutua(columnas=['categoria1', 'categoria2'], variable_objetivo='objetivo')
        
        # Visualizar resultados
        print(resultados)
        
        Notas
        -----
        - Todas las columnas en `columnas` y `variable_objetivo` deben ser de tipo categórico.
        - La información mutua es una medida de dependencia que evalúa cuánta información comparten dos variables, útil para selección de características en modelos de clasificación.
        """
        
        df_info_mutua = self.df.copy()
        
        for col in columnas + [variable_objetivo]:
            le = LabelEncoder()
            df_info_mutua[col] = le.fit_transform(df_info_mutua[col])
            
        X = df_info_mutua[columnas]
        y = df_info_mutua[variable_objetivo]
        
        info_mutua = mutual_info_classif(X, y, discrete_features=True)
        resultados = pd.DataFrame(info_mutua, index=columnas, columns=['Información Mutua'])
        return resultados.sort_values(by='Información Mutua', ascending=False) 

    def apply_pca(self, n_components=2):
        """
        Aplica PCA al DataFrame y almacena el resultado en un DataFrame separado.
        
        Parameters:
        - n_components (int): Número de componentes principales a conservar.
        
        Returns:
        - pd.DataFrame: DataFrame transformado por PCA con las columnas de componentes principales.
        """
        # Normalizar datos y aplicar PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df.select_dtypes(include=['float64', 'int64']).fillna(self.df.mean()))
        
        # Crear el modelo PCA y transformar los datos
        self.pca_model = PCA(n_components=n_components)
        pca_data = self.pca_model.fit_transform(scaled_data)
        
        # Crear DataFrame con componentes principales
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        self.df_pca = pd.DataFrame(pca_data, columns=pca_columns)
        
        # Retornar el DataFrame con los componentes principales
        return self.df_pca
        

    def pca_elbow_method(self, max_components=10, n_clusters=3):
        """
        Aplica el método del codo y calcula el coeficiente de silueta para PCA,
        graficando la varianza explicada acumulada y el coeficiente de silueta
        en función del número de componentes.

        Parameters:
        - max_components (int): Número máximo de componentes principales a considerar para el análisis.
        - n_clusters (int): Número de clusters para calcular el coeficiente de silueta.

        Returns:
        - None: Genera gráficos para visualizar la varianza explicada acumulada y el coeficiente de silueta.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df.select_dtypes(include=['float64', 'int64']).fillna(self.df.mean()))
        
        # Ajustar el número máximo de componentes a la cantidad de características en los datos
        n_features = scaled_data.shape[1]
        max_components = min(max_components, n_features)

        explained_variances = []
        silhouette_scores = []

        # Iterar sobre el número de componentes principales de 1 a max_components
        for n_components in range(1, max_components + 1):
            # Aplicar PCA con el número de componentes actual
            pca = PCA(n_components=n_components)
            pca_data = pca.fit_transform(scaled_data)

            # Almacenar la varianza explicada acumulada
            explained_variances.append(np.sum(pca.explained_variance_ratio_))

            # Aplicar K-means y calcular el coeficiente de silueta
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(pca_data)
            silhouette_avg = silhouette_score(pca_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        # Graficar la varianza explicada acumulada y el coeficiente de silueta
        plt.figure(figsize=(12, 5))

        # Gráfico del método del codo (varianza explicada acumulada)
        plt.subplot(1, 2, 1)
        plt.plot(range(1, max_components + 1), explained_variances, marker='o')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Varianza Explicada Acumulada')
        plt.title('Método del Codo para PCA')
        plt.grid(True)

        # Gráfico del coeficiente de silueta
        plt.subplot(1, 2, 2)
        plt.plot(range(1, max_components + 1), silhouette_scores, marker='o')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Coeficiente de Silueta')
        plt.title('Coeficiente de Silueta en función del número de componentes')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    def kmeans_silhouette_and_elbow(self, max_clusters=10):
        """
        Aplica los métodos de la silueta y el codo para K-means, probando distintos números de clusters
        y graficando el coeficiente de silueta promedio y la inercia para cada uno.

        Parameters:
        - max_clusters (int): Número máximo de clusters a probar.
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df)

        silhouette_scores = []
        inertia_values = []
        cluster_range = range(2, max_clusters + 1)

        # Probar diferentes números de clusters
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertia_values.append(kmeans.inertia_)  # Inercia para el método del codo

        # Graficar el método de la silueta
        plt.figure(figsize=(12, 5))
        
        # Coeficiente de Silueta
        plt.subplot(1, 2, 1)
        plt.plot(cluster_range, silhouette_scores, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Coeficiente de Silueta')
        plt.title('Método de la Silueta para Selección de Clusters K-means')
        plt.grid(True)
        
        # Método del Codo
        plt.subplot(1, 2, 2)
        plt.plot(cluster_range, inertia_values, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inercia (Suma de Errores Cuadráticos)')
        plt.title('Método del Codo para Selección de Clusters K-means')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


    def graficar_matriz_confusion(self):
        """
        Genera la gráfica de la matriz de confusión entre los clusters y la clasificación de obesidad original.
    
        Este método verifica si el DataFrame contiene una columna llamada 'Cluster' y otra llamada 'NObeyesdad'.
        Si estas columnas están presentes, convierte la columna 'NObeyesdad' en etiquetas numéricas,
        calcula la matriz de confusión entre los valores originales y los clusters predichos, y
        genera una gráfica de la matriz de confusión para facilitar la visualización de la clasificación.
    
        Parámetros
        ----------
        Ninguno.
    
        Retorna
        -------
        None
            Este método no retorna ningún valor. Genera y muestra una gráfica de la matriz de confusión.
    
        Ejemplo de uso
        --------------
        # Crear instancia del modelo con un DataFrame que incluye las columnas 'NObeyesdad' y 'Cluster'
        model = Models(df)
    
        # Generar la gráfica de la matriz de confusión
        model.graficar_matriz_confusion()
        
        Notas
        -----
        - La columna 'NObeyesdad' se convierte en etiquetas numéricas mediante Label Encoding.
        - La columna 'Cluster' debe contener los clusters predichos del modelo de clustering.
        - Si alguna de estas columnas no está presente en `self.df`, se mostrará un mensaje de advertencia en lugar de la gráfica.
        """
        if self.df is not None and 'Cluster' in self.df.columns:
            # Convertir la variable de obesidad a etiquetas numéricas
            le = LabelEncoder()
            self.df['ObesityClass'] = le.fit_transform(self.df['NObeyesdad'])
            
            # Calcular la matriz de confusión
            conf_matrix = confusion_matrix(self.df['ObesityClass'], self.df['Cluster'])

            # Graficar la matriz de confusión
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=[f"Cluster {i}" for i in range(conf_matrix.shape[1])],
                        yticklabels=le.classes_)
            plt.xlabel("Clusters Predichos")
            plt.ylabel("Clasificación de Obesidad Original")
            plt.title("Matriz de Confusión entre Clusters y Clasificación de Obesidad")
            plt.show()
        else:
            print("Asegúrate de haber aplicado el modelo K-means y de que el DataFrame esté cargado.") 
            
    def plot_clusters_comparison(self, component_1='PC1', component_2='PC2', component_3=None, elevation_azimuth=(30, 120)):
        """
        Grafica los clusters obtenidos antes y después de PCA para comparar.

        Parameters:
        - component_1 (str): Nombre del primer componente principal (eje X).
        - component_2 (str): Nombre del segundo componente principal (eje Y).
        - component_3 (str, opcional): Nombre del tercer componente principal (eje Z en el gráfico 3D).
        - elevation_azimuth (tuple): Par de valores (elevación, azimuth) para ajustar la vista del gráfico 3D.

        Returns:
        - None: Genera gráficos para visualizar la comparación de los clusters.
        """
        elevation, azimuth = elevation_azimuth  # Desempaquetar los valores de elevación y azimuth

        if self.df_kmeans is not None and 'Cluster' in self.df_kmeans.columns:
            plt.figure(figsize=(10, 5))
            
            # Gráfico de clusters antes de PCA
            if component_3:
                # Gráfico 3D
                ax = plt.subplot(1, 2, 1, projection='3d')
                ax.scatter(self.df_kmeans[component_1], self.df_kmeans[component_2], self.df_kmeans[component_3], 
                           c=self.df_kmeans['Cluster'], cmap='viridis', s=50, alpha=0.6)
                ax.set_xlabel(component_1)
                ax.set_ylabel(component_2)
                ax.set_zlabel(component_3)
                ax.set_title("Clusters antes de PCA (3D)")
                ax.view_init(elev=elevation, azim=azimuth)  # Ajustar la vista 3D
            else:
                # Gráfico 2D
                plt.subplot(1, 2, 1)
                plt.scatter(self.df_kmeans[component_1], self.df_kmeans[component_2], 
                            c=self.df_kmeans['Cluster'], cmap='viridis', s=50, alpha=0.5)
                plt.xlabel(component_1)
                plt.ylabel(component_2)
                plt.title("Clusters antes de PCA (2D)")
                plt.grid(True)

        if self.df_pca is not None and 'Cluster' in self.df_pca.columns:
            # Gráfico de clusters después de PCA
            if component_3:
                # Gráfico 3D
                ax = plt.subplot(1, 2, 2, projection='3d')
                ax.scatter(self.df_pca[component_1], self.df_pca[component_2], self.df_pca[component_3], 
                           c=self.df_pca['Cluster'], cmap='viridis', s=50, alpha=0.5)
                ax.set_xlabel(component_1)
                ax.set_ylabel(component_2)
                ax.set_zlabel(component_3)
                ax.set_title("Clusters después de PCA (3D)")
                ax.view_init(elev=elevation, azim=azimuth)  # Ajustar la vista 3D
            else:
                # Gráfico 2D
                plt.subplot(1, 2, 2)
                plt.scatter(self.df_pca[component_1], self.df_pca[component_2], 
                            c=self.df_pca['Cluster'], cmap='viridis', s=50, alpha=0.5)
                plt.xlabel(component_1)
                plt.ylabel(component_2)
                plt.title("Clusters después de PCA (2D)")
                plt.grid(True)

        plt.tight_layout()
        plt.show()