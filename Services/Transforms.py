import os
import sys 
import pandas as pd
from Services.Extract  import PathDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class Transforms:
    def __init__(self, dataset_name):
        self.df = None
        """
        Inicializa la clase Read con el nombre del dataset.

        Args:
            dataset_name (str): Nombre del archivo del dataset (por ejemplo, 'ObesityDataSet_raw_and_data_sinthetic.csv').
        """
        # Crear una instancia de PathDataset con el nombre del archivo del dataset
        self.path = PathDataset(dataset_name)

    def read_df(self, sep=","):
        """
        Lee el archivo CSV en un DataFrame de pandas.
        
        Args:
            sep (str): Separador del archivo CSV (por defecto es una coma).
        
        Returns:
            DataFrame: Datos leídos en un DataFrame de pandas.
        """
        # Obtener la ruta completa del dataset
        dataset_path = self.path.select_dataset()
        
        # Leer el archivo CSV y retornarlo como DataFrame
        self.df = pd.read_csv(dataset_path, sep=sep)
        return  self.df

    def descripcion(self):
        """
        Proporciona una descripción detallada del DataFrame, incluyendo:
        - Tamaño de los datos (shape)
        - Nombres de las columnas
        - Información general de las columnas (info)
        - Estadísticas descriptivas de las variables numéricas
        - Conteo de valores únicos para las variables categóricas
        """
        if self.df is not None:
            print("-------------------------------------------------------------")
            print("Shape del DataFrame:", self.df.shape)
            print("-------------------------------------------------------------")
            print("\nNombres de las columnas:", list(self.df.columns))
            print("-------------------------------------------------------------")
            print("\nInformación del DataFrame:")
            print(self.df.info())
            print("-------------------------------------------------------------")
            print("\nDescripción estadística de las variables numéricas:")
            print(self.df.describe())
            print("-------------------------------------------------------------")
            print("\nConteo de valores únicos para variables categóricas:")
            for col in self.df.select_dtypes(include=['object']).columns:
                print(f"{col}: {self.df[col].nunique()} valores únicos")
            print("-------------------------------------------------------------")
        else:
            print("DataFrame no cargado. Utiliza el método read_df() para cargar los datos.")
            
    def seleccionar_columnas(self,columnas):
        """
        Selecciona un subconjunto de columnas del DataFrame y actualiza `self.df` con las columnas especificadas.
    
        Este método crea una copia del DataFrame `self.df` que contiene solo las columnas indicadas en el parámetro `columnas`,
        y luego actualiza `self.df` con este subconjunto de datos.
    
        Parámetros
        ----------
        columnas : list of str
            Lista de nombres de columnas que se desean mantener en el DataFrame.
    
        Retorna
        -------
        pd.DataFrame
            El DataFrame `self.df` actualizado que contiene únicamente las columnas seleccionadas.
    
        Ejemplo de uso
        --------------
        # Crear instancia del modelo con un DataFrame
        model = Models(df)
    
        # Seleccionar columnas específicas
        df_reducido = model.seleccionar_columnas(columnas=['columna1', 'columna2'])
    
        # Visualizar DataFrame reducido
        print(df_reducido.head())
    
        Notas
        -----
        - Si alguna de las columnas en `columnas` no está presente en el DataFrame, se generará un error.
        """
        datos = self.df.copy()
        datos = datos[columnas]
        self.df = datos
        return self.df
            
    def graficar_categorias(self):
        """
        Filtra las variables categóricas del DataFrame y genera gráficos de barras para cada una.
        """
        if self.df is not None:
            # Filtrar las variables categóricas
            categoricas = self.df.select_dtypes(include=['object']).columns
            
            # Crear gráficos de barras para cada variable categórica
            for col in categoricas:
                plt.figure(figsize=(8, 4))
                sns.countplot(x=self.df[col], palette="viridis", order=self.df[col].value_counts().index)
                plt.title(f'Distribución de {col}')
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.show()
        else:
            print("DataFrame no cargado. Utiliza el método read_df() para cargar los datos.")

    def graficar_correlation(self, titulo="Mapa de Correlaciones de Variables Numéricas", exclude_columns=None):
        """
        Genera una gráfica de correlación que muestra histogramas en la diagonal,
        gráficos de dispersión en la parte inferior y coeficientes de correlación en la parte superior.

        Parámetros:
        - titulo: Título general del gráfico.
        - exclude_columns: Lista de columnas a excluir del análisis.
        """
        # Verificar que el DataFrame esté cargado
        if self.df is not None:
            # Excluir las columnas especificadas si se proporciona alguna
            df_numerico = self.df.select_dtypes(include=['float64', 'int64'])
            if exclude_columns:
                df_numerico = df_numerico.drop(columns=exclude_columns)

            # Obtener las correlaciones
            corr = df_numerico.corr()

            # Definir el tamaño de la gráfica
            fig, axes = plt.subplots(len(df_numerico.columns), len(df_numerico.columns), figsize=(45, 30))

            # Agregar un título general a la figura
            fig.suptitle(titulo, fontsize=80, y=1.02)

            # Recorrer todas las variables (para crear subplots)
            for i, col1 in enumerate(df_numerico.columns):
                for j, col2 in enumerate(df_numerico.columns):
                    if i == j:
                        # Histograma en la diagonal principal
                        axes[i, j].hist(df_numerico[col1], bins=20, color='green', edgecolor='black')
                        axes[i, j].set_title(f'{col1}', fontsize=10)
                    elif i > j:
                        # Gráfico de dispersión en la diagonal inferior
                        axes[i, j].scatter(df_numerico[col2], df_numerico[col1], alpha=0.5, color='blue')
                        # Línea de tendencia
                        m, b = np.polyfit(df_numerico[col2], df_numerico[col1], 1)
                        axes[i, j].plot(df_numerico[col2], m * df_numerico[col2] + b, color='red', linestyle='-')
                    else:
                        # Mostrar burbuja basada en el coeficiente de correlación en la diagonal superior
                        coef = np.corrcoef(df_numerico[col1], df_numerico[col2])[0][1]

                        # Definir el tamaño de la burbuja
                        bubble_size = np.abs(coef) * 30000

                        # Dibujar la burbuja
                        axes[i, j].scatter(10, 10, s=bubble_size, alpha=0.6, color='red')

                        # Mostrar el valor del coeficiente dentro de la burbuja
                        axes[i, j].text(10, 10, f'{coef:.2f}', ha='center', va='center', fontsize=30, color='black')

                        # Ocultar los ejes
                        axes[i, j].axis('off')  # Ocultar ejes en los coeficientes de correlación

                    # Nombre de las columnas
                    if i == 0:
                        axes[i, j].set_title(col2, fontsize=25, rotation=90)

                    # Nombre de las filas
                    if j == 0:
                        axes[i, j].set_ylabel(col1, fontsize=25, rotation=0, labelpad=40, ha='right')

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()
        else:
            print("DataFrame no cargado. Utiliza el método read_df() para cargar los datos.")
            

    def tabla_dinamica(self,filas=None,columnas=None,valores=None,funciones=dict,campo_cantidad=None):
        if self.df is None or self.df.empty:
            print('El DataFrame esta vacio o es None')
            return None
        try:
            if not isinstance(funciones,dict):
                raise ValueError('El parametro funciones debe ser un diccionario')
            datos = self.df.copy()
            # Crear la tabla dinamica
            dinamica = pd.pivot_table(
                data=datos,
                index=filas,
                columns=columnas,
                values=valores,
                aggfunc=funciones
            ).reset_index()
            if dinamica.empty:
                return None
            if campo_cantidad is not None:
                dinamica = dinamica[dinamica[campo_cantidad]!=0]
            self.df = dinamica
            return self.df
        except KeyError as e:
            print(f'Error al generar la tabla dinamica: {e}')
            return None
