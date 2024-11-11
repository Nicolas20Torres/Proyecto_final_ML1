
### Descripción de Carpetas y Archivos

- **`.gitignore`**: Archivo de configuración para excluir archivos temporales y dependencias específicas del entorno Anaconda y herramientas de desarrollo, como Spyder y Jupyter.

- **`.spyproject/`**: Directorio que contiene configuraciones del entorno de desarrollo de Spyder.

- **`Database/`**: Contiene el conjunto de datos necesario para el entrenamiento y prueba de los modelos. En este proyecto, se utiliza el conjunto de datos `ObesityDataSet_raw_and_data_sinthetic`.

- **`Documentation/`**: Carpeta destinada a la documentación del proyecto, detallando el propósito, metodología, resultados y conclusiones obtenidas.

- **`Machine/`**: Aquí se encuentran las clases y métodos de los modelos de machine learning seleccionados para el proyecto. Este directorio contiene la implementación de modelos supervisados y no supervisados, así como métodos para su evaluación.

- **`Notebooks/`**: Carpeta que contiene los archivos Jupyter Notebook que documentan el desarrollo e implementación de los modelos y análisis. Incluye el flujo de trabajo desde la limpieza de datos hasta la evaluación de modelos.

- **`pyproject.toml`**: Archivo de configuración que define las dependencias y la estructura del proyecto.

- **`README.md`**: Este archivo, que proporciona una visión general del proyecto, sus objetivos, estructura y uso.

- **`Services/`**: Contiene clases y métodos para la lectura, transformación y carga de datos, funciones clave para el preprocesamiento y preparación de datos que serán utilizados en los modelos de machine learning.

- **`Temps/`**: Carpeta que almacena archivos temporales y datos generados durante la ejecución de los notebooks de Jupyter. Estos archivos se crean y eliminan conforme a las necesidades del análisis y desarrollo.

---

## Objetivo del Proyecto

El objetivo principal de este proyecto es desarrollar modelos de aprendizaje supervisado y no supervisado, aplicados al conjunto de datos `ObesityDataSet_raw_and_data_sinthetic`. Buscamos contrastar las etiquetas originales del conjunto de datos con las etiquetas generadas mediante modelos no supervisados para evaluar su precisión. Los resultados se utilizarán para comparar y ajustar los modelos de acuerdo con distintas parametrizaciones y técnicas de interpretabilidad.

---

## Metodología

1. **Análisis y Preprocesamiento de Datos**  
   Se realiza una limpieza y transformación inicial de los datos en el módulo `Services/`. Esto incluye la gestión de valores faltantes, normalización y codificación de variables, y la preparación de los datos para el modelado.

2. **Modelos de Machine Learning**  
   - **Modelos Supervisados**: Implementados en el módulo `Machine/`, estos modelos se entrenan con las etiquetas reales del conjunto de datos.
   - **Modelos No Supervisados**: Estos modelos se entrenan sin etiquetas y se evalúan mediante la comparación con las etiquetas generadas automáticamente.

3. **Ajuste de Hiperparámetros**  
   Para optimizar el desempeño de los modelos, se aplican técnicas de **Grid Search** y **Random Search**, que permiten probar diferentes configuraciones de hiperparámetros y seleccionar los que mejor rendimiento ofrecen.

4. **Interpretabilidad de Modelos**  
   Con el fin de entender cómo las características del modelo afectan las predicciones, se emplearán herramientas de interpretabilidad como **SHAP** (SHapley Additive exPlanations) y **LIME** (Local Interpretable Model-agnostic Explanations). Estas técnicas ayudan a explicar el impacto de cada variable en los resultados del modelo.

---

## Requisitos

- Python 3.8 o superior
- Anaconda o Miniconda (para la gestión del entorno)
- Librerías específicas incluidas en `pyproject.toml`, tales como:
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `shap`
  - `lime`
  
Para instalar todas las dependencias, se recomienda ejecutar:
```bash
conda env create -f environment.yml
conda activate Cluster_Machine
 
