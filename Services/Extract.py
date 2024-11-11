import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class NameDataset:
    Dataset_Obesidad  = 'ObesityDataSet_raw_and_data_sinthetic.csv'


class PathDataset:
    """
    Clase que almacena la ruta al archivo del dataset.
    """
    def __init__(self, dataset_name, base_path=None):
        """
        Inicializa la clase con el nombre del dataset y una ruta base opcional.
        
        Args:
            dataset_name (str): Nombre del archivo del dataset.
            base_path (str): Ruta base opcional. Si no se proporciona, usa el directorio 'DataBase'.
        """
        self.dataset_name = dataset_name
        self.base_path = base_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DataBase')
        self.path_data = os.path.join(self.base_path, self.dataset_name)
        
    def select_dataset(self):
        """
        Retorna la ruta del dataset si existe; de lo contrario, lanza un error.
        
        Returns:
            str: Ruta completa del archivo de datos.
        
        Raises:
            FileNotFoundError: Si el archivo no se encuentra.
        """
        if not os.path.exists(self.path_data):
            raise FileNotFoundError(f'El conjunto de datos {self.path_data} no existe')
        
        return self.path_data  
