from ucimlrepo import fetch_ucirepo 
import pandas as pd
from abc import ABC, abstractmethod


class Dataset(ABC):
    """Base class for datasets."""
    name = None
    id = None
    
    def load(self):
        """Load dataset and return features, targets, and their dimensions.
        
        Returns:
            Tuple of (data_x, data_y, num_features, num_targets)
        """
        dataset = fetch_ucirepo(id=self.id)
        data_x = dataset.data.features
        data_y = dataset.data.targets
        return data_x, data_y, data_x.shape[1], data_y.shape[1]


class WineQuality(Dataset):
    """Wine Quality dataset with color selection."""
    name = 'Wine_Quality'
    id = 186
    
    def load(self):
        """Load Wine Quality dataset and prompt for wine color selection."""
        dataset = fetch_ucirepo(id=self.id)
        df = dataset.data.original
        red_wine = df[df['color'].str.strip() == 'red'].copy()
        white_wine = df[df['color'].str.strip() == 'white'].copy()
        red_wine = red_wine.drop(columns=['color'])
        white_wine = white_wine.drop(columns=['color'])

        wine_type = input("Choose wine type (red/white): ")
        if wine_type == 'red':
            data_x = red_wine.drop(columns=['quality'])
            data_y = pd.DataFrame(red_wine['quality'])
        elif wine_type == 'white':
            data_x = white_wine.drop(columns=['quality'])
            data_y = pd.DataFrame(white_wine['quality'])
        else:
            print("Invalid wine type. Please enter 'red' or 'white'.")
            return None, None, None, None
        
        return data_x, data_y, data_x.shape[1], data_y.shape[1]


class AppliancesEnergyPrediction(Dataset):
    """Appliances Energy Prediction dataset."""
    name = 'Appliances Energy Prediction'
    id = 374


class RTIoT2022(Dataset):
    """RT-IoT2022 (RT-IoT) dataset."""
    name = 'RT-IoT2022'
    id = 942


class ConcreteCompressiveStrength(Dataset):
    """Concrete Compressive Strength dataset."""
    name = 'Concrete Compressive Strength'
    id = 165


# Dictionary mapping dataset names to classes
DATASETS = {
    'Wine_Quality': WineQuality,
    'Appliances Energy Prediction': AppliancesEnergyPrediction,
    'RT-IoT2022': RTIoT2022,
    'Concrete Compressive Strength': ConcreteCompressiveStrength,
}

# Reverse mapping for ID lookup
ID_TO_DATASET = {v.id: v for v in DATASETS.values()}


# For backwards compatibility with code that uses id_dict and data_pull_dict
id_dict = {name: dataset_class.id for name, dataset_class in DATASETS.items()}
data_pull_dict = DATASETS


def load_dataset(identifier=None):
    """Load dataset by name or ID.
    
    Args:
        identifier: Dataset name (str) or ID (int/str). If None, prompts user.
    
    Returns:
        Tuple of (data_x, data_y, num_features, num_targets) or (None, None, None, None) if not found
    """
    if identifier is None:
        identifier = input(f"Enter dataset name or ID. Available: {list(DATASETS.keys())}: ")
    
    # Try to find by name first
    if identifier in DATASETS:
        dataset_class = DATASETS[identifier]
    else:
        # Try to find by ID
        try:
            dataset_id = int(identifier)
            dataset_class = ID_TO_DATASET.get(dataset_id)
            if dataset_class is None:
                print(f"Dataset with ID {dataset_id} not found.")
                return None, None, None, None
        except ValueError:
            print(f"Dataset '{identifier}' not found. Please use a name or valid ID.")
            return None, None, None, None
    
    # Instantiate and load
    dataset_instance = dataset_class()
    return dataset_instance.load()