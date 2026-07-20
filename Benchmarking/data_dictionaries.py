from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from abc import ABC, abstractmethod
import datetime as dt
import openml as om
from sklearn.datasets import fetch_openml

class Dataset(ABC):
    """Base class for datasets."""
    name = None
    id = None
    source = None
    
    def load(self):
        """Unified entry point called by the framework loader."""
        if self.source == 'uci':
            return self.load_uci()
        elif self.source == 'openml':
            return self.load_openml()
        else:
            raise ValueError(f"Unknown data source: {self.source}")

    def load_uci(self):
        """Utility fallback to load dataset from UCI repo."""
        dataset = fetch_ucirepo(id=self.id)
        data_x = dataset.data.features
        data_y = dataset.data.targets
        return data_x, data_y, data_x.shape[1], data_y.shape[1] if len(data_y.shape) > 1 else 1

    def load_openml(self):
        """Utility fallback to load dataset from OpenML."""
        # fetch_openml returns a bunch-like object; parser='auto' is recommended
        dataset = fetch_openml(data_id=self.id, as_frame=True, parser='auto')
        
        # If your targets/features aren't automatically split by openml, 
        # we pull the full frame or explicitly slice them.
        if dataset.target is not None:
            data_x = pd.DataFrame(dataset.data)
            data_y = pd.DataFrame(dataset.target)
        else:
            # Fallback if target is unassigned in OpenML metadata
            df = dataset.frame
            data_x = df.iloc[:, :-1]
            data_y = df.iloc[:, [-1]]

        return data_x, data_y, data_x.shape[1], data_y.shape[1] if len(data_y.shape) > 1 else 1


class WineQuality(Dataset):
    """Wine Quality dataset with color selection."""
    def __init__(self, wine_color=None):
        self.name = 'Wine_Quality'
        self.id = 186
        self.wine_color = wine_color
    def load_uci(self):
        """Load Wine Quality dataset and prompt for wine color selection."""
        dataset = fetch_ucirepo(id=self.id)
        df = dataset.data.original
        red_wine = df[df['color'].str.strip() == 'red'].copy()
        white_wine = df[df['color'].str.strip() == 'white'].copy()
        red_wine = red_wine.drop(columns=['color'])
        white_wine = white_wine.drop(columns=['color'])

        if self.wine_color == 'red':
            data_x = red_wine.drop(columns=['quality'])
            data_y = pd.DataFrame(red_wine['quality'])
            self.name = 'Red_Wine_Quality'
        elif self.wine_color == 'white':
            data_x = white_wine.drop(columns=['quality'])
            data_y = pd.DataFrame(white_wine['quality'])
            self.name = 'White_Wine_Quality'
        
        return data_x, data_y, data_x.shape[1], data_y.shape[1] if len(data_y.shape) > 1 else 1


class AppliancesEnergyPrediction(Dataset):
    """Appliances Energy Prediction dataset."""
    name = 'Appliances Energy Prediction'
    id = 374
    def load_uci(self):
        """Load Appliances Energy Prediction dataset."""
        #try: 
        dataset = fetch_ucirepo(id=self.id)
        data_x = dataset.data.features
        datetime_data = pd.to_datetime(data_x['date'], format='%Y-%m-%d%H:%M:%S')
        data_x.drop(columns=['date'], inplace=True)
        jan_first = dt.date(datetime_data[0].year, 1, 1)
        data_x['days_since_jan1'] = (datetime_data - pd.to_datetime(jan_first)).dt.days
        data_x['hours_since_midnight'] = datetime_data.dt.hour + datetime_data.dt.minute / 60
        data_x = pd.concat([data_x, pd.DataFrame(OneHotEncoder(sparse_output=False).fit_transform(datetime_data.dt.weekday.to_frame()), columns=[f'weekday_{i}' for i in range(7)])], axis=1)
        data_y = dataset.data.targets
        return data_x, data_y, data_x.shape[1], data_y.shape[1] if len(data_y.shape) > 1 else 1


class RTIoT2022(Dataset):
    """RT-IoT2022 (RT-IoT) dataset."""
    name = 'RT-IoT2022'
    id = 942
    def load_uci(self):
        """Load RT-IoT2022 dataset."""
        dataset = fetch_ucirepo(id=self.id)
        data_x = dataset.data.features
        proto_data = data_x['proto']
        data_x.drop(columns=['proto'], inplace=True)
        data_x = pd.concat([data_x, pd.DataFrame(OneHotEncoder(sparse_output=False).fit_transform(proto_data.to_frame()), columns=[f'proto_{i}' for i in range(proto_data.nunique())])], axis=1)
        service_data = data_x['service']
        data_x.drop(columns=['service'], inplace=True)
        data_x = pd.concat([data_x, pd.DataFrame(OneHotEncoder(sparse_output=False).fit_transform(service_data.to_frame()), columns=[f'service_{i}' for i in range(service_data.nunique())])], axis=1)
        data_y = dataset.data.targets
        attack_type = data_y['Attack_type']
        data_y.drop(columns=['Attack_type'], inplace=True)
        #One hot encode attack type and concatenate with features
        data_y = pd.concat([data_y, pd.DataFrame(OneHotEncoder(sparse_output=False).fit_transform(attack_type.to_frame()), columns=[f'attack_type_{i}' for i in range(attack_type.nunique())])], axis=1)
        return data_x, data_y, data_x.shape[1], data_y.shape[1] if len(data_y.shape) > 1 else 1


class ConcreteCompressiveStrength(Dataset):
    """Concrete Compressive Strength dataset."""
    def __init__(self):
        self.name = 'Concrete Compressive Strength'
        self.id = 165
        self.source = 'uci'

class CombinedCyclePowerPlant(Dataset):
    """Combined Cycle Power Plant dataset."""
    def __init__(self):
        self.name = 'Combined Cycle Power Plant'
        self.id = 294
        self.source = 'uci'

class Space_Ga(Dataset):
    """Space_Ga dataset."""
    def __init__(self):
        self.name = 'Space_Ga'
        self.id = 507
        self.source = 'openml'

class House_Sales(Dataset):
    """House Sales dataset."""
    def __init__(self):
        self.name = 'House_Sales'
        self.id = 42731
        self.source = 'openml'

class CpuActOpenML(Dataset):
    """Cpu_act dataset from Grinsztajn et al. Benchmark."""
    def __init__(self):
        self.name = 'Cpu_Act'
        self.id = 44132
        self.source = 'openml'

class AileronsOpenML(Dataset):
    """Ailerons dataset from Grinsztajn et al. Benchmark."""
    def __init__(self):
        self.name = 'Ailerons'
        self.id = 44135
        self.source = 'openml'

class HousesOpenML(Dataset):
    """Houses dataset (California / Boston Housing variants) from Grinsztajn et al. Benchmark."""
    def __init__(self):
        self.name = 'Houses_OpenML'
        self.id = 44141
        self.source = 'openml'

class ElevatorsOpenML(Dataset):
    """Elevators dataset from Grinsztajn et al. Benchmark."""
    def __init__(self):
        self.name = 'Elevators'
        self.id = 44133
        self.source = 'openml'

class PolOpenML(Dataset):
    """Pol dataset from Grinsztajn et al. Benchmark."""
    def __init__(self):
        self.name = 'Pol_OpenML'
        self.id = 44134
        self.source = 'openml'

class BikeSharingOpenML(Dataset):
    """Bike Sharing dataset from Grinsztajn et al. Benchmark."""
    def __init__(self):
        self.name = 'Bike_Sharing_OpenML'
        self.id = 44148
        self.source = 'openml'

class MiamiHousingOpenML(Dataset):
    """Miami Housing dataset from Grinsztajn et al. Benchmark."""
    def __init__(self):
        self.name = 'Miami_Housing_OpenML'
        self.id = 44147
        self.source = 'openml'

# Dictionary mapping dataset names to classes
DATASETS = {
    # Existing UCI Datasets
    'Red_Wine_Quality': lambda: WineQuality(wine_color='red'),
    'White_Wine_Quality': lambda: WineQuality(wine_color='white'),
    'Concrete_Compressive_Strength': lambda: ConcreteCompressiveStrength(),
    'Combined_Cycle_Power_Plant': lambda: CombinedCyclePowerPlant(),
    
    # New OpenML Paper Benchmarks (Numerical Suite)
    'Cpu_Act': lambda: CpuActOpenML(),
    'Ailerons': lambda: AileronsOpenML(),
    'Houses_OpenML': lambda: HousesOpenML(),
    'Elevators': lambda: ElevatorsOpenML(),
    'Pol_OpenML': lambda: PolOpenML(),
    
    # New OpenML Paper Benchmarks (Mixed Suite)
    'Bike_Sharing_OpenML': lambda: BikeSharingOpenML(),
    'Miami_Housing_OpenML': lambda: MiamiHousingOpenML(),
}

# The automated downstream indices look up everything flawlessly from here:
ID_TO_DATASET = {v().id: v for v in DATASETS.values()}
id_dict = {name: dataset_class().id for name, dataset_class in DATASETS.items()}
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