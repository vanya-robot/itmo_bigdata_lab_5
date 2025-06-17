from dataclasses import dataclass

@dataclass
class Config:
    raw_data_path: str = "/home/alexe/itmo/bigdata/itmo_bigdata_lab_5/data/raw/en.openfoodfacts.org.products.csv.gz"
    processed_data_path: str = "/home/alexe/itmo/bigdata/itmo_bigdata_lab_5/data/processed/processed_data.parquet"

    k: int = 5
    max_iter: int = 20
    seed: int = 123456

    numeric_columns: list = None
    selected_columns: list = None
    
    def __post_init__(self):
        self.numeric_columns = [
            'energy_100g', 'fat_100g', 'carbohydrates_100g',
            'sugars_100g', 'proteins_100g', 'salt_100g',
            'sodium_100g', 'fiber_100g'
        ]
        
        self.selected_columns = [
            'energy_100g', 'fat_100g', 'carbohydrates_100g',
            'proteins_100g', 'sugars_100g'
        ]