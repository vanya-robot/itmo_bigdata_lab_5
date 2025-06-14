from dataclasses import dataclass

@dataclass
class Config:
    # Пути к данным
    raw_data_path: str = "../data/raw/en.openfoodfacts.org.products.csv"
    processed_data_path: str = "../data/processed/processed_data.parquet"
    
    # Параметры модели
    k: int = 5  # Количество кластеров
    max_iter: int = 20  # Максимальное количество итераций
    seed: int = 42  # Random seed
    
    # Колонки для анализа
    numeric_columns: list = None
    selected_columns: list = None
    
    def __post_init__(self):
        # Выберем некоторые числовые колонки для анализа
        self.numeric_columns = [
            'energy_100g', 'fat_100g', 'carbohydrates_100g',
            'sugars_100g', 'proteins_100g', 'salt_100g',
            'sodium_100g', 'fiber_100g'
        ]
        
        # Колонки, которые мы будем использовать для кластеризации
        self.selected_columns = [
            'energy_100g', 'fat_100g', 'carbohydrates_100g',
            'proteins_100g', 'sugars_100g'
        ]