from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class SparkConfig:
    app_name: str = "BigDataLab5"
    master: str = "local[*]"
    executor_memory: str = "2g"
    driver_memory: str = "2g"
    spark_executor_cores: int = 2
    spark_sql_shuffle_partitions: int = 4
    spark_dynamic_allocation_enabled: bool = False
    
    def get_spark_session(self):
        from pyspark.sql import SparkSession
        return SparkSession.builder \
            .appName(self.app_name) \
            .master(self.master) \
            .config("spark.executor.memory", self.executor_memory) \
            .config("spark.driver.memory", self.driver_memory) \
            .config("spark.executor.cores", self.spark_executor_cores) \
            .config("spark.sql.shuffle.partitions", self.spark_sql_shuffle_partitions) \
            .config("spark.dynamicAllocation.enabled", self.spark_dynamic_allocation_enabled) \
            .getOrCreate()

@dataclass
class Config:
    raw_data_path: str = os.getenv("RAW_DATA_PATH")
    processed_data_path: str = os.getenv("PROCESSED_DATA_PATH")

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