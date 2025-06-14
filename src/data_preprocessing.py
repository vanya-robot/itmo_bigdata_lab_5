from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from src.config import Config

class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.spark = SparkSession.builder \
            .appName("DataPreprocessing") \
            .getOrCreate()

    def load_data(self):
        """Загрузка сырых данных"""
        return self.spark.read.csv(
            self.config.raw_data_path, 
            sep="\t", 
            header=True, 
            inferSchema=True
        )

    def preprocess(self, df):
        """Предобработка данных"""
        # Выбираем только нужные колонки
        df = df.select(self.config.selected_columns)
        
        # Заменяем некорректные значения на NULL
        for column in self.config.selected_columns:
            df = df.withColumn(
                column,
                when(col(column) < 0, None).otherwise(col(column)))
        
        # Импутация пропущенных значений
        imputers = [Imputer(
            inputCol=col, 
            outputCol=col, 
            strategy="mean"
        ) for col in self.config.selected_columns]
        
        # Создаем pipeline для импутации и масштабирования
        assembler = VectorAssembler(
            inputCols=self.config.selected_columns,
            outputCol="features"
        )
        
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        pipeline = Pipeline(stages=imputers + [assembler, scaler])
        model = pipeline.fit(df)
        processed_df = model.transform(df)
        
        return processed_df.select("scaled_features")

    def save_data(self, df, path):
        """Сохранение обработанных данных"""
        df.write.parquet(path, mode="overwrite")

    def run(self):
        """Запуск всего пайплайна предобработки"""
        df = self.load_data()
        processed_df = self.preprocess(df)
        self.save_data(processed_df, self.config.processed_data_path)
        self.spark.stop()
        return processed_df