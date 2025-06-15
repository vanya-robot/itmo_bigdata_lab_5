from pyspark.ml.clustering import KMeansModel
from pyspark.sql import SparkSession
from src.config import Config
from src.data_preprocessing import DataPreprocessor

class KMeansTester:
    def __init__(self, config: Config):
        self.config = config
        self.spark = SparkSession.builder \
            .appName("KMeansTesting") \
            .getOrCreate()

    def load_model(self, path):
        """Загрузка модели"""
        return KMeansModel.load(path)

    def load_data(self):
        """Загрузка данных для тестирования"""
        return self.spark.read.parquet(self.config.processed_data_path)

    def predict(self, model, df):
        """Предсказание кластеров"""
        return model.transform(df)

    def show_predictions(self, predictions):
        """Отображение предсказаний"""
        predictions.select("prediction").show(20)
        
        # Статистика по кластерам
        predictions.groupBy("prediction").count().orderBy("prediction").show()

    def run(self):
        """Запуск тестирования модели"""
        model = self.load_model("./models/kmeans_model")
        df = self.load_data()
        predictions = self.predict(model, df)
        self.show_predictions(predictions)
        self.spark.stop()
        return predictions

if __name__ == "__main__":
    config = Config()
    tester = KMeansTester(config)
    tester.run()