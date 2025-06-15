from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
from src.config import Config

class KMeansTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.spark = SparkSession.builder \
            .appName("KMeansTraining") \
            .getOrCreate()

    def load_data(self):
        """Загрузка обработанных данных"""
        return self.spark.read.parquet(self.config.processed_data_path)

    def train_model(self, df):
        """Обучение модели K-средних"""
        kmeans = KMeans(
            k=self.config.k,
            maxIter=self.config.max_iter,
            seed=self.config.seed,
            featuresCol="scaled_features"
        )
        return kmeans.fit(df)

    def evaluate_model(self, model, df):
        """Оценка модели с помощью Silhouette score"""
        predictions = model.transform(df)
        
        evaluator = ClusteringEvaluator(
            featuresCol='scaled_features',
            metricName='silhouette',
            distanceMeasure='squaredEuclidean'
        )
        
        silhouette = evaluator.evaluate(predictions)
        print(f"\nSilhouette Score: {silhouette:.4f}")
        
        # Размеры кластеров
        cluster_sizes = predictions.groupBy("prediction").count().orderBy("prediction").collect()
        print("\nCluster Sizes:")
        for row in cluster_sizes:
            print(f"Cluster {row['prediction']}: {row['count']} items")
        
        # Центры кластеров
        centers = model.clusterCenters()
        print("\nCluster Centers:")
        for i, center in enumerate(centers):
            print(f"Cluster {i}: {[round(x, 4) for x in center]}")

    def save_model(self, model, path):
        """Сохранение модели"""
        model.save(path)
    
    def load_model(self, path):
        """Загрузка модели"""
        model = KMeans.load(path)
        return model

    def run(self):
        """Запуск обучения модели"""
        df = self.load_data()
        model = self.train_model(df)
        self.evaluate_model(model, df)
        self.save_model(model, "./models/kmeans_model")
        self.spark.stop()
        return model

if __name__ == "__main__":
    config = Config()
    trainer = KMeansTrainer(config)
    trainer.run()