from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
from config import MODEL_PARAMS
import os

def compute_wssse(df, k_values):
    """Вычисление WSSSE для разных значений k"""
    wssse = []
    for k in k_values:
        kmeans = KMeans(featuresCol="features", k=k, seed=42)
        model = kmeans.fit(df)
        wssse.append(model.computeCost(df))
    return wssse

def plot_elbow(wssse, k_values, output_path):
    """Построение графика метода локтя"""
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wssse, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WSSSE')
    plt.title('The Elbow Method showing optimal k')
    plt.savefig(output_path)
    plt.close()

def train_kmeans(df, k, output_path):
    """Обучение модели K-means"""
    kmeans = KMeans(featuresCol="features", k=k, seed=42)
    model = kmeans.fit(df)
    
    # Сохраняем модель
    model.save(output_path)
    
    # Предсказания
    predictions = model.transform(df)
    
    # Оценка качества
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    
    return model, predictions, silhouette

def run_clustering(df, output_dir):
    """Основной пайплайн кластеризации"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Определяем оптимальное k методом локтя
    k_values = range(MODEL_PARAMS['k_min'], MODEL_PARAMS['k_max'] + 1)
    wssse = compute_wssse(df, k_values)
    
    # Сохраняем график метода локтя
    elbow_path = os.path.join(output_dir, "elbow_method.png")
    plot_elbow(wssse, k_values, elbow_path)
    
    # Выбираем оптимальное k (здесь просто берем середину диапазона)
    optimal_k = (MODEL_PARAMS['k_min'] + MODEL_PARAMS['k_max']) // 2
    
    # Обучаем модель с оптимальным k
    model_path = os.path.join(output_dir, "kmeans_model")
    model, predictions, silhouette = train_kmeans(df, optimal_k, model_path)
    
    print(f"Optimal k: {optimal_k}")
    print(f"Silhouette score: {silhouette}")
    
    return model, predictions, silhouette