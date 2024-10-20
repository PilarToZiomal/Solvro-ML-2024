# src/clustering.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import umap
import src.config as config
import warnings

# Wyłączenie ostrzeżeń
warnings.filterwarnings("ignore")

def main():
    # Wczytanie danych z pliku CSV
    file_path = config.ALL_COCKTAILS_CSV
    cocktails_df = pd.read_csv(file_path)

    # Wybranie cech do klasteryzacji
    X = cocktails_df[['category_vector', 'glass_vector', 'tag_vector_norm', 'ingredient_vector_norm']]

    # Standardyzacja danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Klasteryzacja K-means z 6 klastrami
    kmeans = KMeans(n_clusters=6, random_state=42)
    cocktails_df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

    # Obliczanie wskaźników jakości klasteryzacji
    silhouette_avg = silhouette_score(X_scaled, cocktails_df['kmeans_cluster'])
    calinski_harabasz = calinski_harabasz_score(X_scaled, cocktails_df['kmeans_cluster'])
    davies_bouldin = davies_bouldin_score(X_scaled, cocktails_df['kmeans_cluster'])
    sse = kmeans.inertia_

    # Zapis wyników wskaźników do pliku
    quality_metrics = {
        'Silhouette Score': silhouette_avg,
        'Calinski-Harabasz Index': calinski_harabasz,
        'Davies-Bouldin Index': davies_bouldin,
        'SSE': sse
    }
    pd.DataFrame([quality_metrics]).to_csv(config.OUTPUT_DIR / 'kmeans_quality_metrics.csv', index=False)

    # Grupowanie drinków według klastrów i zapisanie wyników
    cluster_summary = cocktails_df.groupby('kmeans_cluster')['name'].apply(list)
    cluster_summary_df = pd.DataFrame({'Cluster': cluster_summary.index, 'Drinks': cluster_summary.values})
    cluster_summary_df.to_csv(config.OUTPUT_DIR / 'cluster_summary.csv', index=False)

    # Redukcja wymiarów za pomocą PCA (do 2 wymiarów)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    cocktails_df['PCA1'] = X_pca[:, 0]
    cocktails_df['PCA2'] = X_pca[:, 1]

    # Wizualizacja klastrów
    plt.figure(figsize=(10, 7))
    for cluster in range(6):
        cluster_points = cocktails_df[cocktails_df['kmeans_cluster'] == cluster]
        plt.scatter(cluster_points['PCA1'], cluster_points['PCA2'], label=f'Cluster {cluster}', alpha=0.6)

    plt.title('Wizualizacja klastrów (K-means + PCA)')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()
    plt.savefig(config.OUTPUT_DIR / 'kmeans_clusters_pca.png')
    plt.close()

    # Redukcja wymiarów za pomocą UMAP (do 2 wymiarów)
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embedding = reducer.fit_transform(X_scaled)
    cocktails_df['UMAP1'] = umap_embedding[:, 0]
    cocktails_df['UMAP2'] = umap_embedding[:, 1]

    # Wizualizacja klastrów UMAP
    plt.figure(figsize=(10, 7))
    for cluster in range(6):
        cluster_points = cocktails_df[cocktails_df['kmeans_cluster'] == cluster]
        plt.scatter(cluster_points['UMAP1'], cluster_points['UMAP2'], label=f'Cluster {cluster}', alpha=0.6)

    plt.title('Wizualizacja klastrów (K-means + UMAP)')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend()
    plt.savefig(config.OUTPUT_DIR / 'kmeans_clusters_umap.png')
    plt.close()

if __name__ == "__main__":
    main()
