from src.utils import load_data
from src.eda import eda
from src.preprocessing import preprocess_data
from src.clustering import Clustering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Wczytanie danych
cocktail_df = load_data('data/cocktail_dataset.json')

# Analiza eksploracyjna danych
eda(cocktail_df)

# Preprocessing danych
X_scaled, prepared_df = preprocess_data(cocktail_df)

# Klasteryzacja
clustering = Clustering(n_clusters=4, random_state=42)
labels = clustering.fit(X_scaled)

# Ewaluacja ilościowa - Metryki jakości klasteryzacji
evaluation_metrics = clustering.evaluate(X_scaled)
print(f"Silhouette Score: {evaluation_metrics['silhouette_score']}")
print(f"Davies-Bouldin Score: {evaluation_metrics['davies_bouldin_score']}")

# Dodanie etykiet klastrów do DataFrame
prepared_df['cluster'] = labels

# Wizualizacja wyników klasteryzacji
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis')
plt.title('Wizualizacja klastrów (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Klastry')
plt.savefig('outputs/clusters_pca.pdf', format='pdf')

# Zapisanie wyników do pliku CSV
clustering.save_results(prepared_df, 'outputs/clustered_cocktails.csv')