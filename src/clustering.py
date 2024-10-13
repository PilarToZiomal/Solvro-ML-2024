from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd

class Clustering:
    def __init__(self, n_clusters=4, random_state=42):
        """
        Inicjalizacja klasy Clustering.

        Argumenty:
            n_clusters (int): Liczba klastrów do utworzenia.
            random_state (int): Ziarno losowe dla powtarzalności wyników.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def fit(self, X):
        """
        Dopasowanie modelu klasteryzacji do danych.

        Argumenty:
            X (np.ndarray): Znormalizowane dane wejściowe do klasteryzacji.

        Zwraca:
            np.ndarray: Etykiety przypisane do każdego punktu danych.
        """
        self.model.fit(X)
        self.labels = self.model.labels_
        return self.labels

    def evaluate(self, X):
        """
        Ewaluacja jakości klasteryzacji za pomocą różnych metryk.

        Argumenty:
            X (np.ndarray): Znormalizowane dane wejściowe do klasteryzacji.

        Zwraca:
            dict: Słownik zawierający różne metryki ewaluacji.
        """
        sil_score = silhouette_score(X, self.labels)
        db_score = davies_bouldin_score(X, self.labels)
        return {
            'silhouette_score': sil_score,
            'davies_bouldin_score': db_score
        }

    def save_results(self, df, output_path):
        """
        Zapisanie wyników klasteryzacji do pliku CSV.

        Argumenty:
            df (pd.DataFrame): Dane wejściowe z przypisanymi etykietami klastrów.
            output_path (str): Ścieżka do pliku wyjściowego.
        """
        df['cluster'] = self.labels
        df.to_csv(output_path, index=False)