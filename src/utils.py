import pandas as pd


def load_data(path):
    """
    Funkcja wczytująca dane z pliku JSON.

    Argumenty:
    path (str): Ścieżka do pliku JSON z danymi.

    Zwraca:
    DataFrame: DataFrame z wczytanymi danymi.
    """
    return pd.read_json(path)


def save_data(df, path):
    """
    Funkcja zapisująca dane do pliku CSV.

    Argumenty:
    df (DataFrame): DataFrame do zapisania.
    path (str): Ścieżka do pliku CSV, do którego dane zostaną zapisane.
    """
    df.to_csv(path, index=False)
