import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_data(cocktail_df):
    """
    Funkcja przetwarzająca dane dotyczące koktajli.

    Argumenty:
    cocktail_df (DataFrame): DataFrame zawierający dane o koktajlach.

    Zwraca:
    tuple:
        - X_scaled (ndarray): Znormalizowane dane wejściowe, gotowe do klasteryzacji lub innych algorytmów uczenia maszynowego.
        - prepared_df (DataFrame): DataFrame z oryginalnymi danymi, a także z dodatkowymi kolumnami, takimi jak liczba składników
          oraz zakodowane kategorie koktajli.
    """
    
    # Dodanie kolumny z liczbą składników
    cocktail_df["num_ingredients"] = cocktail_df["ingredients"].apply(len)

    # One-Hot Encoding dla kategorii koktajli
    encoder = OneHotEncoder()
    encoded_category = encoder.fit_transform(cocktail_df[["category"]]).toarray()
    categories_df = pd.DataFrame(
        encoded_category, columns=encoder.get_feature_names_out(["category"])
    )

    # Połączenie oryginalnych danych z zakodowanymi kategoriami
    prepared_df = pd.concat([cocktail_df, categories_df], axis=1)

    # Wybranie cech do modelowania
    features = ["num_ingredients"] + list(categories_df.columns)
    X = prepared_df[features]

    # Skalowanie cech
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, prepared_df
