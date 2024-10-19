# src/preproc.py

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import (
    DATA_PATH,
    OUTPUT_DIR,
    PREPROCESSED_CSV,
    ALL_COCKTAILS_CSV,
    CATEGORY_CLASSES_PATH,
    GLASS_CLASSES_PATH,
    TAG_CLASSES_PATH,
    INGREDIENT_CLASSES_PATH,
    COLUMNS_TO_KEEP
)


def setup_logging():
    """Konfiguracja logowania."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def load_data(file_path: Path) -> list:
    """Wczytuje dane z pliku JSON."""
    logging.info(f"Ładowanie danych z {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"Znaleziono {len(data)} wpisów.")
    return data


def preprocess_entry(entry: dict, excluded_fields: list) -> dict:
    """Czyszczenie pojedynczego wpisu."""
    for field in excluded_fields:
        entry.pop(field, None)

    # Upewnij się, że 'tags' jest zawsze listą
    tags = entry.get('tags')
    if not isinstance(tags, list):
        logging.warning(f"Nieoczekiwany format 'tags' dla wpisu '{entry.get('name', 'Unknown')}': {tags}")
        tags = [] if tags is None else [tags] if isinstance(tags, str) else []
    entry['tags'] = tags

    # Upewnij się, że 'ingredients' jest zawsze listą nazw
    ingredients = entry.get('ingredients')
    if not isinstance(ingredients, list):
        logging.warning(f"Nieoczekiwany format 'ingredients' dla wpisu '{entry.get('name', 'Unknown')}': {ingredients}")
        ingredients = []
    else:
        ingredients = [ingredient.get('name', '') for ingredient in ingredients if isinstance(ingredient, dict)]
    entry['ingredients'] = ingredients

    return entry


def transform_tags_ingredients(df: pd.DataFrame) -> tuple:
    """Transformuje tagi i składniki na wektory binarne."""
    logging.info("Transformacja 'tags' i 'ingredients' na wektory binarne...")

    # Transformacja 'tags'
    mlb_tags = MultiLabelBinarizer()
    tag_matrix = mlb_tags.fit_transform(df['tags'])
    tag_classes = mlb_tags.classes_.tolist()
    tag_columns = [f"tag_{tag}" for tag in tag_classes]
    df_tags = pd.DataFrame(tag_matrix, columns=tag_columns, index=df.index)
    logging.info(f"Stworzono {len(tag_columns)} kolumn dla tagów.")

    # Transformacja 'ingredients'
    mlb_ingredients = MultiLabelBinarizer()
    ingredient_matrix = mlb_ingredients.fit_transform(df['ingredients'])
    ingredient_classes = mlb_ingredients.classes_.tolist()
    ingredient_columns = [f"ingredient_{ingredient}" for ingredient in ingredient_classes]
    df_ingredients = pd.DataFrame(ingredient_matrix, columns=ingredient_columns, index=df.index)
    logging.info(f"Stworzono {len(ingredient_columns)} kolumn dla składników.")

    # Łączenie DataFrame z wektorami tagów i składników
    df_final = pd.concat([df, df_tags, df_ingredients], axis=1)

    # Tworzenie 'tag_vector' i 'ingredient_vector' jako listy binarne
    df_final['tag_vector'] = df_tags.values.tolist()
    df_final['ingredient_vector'] = df_ingredients.values.tolist()

    logging.info("Dodano kolumny 'tag_vector' i 'ingredient_vector'.")

    # Zapisanie klas tagów i składników
    with open(TAG_CLASSES_PATH, 'w', encoding='utf-8') as f:
        json.dump(tag_classes, f, ensure_ascii=False, indent=4)
    logging.info(f"Zapisano {TAG_CLASSES_PATH.name} z {len(tag_classes)} klasami tagów.")

    with open(INGREDIENT_CLASSES_PATH, 'w', encoding='utf-8') as f:
        json.dump(ingredient_classes, f, ensure_ascii=False, indent=4)
    logging.info(f"Zapisano {INGREDIENT_CLASSES_PATH.name} z {len(ingredient_classes)} klasami składników.")

    # Usunięcie oryginalnych kolumn 'tags' i 'ingredients' oraz poszczególnych tag_* i ingredient_* kolumn
    df_final.drop(['tags', 'ingredients'] + tag_columns + ingredient_columns, axis=1, inplace=True)

    logging.info("Usunięto oryginalne i poszczególne kolumny tagów oraz składników.")

    return df_final, mlb_tags, mlb_ingredients


def one_hot_encode(df: pd.DataFrame, column: str, output_column: str, classes_json_path: Path) -> tuple:
    """One-Hot Encoding dla określonej kolumny i zapis klas do pliku JSON."""
    logging.info(f"One-Hot Encoding dla kolumny '{column}'...")
    unique_classes = sorted(df[column].unique())
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
    df[output_column] = df[column].map(class_mapping)
    
    # Zapis klas do pliku JSON
    with open(classes_json_path, 'w', encoding='utf-8') as f:
        json.dump(unique_classes, f, ensure_ascii=False, indent=4)
    logging.info(f"Zapisano {classes_json_path.name} z {len(unique_classes)} klasami.")
    
    return df, unique_classes


def calculate_norms(df: pd.DataFrame, vector_columns: list) -> pd.DataFrame:
    """Oblicza normę L2 dla wybranych kolumn wektorowych."""
    logging.info("Obliczanie normy wektorów...")
    
    for column in vector_columns:
        norm_column = f"{column}_norm"
        logging.info(f"Obliczanie normy dla '{column}'...")
        df[norm_column] = df[column].apply(lambda x: np.linalg.norm(x))
    
    logging.info("Obliczanie norm zakończone.")
    return df


def reduce_columns_and_save(df: pd.DataFrame, columns_to_keep: list, output_path: Path):
    """Redukuje DataFrame do wybranych kolumn i zapisuje do pliku CSV."""
    logging.info(f"Redukcja kolumn do {len(columns_to_keep)} wybranych cech...")
    df_reduced = df[columns_to_keep]
    df_reduced.to_csv(output_path, index=False)
    logging.info(f"Zapis przetworzonych danych do {output_path} zakończony pomyślnie.")


def save_all_cocktails(df_original: pd.DataFrame, df_processed: pd.DataFrame, output_path: Path):
    """Zapisuje wszystkie dane (oryginalne i przetworzone) do osobnego pliku CSV bez duplikowania kolumn."""
    logging.info(f"Zapis wszystkich danych do {output_path}...")
    
    # Identyfikacja kolumn, które są już w df_original i powinny zostać usunięte z df_processed
    overlapping_columns = df_original.columns.intersection(df_processed.columns).tolist()
    
    if overlapping_columns:
        logging.info(f"Usuwanie kolumn {overlapping_columns} z df_processed, aby uniknąć duplikatów.")
        df_processed = df_processed.drop(columns=overlapping_columns)
    
    # Łączenie DataFrame'ów
    df_all = pd.concat([df_original.reset_index(drop=True), df_processed.reset_index(drop=True)], axis=1)
    
    # Zapis do pliku
    df_all.to_csv(output_path, index=False)
    logging.info("Zapis wszystkich danych zakończony pomyślnie.")


def main():
    """Główna funkcja przetwarzania danych."""
    setup_logging()
    
    # Ładowanie danych
    data = load_data(DATA_PATH)
    
    # Definiowanie pól do wykluczenia
    EXCLUDED_FIELDS = ['createdAt', 'updatedAt', 'instructions', 'imageUrl', 'id']
    
    # Czyszczenie danych
    logging.info("Czyszczenie danych...")
    processed_data = [preprocess_entry(entry, EXCLUDED_FIELDS) for entry in data]
    df_clean = pd.DataFrame(processed_data)
    logging.info("Czyszczenie danych zakończone.")
    
    # Zachowanie oryginalnych danych przed przetworzeniem (do all_cocktails.csv)
    df_original = df_clean[['name', 'category', 'glass', 'alcoholic', 'tags', 'ingredients']].copy()
    
    # Transformacja 'tags' i 'ingredients' na wektory binarne
    df_processed, mlb_tags, mlb_ingredients = transform_tags_ingredients(df_clean)
    
    # One-Hot Encoding dla 'category'
    df_processed, category_classes = one_hot_encode(
        df=df_processed,
        column='category',
        output_column='category_vector',
        classes_json_path=CATEGORY_CLASSES_PATH
    )
    
    # One-Hot Encoding dla 'glass'
    df_processed, glass_classes = one_hot_encode(
        df=df_processed,
        column='glass',
        output_column='glass_vector',
        classes_json_path=GLASS_CLASSES_PATH
    )
    
    # Obliczanie normy wektorów
    vector_columns = ['tag_vector', 'ingredient_vector', 'category_vector', 'glass_vector']
    df_processed = calculate_norms(df_processed, vector_columns)
    
    # Redukcja kolumn i zapis preprocessed_cocktails.csv
    reduce_columns_and_save(df_processed, COLUMNS_TO_KEEP, PREPROCESSED_CSV)
    
    # Zapis wszystkich danych do all_cocktails.csv
    save_all_cocktails(df_original, df_processed, ALL_COCKTAILS_CSV)
    
    logging.info("Proces przetwarzania danych zakończony pomyślnie.")


if __name__ == "__main__":
    main()
