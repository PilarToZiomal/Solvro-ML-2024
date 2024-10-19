# src/eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import ast
from wordcloud import WordCloud
import plotly.graph_objects as go
import warnings

from src.config import (
    OUTPUT_DIR,
    ALL_COCKTAILS_CSV,
    CATEGORY_CLASSES_PATH,
    GLASS_CLASSES_PATH,
    TAG_CLASSES_PATH,
    INGREDIENT_CLASSES_PATH
)

# Ignorowanie ostrzeżeń FutureWarning związanych z Seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)

# Konfiguracja stylu wykresów
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})  # Wyłączenie ostrzeżeń przy otwieraniu wielu wykresów

def check_files_exist(files):
    """
    Sprawdza, czy podane pliki istnieją.

    Parameters:
        files (list of Path): Lista ścieżek do plików.

    Returns:
        None
    """
    missing_files = []
    for file in files:
        if not file.exists():
            missing_files.append(file)
        else:
            print(f"Plik {file} znaleziony.")
    if missing_files:
        print("\nBrakujące pliki:")
        for file in missing_files:
            print(f"- {file}")
        print("\nUpewnij się, że wszystkie pliki są dostępne przed kontynuacją.")
        exit(1)  # Zakończenie skryptu, jeśli brakuje plików

def load_json(file_path):
    """
    Ładuje dane z pliku JSON.

    Parameters:
        file_path (Path): Ścieżka do pliku JSON.

    Returns:
        data (list): Lista elementów z pliku JSON.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def parse_tags(tags):
    """
    Przekształca wartość z kolumny 'tags' na listę tagów.

    Parameters:
        tags (str or list): Tagi jako lista lub ciąg znaków.

    Returns:
        list: Lista tagów.
    """
    if isinstance(tags, list):
        # Jeśli tagi są już listą
        return tags
    elif isinstance(tags, str):
        # Próba przekształcenia ciągu znaków reprezentującego listę na listę
        try:
            return ast.literal_eval(tags)
        except (ValueError, SyntaxError):
            # Jeśli nie jest to poprawny format listy, rozdziel tagi po przecinku
            return [tag.strip() for tag in tags.split(',')]
    else:
        # Jeśli tagi są w innym formacie, zwróć pustą listę
        return []

def parse_ingredients(ingredients):
    """
    Przekształca wartość z kolumny 'ingredients' na listę składników.

    Parameters:
        ingredients (str or list): Składniki jako lista lub ciąg znaków.

    Returns:
        list: Lista składników.
    """
    if isinstance(ingredients, list):
        # Jeśli składniki są już listą
        return ingredients
    elif isinstance(ingredients, str):
        # Rozdzielenie składników po przecinku
        return [ing.strip().lower() for ing in ingredients.split(',')]
    else:
        # Jeśli składniki są w innym formacie, zwróć pustą listę
        return []

def count_tags(tag_vector):
    """
    Liczy indeksy tagów, które są oznaczone jako 1 w wektorze.

    Parameters:
        tag_vector (list of int): Wektor tagów (0 lub 1).

    Returns:
        list of int: Lista indeksów tagów obecnych w koktajlu.
    """
    return [i for i, bit in enumerate(tag_vector) if bit == 1]

def count_ingredients(ingredient_vector):
    """
    Liczy indeksy składników, które są oznaczone jako 1 w wektorze.

    Parameters:
        ingredient_vector (list of int): Wektor składników (0 lub 1).

    Returns:
        list of int: Lista indeksów składników obecnych w koktajlu.
    """
    return [i for i, bit in enumerate(ingredient_vector) if bit == 1]

def main():
    # Krok 1: Sprawdzenie istnienia plików
    print("Sprawdzanie istnienia plików...")
    files_to_check = [
        ALL_COCKTAILS_CSV,
        CATEGORY_CLASSES_PATH,
        GLASS_CLASSES_PATH,
        TAG_CLASSES_PATH,
        INGREDIENT_CLASSES_PATH
    ]
    check_files_exist(files_to_check)
    print("\nWszystkie wymagane pliki są dostępne.\n")

    # Krok 2: Ładowanie danych
    print("Ładowanie danych...")
    df_all = pd.read_csv(ALL_COCKTAILS_CSV)
    print("Dane zostały pomyślnie załadowane.\n")

    # Krok 3: Ładowanie klas kategorii, typów szkła, tagów i składników
    print("Ładowanie klas kategorii, typów szkła, tagów i składników...")
    category_classes = load_json(CATEGORY_CLASSES_PATH) if CATEGORY_CLASSES_PATH.exists() else []
    if not category_classes:
        print("Plik category_classes.json nie istnieje lub jest pusty.")

    glass_classes = load_json(GLASS_CLASSES_PATH) if GLASS_CLASSES_PATH.exists() else []
    if not glass_classes:
        print("Plik glass_classes.json nie istnieje lub jest pusty.")

    tag_classes = load_json(TAG_CLASSES_PATH) if TAG_CLASSES_PATH.exists() else []
    if not tag_classes:
        print("Plik tag_classes.json nie istnieje lub jest pusty.")

    ingredient_classes = load_json(INGREDIENT_CLASSES_PATH) if INGREDIENT_CLASSES_PATH.exists() else []
    if not ingredient_classes:
        print("Plik ingredient_classes.json nie istnieje lub jest pusty.")
    print("Ładowanie klas zakończone.\n")

    # Krok 4: Wyświetlenie pierwszych kilku wierszy
    print("Wyświetlanie pierwszych kilku wierszy danych:")
    print(df_all.head(), "\n")

    # Krok 5: Informacje o DataFrame
    print("Informacje o DataFrame:")
    print(df_all.info(), "\n")

    # Krok 6: Statystyki opisowe dla kolumn numerycznych
    print("Statystyki opisowe dla kolumn numerycznych:")
    print(df_all.describe(), "\n")

    # Krok 7: Analiza kategorii koktajli
    print("Analiza kategorii koktajli...")
    category_counts = df_all['category'].value_counts().sort_values(ascending=False)

    # Tworzenie DataFrame z kategoriami
    categories_df = pd.DataFrame({
        'category': category_counts.index,
        'count': category_counts.values
    })

    # Wyświetlenie tabeli
    print("Liczba koktajli w każdej kategorii:")
    print(categories_df, "\n")

    # Wykres słupkowy kategorii
    print("Generowanie wykresu słupkowego kategorii...")
    plt.figure(figsize=(10,6))
    sns.barplot(x='count', y='category', data=categories_df, palette='viridis')  # Użycie 'palette' zamiast 'color'
    plt.title('Liczba Koktajli w Każdej Kategorii')
    plt.xlabel('Liczba Koktajli')
    plt.ylabel('Kategoria')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'category_counts_barplot.png')
    plt.close()
    print("Wykres słupkowy kategorii zapisany jako 'category_counts_barplot.png'.\n")

    # Krok 8: Analiza typów szkła
    print("Analiza typów szkła...")
    glass_counts = df_all['glass'].value_counts().sort_values(ascending=False)

    # Tworzenie DataFrame z typami szkła
    glasses_df = pd.DataFrame({
        'glass': glass_counts.index,
        'count': glass_counts.values
    })

    # Wyświetlenie tabeli
    print("Liczba koktajli w każdym typie szkła:")
    print(glasses_df, "\n")

    # Wykres słupkowy typów szkła z poprawionym parametrem 'palette'
    print("Generowanie wykresu słupkowego typów szkła...")
    plt.figure(figsize=(10,6))
    sns.barplot(x='count', y='glass', data=glasses_df, palette='magma')  # Użycie 'palette' zamiast 'color'
    plt.title('Liczba Koktajli w Każdym Typie Szkła')
    plt.xlabel('Liczba Koktajli')
    plt.ylabel('Typ Szkła')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'glass_counts_barplot.png')
    plt.close()
    print("Wykres słupkowy typów szkła zapisany jako 'glass_counts_barplot.png'.\n")

    # Krok 9: Przekształcanie kolumny 'tags' na listy
    print("Przekształcanie kolumny 'tags' na listy...")
    df_all['tags'] = df_all['tags'].apply(parse_tags)

    # Zastąpienie NaN pustymi listami
    df_all['tags'] = df_all['tags'].apply(lambda x: x if isinstance(x, list) else [])

    # Sprawdzenie, czy wszystkie wartości są listami
    print("Sprawdzenie typów w kolumnie 'tags':")
    print(df_all['tags'].apply(type).value_counts(), "\n")

    # Krok 10: Zliczanie wszystkich tagów
    print("Zliczanie częstotliwości tagów...")
    all_tags = df_all['tags'].explode()
    all_tags = all_tags.dropna()

    # Zliczanie wystąpień każdego tagu
    tag_counts = all_tags.value_counts().sort_values(ascending=False)

    # Sprawdzenie typu tag_counts.index
    if not tag_counts.empty:
        print("Typ elementów w tag_counts.index:", type(tag_counts.index[0]))
    else:
        print("tag_counts jest pusty.")

    # Poprawione mapowanie tagów (bez używania tag_classes)
    tags_df = pd.DataFrame({
        'tag': tag_counts.index,  # Użyj tagów bez mapowania
        'count': tag_counts.values
    })

    # Wyświetlenie 10 najczęstszych tagów
    print("10 najczęstszych tagów:")
    print(tags_df.head(10), "\n")

    # Krok 11: Wykres 10 najpopularniejszych tagów z poprawionym parametrem 'palette'
    print("Generowanie wykresu 10 najpopularniejszych tagów...")
    plt.figure(figsize=(10,6))
    sns.barplot(x='count', y='tag', data=tags_df.head(10), palette='coolwarm')  # Użycie 'palette' zamiast 'color'
    plt.title('10 Najpopularniejszych Tagów')
    plt.xlabel('Liczba Koktajli')
    plt.ylabel('Tag')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_10_tags_barplot.png')
    plt.close()
    print("Wykres 10 najpopularniejszych tagów zapisany jako 'top_10_tags_barplot.png'.\n")

    # Krok 12: Przekształcanie kolumny 'ingredient_vector' na listy
    print("Przekształcanie kolumny 'ingredient_vector' na listy...")
    df_all['ingredient_vector'] = df_all['ingredient_vector'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df_all['ingredient_vector'] = df_all['ingredient_vector'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Sprawdzenie typów w kolumnie 'ingredient_vector'
    print("Sprawdzenie typów w kolumnie 'ingredient_vector':")
    print(df_all['ingredient_vector'].apply(type).value_counts(), "\n")

    # Krok 13: Zliczanie wszystkich składników
    print("Zliczanie częstotliwości składników...")
    all_ingredients = df_all['ingredient_vector'].apply(count_ingredients).explode()
    all_ingredients = all_ingredients.dropna().astype(int)

    # Zliczanie wystąpień każdego składnika
    ingredient_counts = all_ingredients.value_counts().sort_values(ascending=False)

    # Sprawdzenie typu ingredient_counts.index
    if not ingredient_counts.empty:
        print("Typ elementów w ingredient_counts.index:", type(ingredient_counts.index[0]))
    else:
        print("ingredient_counts jest pusty.")

    # Poprawione mapowanie składników (zamiana indeksów na nazwy)
    ingredients_df = pd.DataFrame({
        'ingredient': [
            ingredient_classes[i] if isinstance(i, int) and i < len(ingredient_classes) else f"Unknown Ingredient {i}"
            for i in ingredient_counts.index
        ],
        'count': ingredient_counts.values
    })

    # Wyświetlenie 10 najczęstszych składników
    print("10 najczęstszych składników:")
    print(ingredients_df.head(10), "\n")

    # Krok 14: Wykres 10 najpopularniejszych składników
    print("Generowanie wykresu 10 najpopularniejszych składników...")
    plt.figure(figsize=(10,6))
    sns.barplot(x='count', y='ingredient', data=ingredients_df.head(10), palette='Blues_d')  # Użycie 'palette' zamiast 'color'
    plt.title('10 Najpopularniejszych Składników')
    plt.xlabel('Liczba Koktajli')
    plt.ylabel('Składnik')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_10_ingredients_barplot.png')
    plt.close()
    print("Wykres 10 najpopularniejszych składników zapisany jako 'top_10_ingredients_barplot.png'.\n")

    # Krok 15: Tworzenie pivot table: Kategoria vs Typ Szkła
    print("Tworzenie pivot table: Kategoria vs Typ Szkła...")
    pivot_table_named = pd.crosstab(df_all['category'], df_all['glass'])
    print("Pivot table utworzony. Przykładowe dane:")
    print(pivot_table_named.head(), "\n")

    # Krok 16: Generowanie heatmapy relacji kategorii z typem szkła
    print("Generowanie heatmapy relacji kategorii z typem szkła...")
    plt.figure(figsize=(12,8))
    sns.heatmap(pivot_table_named, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Relacja Kategorii Koktajli z Typami Szkła')
    plt.xlabel('Typ Szkła')
    plt.ylabel('Kategoria')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'category_glass_heatmap.png')
    plt.close()
    print("Heatmapa relacji kategorii z typami szkła zapisana jako 'category_glass_heatmap.png'.\n")

    # Krok 17: Obliczanie korelacji
    print("Obliczanie korelacji między normami wektorów...")
    norm_columns = ['tag_vector_norm', 'ingredient_vector_norm', 'category_vector_norm', 'glass_vector_norm']
    existing_norm_columns = [col for col in norm_columns if col in df_all.columns]
    if len(existing_norm_columns) == len(norm_columns):
        correlation = df_all[norm_columns].corr()
        print("Macierz korelacji:")
        print(correlation, "\n")

        # Zapisanie macierzy korelacji do CSV
        correlation.to_csv(OUTPUT_DIR / 'vector_norm_correlation.csv')
        print("Macierz korelacji zapisana jako 'vector_norm_correlation.csv'.\n")

        # Wykres korelacji
        print("Generowanie wykresu korelacji...")
        plt.figure(figsize=(6,4))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Korelacja między Normami Wektorów')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'vector_norm_correlation_heatmap.png')
        plt.close()
        print("Wykres korelacji zapisany jako 'vector_norm_correlation_heatmap.png'.\n")
    else:
        missing_cols = set(norm_columns) - set(existing_norm_columns)
        print(f"Niektóre wymagane kolumny do obliczenia korelacji nie istnieją: {missing_cols}\n")

    # Krok 18: Histogram dla 'tag_vector_norm'
    if 'tag_vector_norm' in df_all.columns:
        print("Generowanie histogramu dla 'tag_vector_norm'...")
        plt.figure(figsize=(10,6))
        sns.histplot(df_all['tag_vector_norm'], bins=30, kde=True, color='skyblue')
        plt.title('Rozkład Normy Tag Vector')
        plt.xlabel('Norma Tag Vector')
        plt.ylabel('Liczba Koktajli')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'tag_vector_norm_histogram.png')
        plt.close()
        print("Histogram dla 'tag_vector_norm' zapisany jako 'tag_vector_norm_histogram.png'.\n")
    else:
        print("Kolumna 'tag_vector_norm' nie istnieje w DataFrame. Pomijanie histogramu.\n")

    # Krok 19: Histogram dla 'ingredient_vector_norm'
    if 'ingredient_vector_norm' in df_all.columns:
        print("Generowanie histogramu dla 'ingredient_vector_norm'...")
        plt.figure(figsize=(10,6))
        sns.histplot(df_all['ingredient_vector_norm'], bins=30, kde=True, color='salmon')
        plt.title('Rozkład Normy Ingredient Vector')
        plt.xlabel('Norma Ingredient Vector')
        plt.ylabel('Liczba Koktajli')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'ingredient_vector_norm_histogram.png')
        plt.close()
        print("Histogram dla 'ingredient_vector_norm' zapisany jako 'ingredient_vector_norm_histogram.png'.\n")
    else:
        print("Kolumna 'ingredient_vector_norm' nie istnieje w DataFrame. Pomijanie histogramu.\n")

    # Krok 20: Wykres pudełkowy dla norm wektorów
    print("Generowanie wykresu pudełkowego dla norm wektorów...")
    norm_columns = ['tag_vector_norm', 'ingredient_vector_norm']
    existing_norm_columns = [col for col in norm_columns if col in df_all.columns]
    if existing_norm_columns:
        plt.figure(figsize=(12,6))
        sns.boxplot(data=df_all[existing_norm_columns], palette='Set2')
        plt.title('Porównanie Norm Wektorów')
        plt.xlabel('Wektor')
        plt.ylabel('Norma L2')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'vector_norm_boxplot.png')
        plt.close()
        print("Wykres pudełkowy dla norm wektorów zapisany jako 'vector_norm_boxplot.png'.\n")
    else:
        print(f"Żadne z wymaganych kolumn do wykresu pudełkowego nie istnieją: {norm_columns}\n")

    # Krok 21: Word Cloud dla najpopularniejszych składników
    print("Generowanie Word Cloud dla najpopularniejszych składników...")
    ingredient_counts_dict = ingredients_df.set_index('ingredient')['count'].to_dict()

    # Sprawdzenie, czy ingredient_counts_dict zawiera tylko stringi jako klucze
    if all(isinstance(key, str) for key in ingredient_counts_dict.keys()):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ingredient_counts_dict)
        plt.figure(figsize=(15,7.5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud Najpopularniejszych Składników Koktajli')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'ingredients_wordcloud.png')
        plt.close()
        print("Word Cloud zapisany jako 'ingredients_wordcloud.png'.\n")
    else:
        print("Błąd: Słownik ingredient_counts_dict zawiera nie-stringowe klucze. Word Cloud nie został wygenerowany.\n")

    # Krok 22: Tworzenie wykresu Sankey pokazującego relacje między kategoriami a typami szkła
    print("Generowanie wykresu Sankey relacji między kategoriami a typami szkła...")
    # Przygotowanie etykiet
    category_labels = category_classes
    glass_labels = glass_classes
    all_labels = category_labels + glass_labels

    # Tworzenie mapowania
    category_indices = {category: i for i, category in enumerate(category_labels)}
    glass_indices = {glass: i + len(category_labels) for i, glass in enumerate(glass_labels)}

    # Przygotowanie źródeł, celów i wartości
    sources = []
    targets = []
    values = []

    for category in category_labels:
        for glass in glass_labels:
            if category in pivot_table_named.index and glass in pivot_table_named.columns:
                count = pivot_table_named.at[category, glass]
                if count > 0:
                    sources.append(category_indices[category])
                    targets.append(glass_indices[glass])
                    values.append(count)

    # Tworzenie wykresu Sankey
    sankey = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color="blue"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )

    fig = go.Figure(data=[sankey])
    fig.update_layout(title_text="Relacje Kategorii Koktajli z Typami Szkła", font_size=10)
    fig.write_html(OUTPUT_DIR / 'category_glass_sankey.html')
    print("Wykres Sankey zapisany jako 'category_glass_sankey.html'.\n")

    # Krok 23: Zapisanie wyników zliczeń tagów i składników do CSV
    print("Zapisanie wyników zliczeń tagów i składników do plików CSV...")
    tags_df.to_csv(OUTPUT_DIR / 'tag_counts.csv', index=False)
    print("Zapisano 'tag_counts.csv' w katalogu 'outputs'.")
    
    ingredients_df.to_csv(OUTPUT_DIR / 'ingredient_counts.csv', index=False)
    print("Zapisano 'ingredient_counts.csv' w katalogu 'outputs'.\n")

    # Krok 24: Zakończenie skryptu
    print("EDA zakończona pomyślnie. Wszystkie wyniki zostały zapisane w katalogu 'outputs'.")

if __name__ == "__main__":
    main()
