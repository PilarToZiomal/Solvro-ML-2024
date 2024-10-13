import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def eda(cocktail_df):
    """
    Funkcja przeprowadzająca eksploracyjną analizę danych (EDA) na podstawie danych dotyczących koktajli.

    Argumenty:
    cocktail_df (DataFrame): DataFrame zawierający dane o koktajlach.

    Wykonuje:
    - Wyświetla informacje o DataFrame (typy danych, brakujące wartości itp.).
    - Wyświetla podstawowe statystyki opisowe dla danych liczbowych.
    - Wyświetla liczbę drinków w poszczególnych kategoriach.
    - Wyświetla liczbę drinków przypisanych do określonych typów szklanek.
    - Generuje wykresy (countplot) dla kategorii drinków oraz typów szklanek.
    """
    
    # Wyświetlanie podstawowych informacji o DataFrame
    print(cocktail_df.info())
    
    # Wyświetlanie statystyk opisowych
    print(cocktail_df.describe())
    
    # Wyświetlanie liczby drinków w poszczególnych kategoriach
    print(cocktail_df["category"].value_counts())
    
    # Wyświetlanie liczby drinków przypisanych do typów szklanek
    print(cocktail_df["glass"].value_counts())

    # Wykres liczby drinków w kategoriach
    plt.figure(figsize=(15, 5))
    ax = sns.countplot(y="category", data=cocktail_df)
    plt.title("Liczba drinków w poszczególnych kategoriach")

    # Dodanie wartości słupków na osi X
    for container in ax.containers:
        ax.bar_label(container, label_type='edge')

    plt.savefig('outputs/category_countplot.pdf', format='pdf')

    # Wykres typów szklanek
    plt.figure(figsize=(15, 5))
    ax = sns.countplot(y="glass", data=cocktail_df)
    plt.title("Typy szklanek używane w drinkach")

    # Dodanie wartości słupków na osi X
    for container in ax.containers:
        ax.bar_label(container, label_type='edge')

    plt.savefig('outputs/glass_countplot.pdf', format='pdf')
