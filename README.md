# Analiza i Klasteryzacja Danych Cocktaili

## Opis
Projekt dotyczy eksploracyjnej analizy danych (EDA) oraz klasteryzacji drinków. 
Dane pochodzą z pliku `cocktail_dataset.json`, a analiza i klasteryzacja wykonane zostały w Pythonie z użyciem bibliotek takich jak `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

## Instalacja
Aby uruchomić projekt, należy zainstalować wymagane biblioteki:

```sh
pip install -r requirements.txt
```

## Uruchomienie
Aby powtórzyć eksperymenty:
1. Upewnij się, że posiadasz plik `cocktail_dataset.json`.
2. Uruchom skrypt `main.py`, aby przeprowadzić klasteryzację i zapisać wyniki.
3. Wyniki wizualne można przeglądać w pliku .pdf w folderze `notebooks/`.

## Struktura projektu
- `src/` - Folder z kodem źródłowym:
  - `eda.py` - Moduł do eksploracyjnej analizy danych.
  - `preprocessing.py` - Moduł do przetwarzania danych, zawiera kod do czyszczenia i przygotowywania danych.
  - `clustering.py` - Moduł do klasteryzacji danych, np. KMeans.
  - `utils.py` - Moduł pomocniczy z funkcjami wykorzystywanymi w różnych częściach projektu.
  - `main.py` - Główny skrypt uruchamiający cały pipeline od EDA po klasteryzację.

- `notebooks/` - Folder z plikiem Latex do wizualizacji:
  - `SolvroProjekt.pdf` - Notebook do prezentacji wyników EDA i klasteryzacji.

- `data/` - Folder z danymi wejściowymi:
  - `cocktail_dataset.json` - Plik z danymi wejściowymi do analizy.

- `outputs/` - Folder z wynikami:
  - `clustered_cocktails.csv` - Plik CSV z wynikami klasteryzacji.

- `requirements.txt` - Plik z listą zależności potrzebnych do uruchomienia projektu.

- `README.md` - Instrukcja uruchomienia projektu.

## Uwagi
- Większość docstringów została napisana przy użyciu ChatGPT.com
