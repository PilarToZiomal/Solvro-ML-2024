# Cocktail Clustering Project

## Opis projektu

Ten projekt ma na celu analizę i klasteryzację koktajli przy użyciu algorytmów uczenia maszynowego, takich jak K-means, oraz wizualizację wyników za pomocą technik redukcji wymiarów (PCA i UMAP). Projekt zawiera również przetwarzanie danych, eksploracyjną analizę danych (EDA) oraz generowanie różnorodnych wykresów i metryk jakości klasteryzacji.

## Struktura katalogów

- **data/** - Zawiera zestaw danych `cocktail_dataset.json` z danymi o koktajlach.
- **notebooks/** - Zawiera notebooki Jupyter z analizą kroków.
- **outputs/** - Zawiera pliki wyjściowe generowane przez skrypty, takie jak wykresy, CSV, raporty EDA.
- **src/** - Zawiera pliki źródłowe Pythona do przetwarzania, analizy i klasteryzacji danych:
  - `preproc.py` - Przetwarzanie danych wejściowych.
  - `eda.py` - Eksploracyjna analiza danych (EDA).
  - `clustering.py` - Klasteryzacja danych za pomocą K-means i wizualizacja wyników.
  - `config.py` - Konfiguracja projektu, zawiera ścieżki do plików danych i wyjściowych.
  - `utils.py` - Dodatkowe funkcje pomocnicze, np. do ustawienia logowania.

## Instrukcja uruchamiania projektu - Cocktail Clustering Project

## 1. Instalacja zależności

Przed uruchomieniem projektu należy zainstalować wymagane biblioteki. Możesz to zrobić, tworząc wirtualne środowisko i instalując zależności z pliku `requirements.txt`.

### a. Stwórz i aktywuj wirtualne środowisko (opcjonalnie)

Dla lepszej organizacji zależności zaleca się stworzenie wirtualnego środowiska.

#### W systemie Linux/MacOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
#### W systemie Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

### b. Zainstaluj zależności

Po aktywowaniu wirtualnego środowiska, zainstaluj wymagane biblioteki:

```bash
pip install -r requirements.txt
```

## 2. Uruchomienie projektu

Projekt można uruchomić za pomocą pliku `main.py`, który automatycznie przetworzy dane, przeprowadzi analizę EDA oraz wykona klasteryzację.

## Komenda uruchamiająca główny skrypt:

```bash
python main.py
```

## 3. Uwagi

- Notebooki Jupyter nie zapisują żadnych plików, służą jedynie do eksploracji danych i analizy.
- W folderze `outputs` znajdują się już przetworzone dane. Możesz je usunąć, aby wygenerować nowe wyniki, uruchamiając ponownie skrypt `main.py`.
- Upewnij się, że wszystkie wymagane pliki znajdują się w odpowiednich folderach, jak pokazano w strukturze projektu.
- Po zakończeniu analizy i klasteryzacji wyniki zostaną zapisane automatycznie w folderze `outputs`.

