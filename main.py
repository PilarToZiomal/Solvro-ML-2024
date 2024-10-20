# main.py

from src.preproc import main as preproc_main
from src.eda import main as eda_main
from src.clustering import main as clustering_main

def main():
    """Główny punkt wejścia do projektu."""
    print("Rozpoczynanie przetwarzania danych...")
    preproc_main()
    print("\nPrzetwarzanie danych zakończone.\n")
    
    print("Rozpoczynanie analizy EDA...")
    eda_main()
    print("\nAnaliza EDA zakończona.")

    print("Rozpoczynanie klasteryzacji danych...")
    clustering_main()
    print("\nKlasteryzacja zakończona.")

if __name__ == "__main__":
    main()
