# src/utils.py

import logging
import sys

def setup_logging():
    """Ustawia konfigurację logowania."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
