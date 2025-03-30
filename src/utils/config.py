"""
Ρυθμίσεις για το project ανάλυσης συναισθημάτων
"""

import os
from pathlib import Path

# Βασικός φάκελος του project
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Φάκελοι για δεδομένα, μοντέλα και αποτελέσματα
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_FILE = DATA_DIR / 'Skroutz_dataset.xlsx'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = ROOT_DIR / 'models'

# Παράμετροι για την προεπεξεργασία
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Ρυθμίσεις για τα μοντέλα
MAX_FEATURES = 10000  # Μέγιστος αριθμός χαρακτηριστικών για το TF-IDF
MIN_DF = 5  # Ελάχιστη συχνότητα εμφάνισης όρων

# Παράμετροι για συναισθηματική ανάλυση
EMOTION_CATEGORIES = {
    "χαρά": ["χαρά", "ευχαρίστηση", "ικανοποίηση", "απόλαυση", "ευτυχία"],
    "εμπιστοσύνη": ["εμπιστοσύνη", "αξιοπιστία", "ασφάλεια", "σιγουριά"],
    "ανησυχία": ["ανησυχία", "άγχος", "στρες", "φόβος", "ανασφάλεια"],
    "θυμός": ["θυμός", "οργή", "αγανάκτηση", "εκνευρισμός", "απογοήτευση"],
    "λύπη": ["λύπη", "στεναχώρια", "θλίψη", "μελαγχολία"]
}

# Ρυθμίσεις για την διαδικτυακή εφαρμογή
FLASK_DEBUG = True
FLASK_PORT = 5000
FLASK_HOST = '0.0.0.0'

# Διασφάλιση ότι οι απαραίτητοι φάκελοι υπάρχουν
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True) 