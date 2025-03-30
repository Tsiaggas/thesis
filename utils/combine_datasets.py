"""
Script για τον συνδυασμό του αυξημένου dataset Skroutz (θετικό/αρνητικό)
με ένα νέο dataset ουδέτερων κειμένων, δημιουργώντας ένα ενιαίο dataset 3 κλάσεων.
"""

import pandas as pd
import os
import numpy as np

# --- Διαδρομές Αρχείων (χωρίς ../) ---
AUGMENTED_DATASET_PATH = os.path.join('data', 'processed', 'augmented_skroutz_dataset.csv')
NEUTRAL_DATASET_PATH = os.path.join('data', 'neutral_texts.csv')
OUTPUT_DATASET_PATH = os.path.join('data', 'processed', 'skroutz_3class_dataset.csv')

# --- Ονόματα Στηλών & Ετικέτες ---
TEXT_COL = 'text'
LABEL_COL = 'label' # Το τελικό όνομα της στήλης ετικετών
SENTIMENT_COL_AUGMENTED = 'sentiment' # Όνομα στήλης στο αυξημένο (0/1)
SENTIMENT_COL_NEUTRAL = 'sentiment' # Όνομα στήλης στο ουδέτερο (π.χ., 'neutral')

# Αριθμητικές ετικέτες για το τελικό dataset
NEG_LABEL = 0
POS_LABEL = 1
NEU_LABEL = 2

def combine_datasets():
    """Φορτώνει, επεξεργάζεται και συνδυάζει τα datasets."""
    print("--- Έναρξη συνδυασμού datasets ---")

    # 1. Φόρτωση Αυξημένου Dataset (Θετικό/Αρνητικό)
    try:
        print(f"Φόρτωση θετικών/αρνητικών από: {AUGMENTED_DATASET_PATH}")
        df_augmented = pd.read_csv(AUGMENTED_DATASET_PATH)
        # Έλεγχος στηλών
        if TEXT_COL not in df_augmented.columns or SENTIMENT_COL_AUGMENTED not in df_augmented.columns:
             raise ValueError(f"Το αυξημένο dataset πρέπει να έχει στήλες '{TEXT_COL}' και '{SENTIMENT_COL_AUGMENTED}'. Βρέθηκαν: {df_augmented.columns.tolist()}")
        # Κράτημα απαραίτητων στηλών και μετονομασία label
        df_augmented = df_augmented[[TEXT_COL, SENTIMENT_COL_AUGMENTED]].rename(columns={SENTIMENT_COL_AUGMENTED: LABEL_COL})
        # Έλεγχος ότι τα labels είναι 0 και 1
        if not all(label in [NEG_LABEL, POS_LABEL] for label in df_augmented[LABEL_COL].unique()):
            print(f"Προειδοποίηση: Βρέθηκαν μη αναμενόμενες ετικέτες στο αυξημένο dataset: {df_augmented[LABEL_COL].unique()}. Κρατάμε μόνο 0 και 1.")
            df_augmented = df_augmented[df_augmented[LABEL_COL].isin([NEG_LABEL, POS_LABEL])]
        print(f"Φορτώθηκαν {len(df_augmented)} θετικές/αρνητικές εγγραφές.")
    except FileNotFoundError:
        print(f"Σφάλμα: Το αρχείο {AUGMENTED_DATASET_PATH} δεν βρέθηκε.")
        return
    except Exception as e:
        print(f"Σφάλμα κατά την επεξεργασία του αυξημένου dataset: {e}")
        return

    # 2. Φόρτωση Ουδέτερου Dataset
    try:
        print(f"Φόρτωση ουδέτερων από: {NEUTRAL_DATASET_PATH}")
        df_neutral = pd.read_csv(NEUTRAL_DATASET_PATH)
         # Έλεγχος στήλης κειμένου
        if TEXT_COL not in df_neutral.columns:
            # Προσπάθεια εύρεσης εναλλακτικής (π.χ., 'review_text')
            found_text_col = None
            potential_cols = ['review_text', 'sentence', 'neutral_text']
            for col in potential_cols:
                if col in df_neutral.columns:
                    df_neutral = df_neutral.rename(columns={col: TEXT_COL})
                    found_text_col = col
                    print(f"  (Χρησιμοποιήθηκε η στήλη '{col}' ως '{TEXT_COL}')")
                    break
            if not found_text_col:
                 raise ValueError(f"Το ουδέτερο dataset πρέπει να έχει στήλη κειμένου (π.χ., '{TEXT_COL}'). Βρέθηκαν: {df_neutral.columns.tolist()}")

        # Κράτημα μόνο της στήλης κειμένου και προσθήκη του label 2
        df_neutral = df_neutral[[TEXT_COL]].copy()
        df_neutral[LABEL_COL] = NEU_LABEL
        print(f"Φορτώθηκαν {len(df_neutral)} ουδέτερες εγγραφές.")
    except FileNotFoundError:
        print(f"Σφάλμα: Το αρχείο {NEUTRAL_DATASET_PATH} δεν βρέθηκε.")
        return
    except Exception as e:
        print(f"Σφάλμα κατά την επεξεργασία του ουδέτερου dataset: {e}")
        return

    # 3. Συνδυασμός των DataFrames
    print("Συνδυασμός datasets...")
    df_combined = pd.concat([df_augmented, df_neutral], ignore_index=True)
    print(f"Μέγεθος πριν την αφαίρεση διπλοτύπων: {len(df_combined)}")

    # 4. Αφαίρεση Διπλοτύπων Κειμένων
    initial_count = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=[TEXT_COL], keep='first')
    removed_duplicates = initial_count - len(df_combined)
    if removed_duplicates > 0:
        print(f"Αφαιρέθηκαν {removed_duplicates} διπλότυπα κείμενα.")
    print(f"Μέγεθος μετά την αφαίρεση διπλοτύπων: {len(df_combined)}")

    # 5. Ανακάτεμα (Shuffle)
    print("Ανακάτεμα του τελικού dataset...")
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # 6. Αποθήκευση Τελικού Dataset
    try:
        # Διασφάλιση ότι ο φάκελος εξόδου υπάρχει
        output_dir = os.path.dirname(OUTPUT_DATASET_PATH)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Αποθήκευση τελικού dataset 3 κλάσεων στο: {OUTPUT_DATASET_PATH}")
        df_combined.to_csv(OUTPUT_DATASET_PATH, index=False, encoding='utf-8-sig') # utf-8-sig για καλύτερη συμβατότητα με Excel
        print("Η αποθήκευση ολοκληρώθηκε.")

        # Εκτύπωση τελικής κατανομής (Διόρθωση print statement)
        print("\n--- Τελική Κατανομή Ετικετών ---") # Αφαίρεση λανθασμένου \r
        print(df_combined[LABEL_COL].value_counts())
        print(f"({NEG_LABEL}=Αρνητικό, {POS_LABEL}=Θετικό, {NEU_LABEL}=Ουδέτερο)")
        print("---------------------------------")

    except Exception as e:
        print(f"Σφάλμα κατά την αποθήκευση του τελικού dataset: {e}")

if __name__ == "__main__":
    # Το script πρέπει να τρέξει από τον κύριο φάκελο thesis/ πλέον
    combine_datasets() 