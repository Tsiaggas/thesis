#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script για Data Augmentation στο dataset κριτικών Skroutz.
Στόχος: Βελτίωση της απόδοσης του μοντέλου σε σύντομες αρνητικές κριτικές.
Τεχνικές: Αντικατάσταση Συνωνύμων, Περικοπή, Χειροκίνητη Προσθήκη.
"""

import pandas as pd
import numpy as np
import random
import os
import re

# --- Παράμετροι ---
# Αφαίρεση του '../' από τη διαδρομή
INPUT_DATASET = 'data/Skroutz_dataset.xlsx'
# Αφαίρεση του '../' από τη διαδρομή
OUTPUT_DATASET = 'data/processed/augmented_skroutz_dataset.csv'
TEXT_COLUMN = 'review_text'  # Ενημέρωσε με το σωστό όνομα στήλης αν διαφέρει
SENTIMENT_COLUMN = 'sentiment_label'  # Ενημέρωσε με το σωστό όνομα στήλης αν διαφέρει
# Αλλαγή σε 0 για να ταιριάζει με την αριθμητική αναπαράσταση μετά τη μετατροπή
NEGATIVE_LABEL = 0  # Η *αριθμητική* τιμή για αρνητικό συναίσθημα μετά τη μετατροπή
POSITIVE_LABEL = 1 # Η *αριθμητική* τιμή για θετικό συναίσθημα μετά τη μετατροπή
MAX_WORDS_SHORT = 10      # Μέγιστος αριθμός λέξεων για "σύντομη" κριτική
# Αύξηση σε 3 για περισσότερες προσπάθειες αντικατάστασης
NUM_SYNONYMS = 3           # Πόσες *προσπάθειες* αντικατάστασης συνωνύμων να γίνουν ανά κριτική
# Μετονομασία και αύξηση του στόχου περικοπής
TARGET_NUM_TRUNCATED = 500 # *Στόχος* για τον αριθμό των περικομμένων κριτικών που θα δημιουργηθούν
MIN_WORDS_LONG_FOR_TRUNCATION = 20 # Ελάχιστες λέξεις για να θεωρηθεί "μακροσκελής" και να περικοπεί

# --- Βοηθητικές Συναρτήσεις ---

def count_words(text):
    """Μετράει τις λέξεις σε ένα κείμενο."""
    if pd.isna(text):
        return 0
    # Απλή μέτρηση με βάση τα κενά, αφαιρώντας σημεία στίξης για καλύτερη προσέγγιση
    text_cleaned = re.sub(r'[^\w\s]', '', str(text))
    return len(text_cleaned.split())

# --- Λεξικό Συνωνύμων (Απλό Παράδειγμα) ---
# Αυτό θα μπορούσε να επεκταθεί σημαντικά ή να χρησιμοποιηθεί εξωτερική πηγή
SYNONYM_DICT = {
    "κακό": ["άθλιο", "απαράδεκτο", "χάλια", "μάπα"],
    "απογοήτευση": ["δυσαρέσκεια", "πίκρα"],
    "δεν δουλεύει": ["χαλασμένο", "εκτός λειτουργίας", "προβληματικό"],
    "ακριβό": ["πανάκριβο", "τσουχτερό"],
    "αργό": ["αργοκίνητο", "κολλάει", "βαρύ"],
    "σπάει": ["έσπασε", "διαλύθηκε", "χάλασε"],
    "μικρό": ["πολύ μικρό", "λιλιπούτειο"],
    "λάθος": ["εσφαλμένο", "λανθασμένο"],
    # Προσθέστε περισσότερα ζεύγη λέξεων/φράσεων και συνωνύμων
}

def synonym_replacement(text, num_replacements=1):
    """Αντικαθιστά τυχαίες λέξεις με συνώνυμα από το SYNONYM_DICT."""
    words = text.split()
    augmented_text = words[:]
    words_replaced = 0

    available_synonyms = {k: v for k, v in SYNONYM_DICT.items() if k in text}
    if not available_synonyms:
        return text # Δεν βρέθηκαν λέξεις για αντικατάσταση

    keys_to_replace = list(available_synonyms.keys())
    random.shuffle(keys_to_replace)

    for key in keys_to_replace:
        if words_replaced >= num_replacements:
            break
        try:
            synonyms = available_synonyms[key]
            if synonyms:
                synonym = random.choice(synonyms)
                # Απλή αντικατάσταση (μπορεί να μην είναι τέλεια γραμματικά)
                augmented_text_str = " ".join(augmented_text)
                # Χρήση regex για αντικατάσταση ολόκληρης της λέξης/φράσης
                new_text_str = re.sub(r'\b' + re.escape(key) + r'\b', synonym, augmented_text_str, count=1)
                if new_text_str != augmented_text_str: # Αν έγινε αντικατάσταση
                     augmented_text = new_text_str.split()
                     words_replaced += 1
        except Exception as e:
            print(f"Σφάλμα στην αντικατάσταση συνωνύμου για '{key}': {e}")
            continue # Προχωράμε στην επόμενη λέξη

    return " ".join(augmented_text)

def truncate_review(text, max_words=MAX_WORDS_SHORT):
    """Περικόπτει μια κριτική στις πρώτες max_words λέξεις."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..." # Προσθέτουμε ... για ένδειξη περικοπής
    return text

# --- Κύρια Λογική ---

print(f"Φόρτωση dataset: {INPUT_DATASET}")
try:
    df = pd.read_excel(INPUT_DATASET)
    print(f"Το dataset φορτώθηκε. Σχήμα: {df.shape}")
    print(f"Στήλες: {df.columns.tolist()}")

    # --- Έλεγχος και προσαρμογή ονομάτων στηλών ---
    if TEXT_COLUMN not in df.columns:
        # Προσπάθεια εύρεσης εναλλακτικών ονομάτων
        alt_text_cols = ['review', 'comment', 'κείμενο', 'Κείμενο']
        found = False
        for col in alt_text_cols:
            if col in df.columns:
                TEXT_COLUMN = col
                print(f"Βρέθηκε η στήλη κειμένου: '{TEXT_COLUMN}'")
                found = True
                break
        if not found:
            raise ValueError(f"Δεν βρέθηκε η στήλη κειμένου (αναμενόταν '{TEXT_COLUMN}' ή εναλλακτικές).")

    if SENTIMENT_COLUMN not in df.columns:
         # Προσπάθεια εύρεσης εναλλακτικών ονομάτων
        alt_sentiment_cols = ['rating', 'score', 'label', 'συναίσθημα']
        found = False
        for col in alt_sentiment_cols:
             if col in df.columns:
                 SENTIMENT_COLUMN = col
                 print(f"Βρέθηκε η στήλη συναισθήματος: '{SENTIMENT_COLUMN}'")
                 found = True
                 break
        if not found:
             raise ValueError(f"Δεν βρέθηκε η στήλη συναισθήματος (αναμενόταν '{SENTIMENT_COLUMN}' ή εναλλακτικές).")

    # Μετονομασία για συνέπεια
    df.rename(columns={TEXT_COLUMN: 'text', SENTIMENT_COLUMN: 'sentiment'}, inplace=True)
    TEXT_COLUMN = 'text'
    SENTIMENT_COLUMN = 'sentiment'

    # Μετατροπή συναισθήματος σε αριθμητικό αν χρειάζεται (π.χ. από Positive/Negative)
    if df[SENTIMENT_COLUMN].dtype == 'object':
        print("Μετατροπή κειμενικών ετικετών συναισθήματος σε αριθμητικές (0=Αρνητικό, 1=Θετικό)")
        # Απλή υπόθεση, μπορεί να χρειαστεί προσαρμογή
        df[SENTIMENT_COLUMN] = df[SENTIMENT_COLUMN].apply(lambda x: 0 if isinstance(x, str) and x.lower() in ['negative', 'αρνητικό', 'bad'] else 1)
    # Εξασφάλιση ότι η στήλη sentiment είναι αριθμητική
    df[SENTIMENT_COLUMN] = pd.to_numeric(df[SENTIMENT_COLUMN], errors='coerce')
    df.dropna(subset=[SENTIMENT_COLUMN], inplace=True) # Αφαίρεση γραμμών όπου η μετατροπή απέτυχε
    df[SENTIMENT_COLUMN] = df[SENTIMENT_COLUMN].astype(int)

    # --- Εντοπισμός Κριτικών ---
    df['word_count'] = df[TEXT_COLUMN].apply(count_words)
    short_negative_reviews = df[
        (df[SENTIMENT_COLUMN] == NEGATIVE_LABEL) &
        (df['word_count'] <= MAX_WORDS_SHORT) &
        (df['word_count'] > 0) # Αποκλεισμός εντελώς κενών
    ]
    long_negative_reviews = df[
        (df[SENTIMENT_COLUMN] == NEGATIVE_LABEL) &
        (df['word_count'] > MIN_WORDS_LONG_FOR_TRUNCATION)
    ]

    print(f"Βρέθηκαν {len(short_negative_reviews)} σύντομες (<={MAX_WORDS_SHORT} λέξεις) αρνητικές κριτικές.")
    print(f"Βρέθηκαν {len(long_negative_reviews)} μεγάλες αρνητικές κριτικές για περικοπή.")

    augmented_data = []
    num_added_synonym = 0
    num_truncated_added = 0
    num_added_manual = 0

    # --- 1. Augmentation με Συνώνυμα ---
    print("Δημιουργία παραλλαγών με αντικατάσταση συνωνύμων...")
    for index, row in short_negative_reviews.iterrows():
        original_text = str(row[TEXT_COLUMN]) # Εξασφάλιση ότι είναι string
        added_for_this_review = 0
        for _ in range(NUM_SYNONYMS): # Προσπάθησε NUM_SYNONYMS φορές
            augmented_text = synonym_replacement(original_text)
            if augmented_text != original_text and augmented_text not in [d[TEXT_COLUMN] for d in augmented_data if d['source'] == 'synonym']: # Προσθήκη μόνο αν άλλαξε ΚΑΙ δεν υπάρχει ήδη
                augmented_data.append({TEXT_COLUMN: augmented_text, SENTIMENT_COLUMN: NEGATIVE_LABEL, 'source': 'synonym'})
                num_added_synonym += 1
                added_for_this_review += 1
        # if added_for_this_review > 0:
        #     print(f"  -> Προστέθηκαν {added_for_this_review} παραλλαγές για κριτική #{index}")
    print(f"-> Ολοκληρώθηκε η αντικατάσταση συνωνύμων. Προστέθηκαν {num_added_synonym} νέες κριτικές.")

    # --- 2. Augmentation με Περικοπή ---
    print("Δημιουργία παραλλαγών με περικοπή μεγάλων κριτικών...")
    if not long_negative_reviews.empty and TARGET_NUM_TRUNCATED > 0:
        # Προσπαθούμε να φτάσουμε τον στόχο, επιλέγοντας τυχαία
        # Διόρθωση: Χρήση TARGET_NUM_TRUNCATED
        num_to_select = min(TARGET_NUM_TRUNCATED, len(long_negative_reviews))
        print(f"  -> Επιλογή {num_to_select} από τις {len(long_negative_reviews)} διαθέσιμες μεγάλες αρνητικές κριτικές για περικοπή.")
        # Χρήση replace=False για να μην επιλεγεί η ίδια κριτική πολλές φορές
        reviews_to_truncate = long_negative_reviews.sample(n=num_to_select, replace=False)

        for index, row in reviews_to_truncate.iterrows():
            original_text = str(row[TEXT_COLUMN])
            # Χρήση MAX_WORDS_SHORT για τον στόχο περικοπής
            truncated_text = truncate_review(original_text, max_words=MAX_WORDS_SHORT)
            if truncated_text != original_text and len(truncated_text.split()) > 0: # Έλεγχος ότι δεν είναι κενή μετά την περικοπή
                augmented_data.append({TEXT_COLUMN: truncated_text, SENTIMENT_COLUMN: NEGATIVE_LABEL, 'source': 'truncated'})
                num_truncated_added += 1
    print(f"-> Ολοκληρώθηκε η περικοπή. Προστέθηκαν {num_truncated_added} νέες κριτικές.")

    # --- 3. Χειροκίνητη Προσθήκη ---
    print("Προσθήκη χειροκίνητων σύντομων αρνητικών κριτικών...")
    manual_reviews = [
        "Πολύ κακό.", "Απογοήτευση.", "Μην το αγοράσετε.", "Δεν αξίζει.",
        "Χάσιμο χρημάτων.", "Κακή ποιότητα.", "Δεν δουλεύει σωστά.", "Πολύ αργό.",
        "Ελαττωματικό προϊόν.", "Άθλιο.", "Μακριά.", "Απαράδεκτο.", "Χάλια.",
        "Πεταμένα λεφτά.", "Μη λειτουργικό.", "Πολύ μικρό.", "Καθόλου καλό.",
        "Τραγικό.", "Απογοητεύτηκα πλήρως.", "Πρόβλημα από την αρχή."
    ]
    for review in manual_reviews:
        augmented_data.append({TEXT_COLUMN: review, SENTIMENT_COLUMN: NEGATIVE_LABEL})

    print(f"Προστέθηκαν {len(manual_reviews)} χειροκίνητες κριτικές.")
    num_added_manual = len(manual_reviews)

    # --- Συνδυασμός και Αποθήκευση ---
    augmented_df = pd.DataFrame(augmented_data)

    # Κράτημα μόνο των απαραίτητων στηλών από το αρχικό df
    original_df_subset = df[[TEXT_COLUMN, SENTIMENT_COLUMN]].copy()

    # Συνένωση του αρχικού (επιλεγμένες στήλες) με τα αυξημένα δεδομένα
    final_df = pd.concat([original_df_subset, augmented_df], ignore_index=True)

    # Αφαίρεση διπλοτύπων που μπορεί να προέκυψαν
    final_df.drop_duplicates(subset=[TEXT_COLUMN], inplace=True)

    # Ανακάτεμα (shuffle) του τελικού dataset
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    print(f"Το αρχικό dataset είχε {len(df)} γραμμές.")
    print(f"Προστέθηκαν συνολικά {num_added_synonym + num_truncated_added + num_added_manual} νέες κριτικές (μετά την αφαίρεση διπλοτύπων μπορεί να είναι λιγότερες).")
    print(f"Το τελικό αυξημένο dataset έχει {len(final_df)} γραμμές.")

    # Δημιουργία του φακέλου processed αν δεν υπάρχει
    output_dir = os.path.dirname(OUTPUT_DATASET)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Δημιουργήθηκε ο φάκελος: {output_dir}")

    # Αποθήκευση του νέου dataset
    final_df.to_csv(OUTPUT_DATASET, index=False, encoding='utf-8-sig')
    print(f"Το αυξημένο dataset αποθηκεύτηκε στο: {OUTPUT_DATASET}")

except FileNotFoundError:
    print(f"Σφάλμα: Το αρχείο {INPUT_DATASET} δεν βρέθηκε.")
except ValueError as ve:
    print(f"Σφάλμα τιμής/στήλης: {ve}")
except Exception as e:
    print(f"Προέκυψε ένα απρόσμενο σφάλμα: {e}") 