"""
Προετοιμασία δεδομένων για εκπαίδευση και αξιολόγηση μοντέλων
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

from src.utils.config import (
    RAW_DATA_FILE, PROCESSED_DATA_DIR, RANDOM_SEED, 
    TEST_SIZE, VALIDATION_SIZE
)
from src.preprocessing.text_preprocessing import (
    load_and_preprocess_data, count_emotion_words,
    download_nltk_resources
)

def prepare_data():
    """
    Προετοιμασία των δεδομένων για εκπαίδευση και αξιολόγηση
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Κατέβασμα των απαραίτητων resources από το NLTK
    download_nltk_resources()
    
    # Φόρτωση και προεπεξεργασία των δεδομένων
    print("Φόρτωση και προεπεξεργασία των δεδομένων...")
    data = load_and_preprocess_data(RAW_DATA_FILE)
    
    # Προσθήκη στήλης με τα συναισθήματα
    print("Ανάλυση συναισθημάτων στις κριτικές...")
    data['emotions'] = data['review'].apply(count_emotion_words)
    
    # Εξαγωγή χαρακτηριστικών συναισθημάτων σε ξεχωριστές στήλες
    for emotion in data['emotions'].iloc[0].keys():
        data[f'emotion_{emotion}'] = data['emotions'].apply(lambda x: x[emotion])
    
    # Μετατροπή των ετικετών σε αριθμητικές τιμές
    data['label_numeric'] = data['label'].apply(lambda x: 1 if x == 'Positive' else 0)
    
    # Αποθήκευση των προεπεξεργασμένων δεδομένων
    processed_data_file = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv')
    data.to_csv(processed_data_file, index=False)
    
    # Χωρισμός σε features και target
    X = data[['processed_review'] + [f'emotion_{emotion}' for emotion in data['emotions'].iloc[0].keys()]]
    y = data['label_numeric']
    
    # Χωρισμός σε training, validation και test set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VALIDATION_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    # Υπολογισμός του validation_size ως ποσοστό του υπόλοιπου συνόλου
    relative_val_size = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE/(TEST_SIZE + VALIDATION_SIZE), 
        random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"Μέγεθος training set: {X_train.shape[0]} δείγματα")
    print(f"Μέγεθος validation set: {X_val.shape[0]} δείγματα")
    print(f"Μέγεθος test set: {X_test.shape[0]} δείγματα")
    
    # Αποθήκευση των συνόλων δεδομένων
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_data.csv'), index=False)
    val_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'val_data.csv'), index=False)
    test_data.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_data.csv'), index=False)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_or_prepare_data():
    """
    Φόρτωση προεπεξεργασμένων δεδομένων αν υπάρχουν, αλλιώς προετοιμασία νέων
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    train_file = os.path.join(PROCESSED_DATA_DIR, 'train_data.csv')
    val_file = os.path.join(PROCESSED_DATA_DIR, 'val_data.csv')
    test_file = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')
    
    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
        print("Φόρτωση προεπεξεργασμένων δεδομένων...")
        train_data = pd.read_csv(train_file)
        val_data = pd.read_csv(val_file)
        test_data = pd.read_csv(test_file)
        
        # Διαχωρισμός σε features και target
        emotion_cols = [col for col in train_data.columns if col.startswith('emotion_')]
        X_train = train_data[['processed_review'] + emotion_cols]
        y_train = train_data['label_numeric']
        
        X_val = val_data[['processed_review'] + emotion_cols]
        y_val = val_data['label_numeric']
        
        X_test = test_data[['processed_review'] + emotion_cols]
        y_test = test_data['label_numeric']
    else:
        print("Προεπεξεργασμένα δεδομένα δεν βρέθηκαν. Προετοιμασία νέων δεδομένων...")
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Εκτέλεση της προετοιμασίας δεδομένων
    X_train, X_val, X_test, y_train, y_val, y_test = load_or_prepare_data()
    
    # Εμφάνιση πληροφοριών για τα δεδομένα
    print("\nΚατανομή ετικετών στο training set:")
    print(y_train.value_counts())
    
    print("\nΚατανομή ετικετών στο validation set:")
    print(y_val.value_counts())
    
    print("\nΚατανομή ετικετών στο test set:")
    print(y_test.value_counts()) 