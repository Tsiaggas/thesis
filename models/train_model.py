#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script για την εκπαίδευση του μοντέλου ανάλυσης συναισθημάτων.
Χρησιμοποιεί το dataset που βρίσκεται στον φάκελο data και εκπαιδεύει
ένα μοντέλο που αποθηκεύεται στον φάκελο models.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Προσθήκη του parent directory στο path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.text_preprocessing import preprocess_text

def create_sample_data():
    """Δημιουργεί ένα δείγμα δεδομένων για εκπαίδευση όταν δεν υπάρχει το dataset."""
    print("Δημιουργία δείγματος δεδομένων για εκπαίδευση...")
    
    data = {
        'text': [
            'Το προϊόν είναι εξαιρετικό, είμαι πολύ ευχαριστημένος!',
            'Μου άρεσε πολύ η εξυπηρέτηση, θα ξαναγοράσω σίγουρα.',
            'Καλή ποιότητα προϊόντος, άμεση εξυπηρέτηση.',
            'Η παράδοση ήταν γρήγορη και το προϊόν σε άριστη κατάσταση.',
            'Εξαιρετική αγορά, προτείνω ανεπιφύλακτα!',
            'Άψογη εμπειρία αγοράς, θα ξαναγοράσω σίγουρα.',
            'Είμαι πολύ ικανοποιημένος από την αγορά μου.',
            'Καλή σχέση ποιότητας-τιμής, αξίζει τα χρήματά του.',
            'Άριστη εξυπηρέτηση και ποιοτικό προϊόν.',
            'Ικανοποιημένος από την αγορά, καλή επιλογή.',
            'Το προϊόν ήταν κατώτερο των προσδοκιών μου.',
            'Δεν είμαι καθόλου ευχαριστημένος, θα ζητήσω επιστροφή χρημάτων.',
            'Κακή ποιότητα, το προϊόν χάλασε μετά από λίγες μέρες.',
            'Η εξυπηρέτηση ήταν απαράδεκτη, περίμενα ώρες στο τηλέφωνο.',
            'Μεγάλη καθυστέρηση στην παράδοση, δεν θα ξαναπαραγγείλω.',
            'Προβληματικό προϊόν, δεν λειτουργεί σωστά.',
            'Απογοητευτική εμπειρία, δεν το συνιστώ.',
            'Κακή συσκευασία, το προϊόν έφτασε κατεστραμμένο.',
            'Υπερτιμημένο προϊόν, δεν αξίζει τα χρήματά του.',
            'Η ποιότητα είναι πολύ χαμηλή, μετανιώνω για την αγορά.',
        ],
        'label': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'emotions': [
            {'χαρά': 1, 'λύπη': 0, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 1, 'λύπη': 0, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 1, 'λύπη': 0, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 1, 'λύπη': 0, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 1, 'λύπη': 0, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 1, 'λύπη': 0, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 1, 'λύπη': 0, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 1, 'λύπη': 0, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 1, 'λύπη': 0, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 1, 'λύπη': 0, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 0, 'λύπη': 1, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 0, 'λύπη': 0, 'θυμός': 1, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 0, 'λύπη': 1, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 0, 'λύπη': 0, 'θυμός': 1, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 0, 'λύπη': 0, 'θυμός': 1, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 0, 'λύπη': 1, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 0, 'λύπη': 1, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 0, 'λύπη': 0, 'θυμός': 1, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 0, 'λύπη': 0, 'θυμός': 1, 'έκπληξη': 0, 'φόβος': 0},
            {'χαρά': 0, 'λύπη': 1, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0},
        ]
    }
    
    return pd.DataFrame(data)

def train_model(df):
    """Εκπαιδεύει το μοντέλο ανάλυσης συναισθημάτων."""
    # Προεπεξεργασία κειμένου
    print("Προεπεξεργασία κειμένου...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Χωρισμός σε σύνολα εκπαίδευσης και ελέγχου
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Δημιουργία pipeline
    print("Εκπαίδευση μοντέλου...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    # Εκπαίδευση
    pipeline.fit(X_train, y_train)
    
    # Αξιολόγηση
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ακρίβεια: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Αρνητικό', 'Θετικό']))
    
    # Δημιουργία και αποθήκευση λεξικού συναισθημάτων
    print("Δημιουργία μοντέλου συναισθημάτων...")
    emotion_model = {}
    for i, row in df.iterrows():
        terms = row['processed_text'].split()
        emotions = row['emotions']
        for term in terms:
            if term not in emotion_model:
                emotion_model[term] = {
                    'χαρά': 0, 'λύπη': 0, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0
                }
            for emotion, value in emotions.items():
                emotion_model[term][emotion] += value
    
    # Κανονικοποίηση τιμών
    for term in emotion_model:
        total = sum(emotion_model[term].values())
        if total > 0:
            for emotion in emotion_model[term]:
                emotion_model[term][emotion] /= total
    
    # Συνδυασμός των δύο μοντέλων
    model = {
        'sentiment_model': pipeline,
        'emotion_model': emotion_model
    }
    
    return model

def save_model(model, model_path):
    """Αποθηκεύει το μοντέλο στον δίσκο."""
    print(f"Αποθήκευση μοντέλου στο {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Το μοντέλο αποθηκεύτηκε επιτυχώς!")

def main():
    """Κύρια συνάρτηση."""
    # Φάκελος δεδομένων
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    models_dir = os.path.dirname(__file__)
    
    # Έλεγχος για ύπαρξη αρχείου δεδομένων
    try:
        data_file = os.path.join(data_dir, 'processed', 'cleaned_data.csv')
        if os.path.exists(data_file):
            print(f"Φόρτωση δεδομένων από {data_file}...")
            df = pd.read_csv(data_file)
            # Έλεγχος για ύπαρξη απαραίτητων στηλών
            if 'text' not in df.columns or 'label' not in df.columns:
                print("Το αρχείο δεδομένων δεν περιέχει τις απαραίτητες στήλες. Χρήση δεδομένων δείγματος.")
                df = create_sample_data()
        else:
            print("Δεν βρέθηκε αρχείο δεδομένων. Χρήση δεδομένων δείγματος.")
            df = create_sample_data()
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση δεδομένων: {e}")
        print("Χρήση δεδομένων δείγματος.")
        df = create_sample_data()
    
    # Δημιουργία στήλης emotions αν δεν υπάρχει
    if 'emotions' not in df.columns:
        print("Προσθήκη στήλης emotions...")
        # Απλοποιημένο: Θετικό = χαρά, Αρνητικό = λύπη/θυμός
        df['emotions'] = df['label'].apply(lambda x: 
            {'χαρά': 1, 'λύπη': 0, 'θυμός': 0, 'έκπληξη': 0, 'φόβος': 0} if x == 1 
            else {'χαρά': 0, 'λύπη': 0.5, 'θυμός': 0.5, 'έκπληξη': 0, 'φόβος': 0}
        )
    
    # Εκπαίδευση μοντέλου
    model = train_model(df)
    
    # Αποθήκευση μοντέλου
    model_path = os.path.join(models_dir, 'best_model_with_emotions.pkl')
    save_model(model, model_path)

if __name__ == "__main__":
    main() 