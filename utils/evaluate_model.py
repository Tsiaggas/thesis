#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Αξιολόγηση του fine-tuned μοντέλου BERT για ανάλυση συναισθημάτων 
χρησιμοποιώντας το dataset Skroutz.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Προσθήκη του φακέλου του έργου στο path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Βοηθητικές Συναρτήσεις --- 

def load_dataset(file_path, sample_size=None, test_ratio=0.2):
    """
    Φορτώνει το dataset και το προετοιμάζει για αξιολόγηση.
    Επιστρέφει μόνο το test set.
    """
    print(f"Φόρτωση του dataset από το αρχείο: {file_path}")
    
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Μη υποστηριζόμενος τύπος αρχείου: {file_path}")
    except Exception as e:
        print(f"Σφάλμα κατά την ανάγνωση του αρχείου: {e}")
        return None, None
    
    print(f"Φορτώθηκαν {len(df)} εγγραφές.")
    
    # Έλεγχος και προετοιμασία των στηλών (παρόμοια με το train script)
    if 'review' not in df.columns or 'label' not in df.columns:
        print("Το dataset πρέπει να περιέχει τις στήλες 'review' και 'label'.")
        return None, None
        
    df = df.rename(columns={'review': 'text', 'label': 'labels'})
    df = df.dropna(subset=['text', 'labels'])
    df['text'] = df['text'].astype(str)
    
    label_map = {
        'positive': 1, 'Positive': 1, 'POSITIVE': 1,
        'negative': 0, 'Negative': 0, 'NEGATIVE': 0
    }
    df['labels'] = df['labels'].map(label_map)
    df = df.dropna(subset=['labels'])
    df['labels'] = df['labels'].astype(int)
    
    print(f"Κατανομή συναισθημάτων στο αρχικό dataset: {df['labels'].value_counts().to_dict()}")
    
    # Λήψη δείγματος αν έχει καθοριστεί
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Επιλέχθηκε τυχαίο δείγμα {sample_size} εγγραφών για αξιολόγηση.")
    else:
        print("Χρήση ολόκληρου του dataset για αξιολόγηση.")
        
    # Χρησιμοποιούμε τα δεδομένα ως test set
    X_test = df['text'].values
    y_test = df['labels'].values
    
    print(f"Μέγεθος συνόλου αξιολόγησης: {len(X_test)}")
    
    return X_test, y_test

def load_finetuned_model(model_path):
    """
    Φορτώνει το fine-tuned μοντέλο και tokenizer από έναν τοπικό φάκελο.
    """
    print(f"Φόρτωση fine-tuned μοντέλου από: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Έλεγχος για GPU
        device = 0 if torch.cuda.is_available() else -1 # device=0 για πρώτη GPU, -1 για CPU
        print(f"Χρήση συσκευής για pipeline: {'GPU' if device == 0 else 'CPU'}")
        
        # Δημιουργία pipeline για ευκολότερες προβλέψεις
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=model, 
            tokenizer=tokenizer, 
            device=device,
            return_all_scores=True # Επιστρέφει σκορ για όλες τις ετικέτες
        )
        print("Fine-tuned μοντέλο και pipeline φορτώθηκαν επιτυχώς.")
        return sentiment_pipeline
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση του fine-tuned μοντέλου: {e}")
        return None

def evaluate_model(sentiment_pipeline, X_test, y_test, model_name="Fine-Tuned BERT"):
    """
    Αξιολογεί το fine-tuned μοντέλο χρησιμοποιώντας το pipeline.
    
    Args:
        sentiment_pipeline: Το sentiment analysis pipeline των transformers
        X_test (array): Τα δεδομένα ελέγχου
        y_test (array): Οι αληθείς ετικέτες
        model_name (str): Το όνομα του μοντέλου για αναφορά
        
    Returns:
        dict: Αποτελέσματα αξιολόγησης
    """
    if sentiment_pipeline is None:
        print("Το pipeline δεν φορτώθηκε, αδυναμία αξιολόγησης.")
        return None
        
    print(f"\nΑξιολόγηση του μοντέλου: {model_name}")
    
    # Πρόβλεψη συναισθημάτων χρησιμοποιώντας το pipeline
    # Το pipeline χειρίζεται το batching εσωτερικά αν χρειαστεί
    print("Εκτέλεση προβλέψεων με το pipeline...")
    try:
        # Το pipeline επιστρέφει λίστα από λίστες λεξικών (ένα λεξικό για κάθε label)
        # π.χ. [[{'label': 'LABEL_0', 'score': 0.01}, {'label': 'LABEL_1', 'score': 0.99}], ...]
        # LABEL_0 συνήθως αντιστοιχεί στο αρνητικό (0), LABEL_1 στο θετικό (1)
        predictions = sentiment_pipeline(X_test.tolist(), batch_size=16) # Προσαρμόστε το batch_size αν χρειάζεται
        
        # Εξαγωγή της προβλεπόμενης ετικέτας (αυτή με το υψηλότερο σκορ)
        y_pred = []
        for result_list in predictions:
            # Βρίσκουμε το λεξικό με το υψηλότερο σκορ
            best_pred = max(result_list, key=lambda x: x['score'])
            # Μετατρέπουμε την ετικέτα (π.χ., 'LABEL_1') σε int (1)
            predicted_label = int(best_pred['label'].split('_')[1])
            y_pred.append(predicted_label)
            
    except Exception as e:
        print(f"Σφάλμα κατά την πρόβλεψη με το pipeline: {e}")
        return None

    print("Ολοκληρώθηκαν οι προβλέψεις.")
    
    # Υπολογισμός μετρικών
    accuracy = accuracy_score(y_test, y_pred)
    # Χρησιμοποιούμε average='binary' γιατί έχουμε 2 κλάσεις και το y_test/y_pred είναι 0 ή 1
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary') 
    cm = confusion_matrix(y_test, y_pred)
    
    # Εκτύπωση αποτελεσμάτων
    print(f"\nΑποτελέσματα για το μοντέλο {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Positive Class): {precision:.4f}")
    print(f"Recall (Positive Class): {recall:.4f}")
    print(f"F1 Score (Positive Class): {f1:.4f}")
    
    # Δημιουργία του confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Αρνητικό (0)', 'Θετικό (1)'],
                yticklabels=['Αρνητικό (0)', 'Θετικό (1)'])
    plt.xlabel('Προβλεπόμενη Ετικέτα')
    plt.ylabel('Αληθής Ετικέτα')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Αποθήκευση του γραφήματος
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png'))
    print(f"Το Confusion Matrix αποθηκεύτηκε στο: {output_dir}")
    plt.close()
    
    # Αποθήκευση αποτελεσμάτων σε αρχείο text
    results_text = (
        f"Αποτελέσματα Αξιολόγησης - {model_name}\n"
        f"----------------------------------------\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision (Positive): {precision:.4f}\n"
        f"Recall (Positive): {recall:.4f}\n"
        f"F1 Score (Positive): {f1:.4f}\n\n"
        f"Confusion Matrix:\n{cm}\n"
    )
    results_file_path = os.path.join(output_dir, f'evaluation_results_{model_name.replace(" ", "_")}.txt')
    try:
        with open(results_file_path, 'w', encoding='utf-8') as f:
            f.write(results_text)
        print(f"Τα αποτελέσματα κειμένου αποθηκεύτηκαν στο: {results_file_path}")
    except Exception as e:
        print(f"Σφάλμα κατά την αποθήκευση των αποτελεσμάτων κειμένου: {e}")
        
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
    }


def main():
    """
    Κύρια συνάρτηση του σεναρίου αξιολόγησης.
    """
    # Διαδρομή στο fine-tuned μοντέλο
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'models', 'fine_tuned_bert')
                             
    # Διαδρομή στο dataset
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'data', 'Skroutz_dataset.xlsx')
                             
    # Φόρτωση δεδομένων (χρήση ολόκληρου του dataset ως test set)
    X_test, y_test = load_dataset(data_path, sample_size=None) 
    
    if X_test is None or y_test is None:
        print("Αδυναμία φόρτωσης ή προετοιμασίας των δεδομένων. Τερματισμός.")
        return
        
    # Φόρτωση του fine-tuned μοντέλου ως pipeline
    sentiment_pipeline = load_finetuned_model(model_dir)
    
    # Αξιολόγηση του μοντέλου
    if sentiment_pipeline:
        evaluation_results = evaluate_model(sentiment_pipeline, X_test, y_test)
        if evaluation_results:
            print("\nΗ αξιολόγηση ολοκληρώθηκε επιτυχώς.")
        else:
            print("\nΗ αξιολόγηση απέτυχε.")
    else:
        print("Αδυναμία φόρτωσης του μοντέλου. Η αξιολόγηση δεν μπορεί να συνεχιστεί.")

if __name__ == "__main__":
    main() 