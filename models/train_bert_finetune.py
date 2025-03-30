#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning ενός προ-εκπαιδευμένου μοντέλου BERT 
(nlpaueb/bert-base-greek-uncased-v1) για ανάλυση συναισθημάτων 
στο dataset Skroutz **3 κλάσεων (Αρνητικό/Θετικό/Ουδέτερο)**.
"""

import os
import sys
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

# --- Ρυθμίσεις --- 
MODEL_NAME = "nlpaueb/bert-base-greek-uncased-v1"
# Αλλαγή στο dataset 3 κλάσεων
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed', 'skroutz_3class_dataset.csv') 
# Αλλαγή στον φάκελο εξόδου για το μοντέλο 3 κλάσεων
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fine_tuned_bert_3class') 
LOGGING_DIR = os.path.join(OUTPUT_DIR, 'logs')
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_EPOCHS = 3  
BATCH_SIZE = 16 
LEARNING_RATE = 2e-5

# --- Βοηθητικές Συναρτήσεις --- 

def load_and_prepare_data(file_path):
    """
    Φορτώνει και προετοιμάζει το dataset 3 κλάσεων από CSV.
    """
    print(f"Φόρτωση δεδομένων από: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Φορτώθηκαν {len(df)} εγγραφές.")
        print("Στήλες που διαβάστηκαν:", df.columns.tolist())
        
        expected_text_col = 'text'
        expected_label_col = 'label' # Το script συνδυασμού έσωσε τη στήλη ως 'label'
        if expected_text_col not in df.columns or expected_label_col not in df.columns:
            raise ValueError(f"Το dataset πρέπει να περιέχει τις στήλες '{expected_text_col}' και '{expected_label_col}'. Βρέθηκαν: {df.columns.tolist()}")

        # Δεν χρειάζεται μετονομασία, το όνομα 'label' είναι ήδη σωστό
        df = df[[expected_text_col, expected_label_col]].rename(columns={expected_label_col: 'labels'})

        df = df.dropna(subset=['text', 'labels'])
        df['text'] = df['text'].astype(str)
        df['labels'] = df['labels'].astype(int)
        
        print(f"Δεδομένα μετά τον καθαρισμό: {len(df)}")
        print(f"Κατανομή ετικετών (στήλη 'labels'): {df['labels'].value_counts().sort_index().to_dict()}")
        
        # Έλεγχος ότι υπάρχουν μόνο 0, 1, 2
        expected_labels = {0, 1, 2}
        actual_labels = set(df['labels'].unique())
        if not actual_labels.issubset(expected_labels):
             print(f"Προειδοποίηση: Βρέθηκαν μη αναμενόμενες ετικέτες: {actual_labels}. Αναμένονται μόνο {expected_labels}.")
             df = df[df['labels'].isin(expected_labels)]
             print(f"Δεδομένα μετά το φιλτράρισμα μη αναμενόμενων ετικετών: {len(df)}")
             
        return df
        
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση/προετοιμασία δεδομένων: {e}")
        sys.exit(1)

def tokenize_data(batch, tokenizer):
    """
    Κάνει tokenize ένα batch δεδομένων.
    """
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=512)

# Προσαρμογή για multi-class
def compute_metrics(pred):
    """
    Υπολογίζει τις μετρικές αξιολόγησης για multi-class.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Χρήση 'weighted' average για να ληφθεί υπόψη η ανισορροπία κλάσεων
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted') 
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_weighted': f1, # Μετονομασία για σαφήνεια
        'precision_weighted': precision,
        'recall_weighted': recall
    }

# --- Κύρια Λογική --- 

def main():
    """
    Εκτελεί τη διαδικασία fine-tuning για 3 κλάσεις.
    """
    # 1. Φόρτωση και προετοιμασία δεδομένων (χρησιμοποιεί την τροποποιημένη συνάρτηση)
    df = load_and_prepare_data(DATA_PATH)
    
    # 2. Διαχωρισμός σε train/test sets (με stratify για τις 3 κλάσεις)
    train_df, test_df = train_test_split(
        df, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=df['labels'] # Διατηρεί την αναλογία των 3 κλάσεων
    )
    print(f"Μέγεθος συνόλου εκπαίδευσης: {len(train_df)}")
    print(f"Μέγεθος συνόλου δοκιμής: {len(test_df)}")
    print(f"Κατανομή στο Train set: {train_df['labels'].value_counts().sort_index().to_dict()}")
    print(f"Κατανομή στο Test set: {test_df['labels'].value_counts().sort_index().to_dict()}")
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # 3. Φόρτωση Tokenizer και Tokenization
    print(f"Φόρτωση tokenizer για το μοντέλο: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(lambda batch: tokenize_data(batch, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda batch: tokenize_data(batch, tokenizer), batched=True)
    
    # Αφαίρεση μη απαραίτητων στηλών και ρύθμιση format
    cols_to_remove = ['text']
    if '__index_level_0__' in train_dataset.column_names:
        cols_to_remove.append('__index_level_0__')
    train_dataset = train_dataset.remove_columns(cols_to_remove)
    test_dataset = test_dataset.remove_columns(cols_to_remove)
        
    train_dataset.set_format('torch')
    test_dataset.set_format('torch')
    
    # 4. Φόρτωση προ-εκπαιδευμένου μοντέλου με num_labels=3
    print(f"Φόρτωση προ-εκπαιδευμένου μοντέλου: {MODEL_NAME} (για 3 κλάσεις)")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Χρήση συσκευής: {device}")

    # 5. Ορισμός Παραμέτρων Εκπαίδευσης
    print("Ορισμός παραμέτρων εκπαίδευσης...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=LOGGING_DIR,
        logging_steps=50, 
        load_best_model_at_end=True,
        # Χρήση f1_weighted ως κύρια μετρική
        metric_for_best_model="f1_weighted", 
        greater_is_better=True,
        report_to="none"
    )

    # 6. Δημιουργία του προεπιλεγμένου Trainer (όχι CustomTrainer)
    print("Δημιουργία προεπιλεγμένου Trainer...")
    trainer = Trainer( # Χρήση του standard Trainer
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
        # Αφαίρεση παραμέτρων focal loss
    )

    # 7. Εκτέλεση Fine-tuning
    print("Έναρξη fine-tuning για 3 κλάσεις...")
    train_result = trainer.train()
    print("Το fine-tuning ολοκληρώθηκε.")

    # 8. Αποθήκευση του τελικού μοντέλου και tokenizer
    print(f"Αποθήκευση του fine-tuned μοντέλου (3 κλάσεων) στο: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR) 
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # 9. Αξιολόγηση στο test set
    print("\nΑξιολόγηση στο test set...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    print(f"\nΑποτελέσματα Αξιολόγησης στο Test Set (3 κλάσεις):")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")
        
    print(f"\nΤο Fine-tuned μοντέλο (3 κλάσεων) αποθηκεύτηκε επιτυχώς στο φάκελο: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 