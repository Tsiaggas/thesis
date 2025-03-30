"""
Βοηθητικές συναρτήσεις για το project
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from src.utils.config import MODELS_DIR


def save_model(model, model_name, vectorizer=None):
    """
    Αποθήκευση μοντέλου και vectorizer
    
    Args:
        model: Το μοντέλο προς αποθήκευση
        model_name (str): Το όνομα του μοντέλου
        vectorizer: Ο vectorizer που χρησιμοποιήθηκε (προαιρετικό)
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    
    if vectorizer is not None:
        vectorizer_path = os.path.join(MODELS_DIR, f"{model_name}_vectorizer.pkl")
        joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Το μοντέλο αποθηκεύτηκε επιτυχώς στο: {model_path}")


def load_model(model_name, with_vectorizer=False):
    """
    Φόρτωση αποθηκευμένου μοντέλου και vectorizer
    
    Args:
        model_name (str): Το όνομα του μοντέλου
        with_vectorizer (bool): Αν θα φορτωθεί και ο vectorizer
    
    Returns:
        model: Το φορτωμένο μοντέλο
        vectorizer (optional): Ο φορτωμένος vectorizer, αν with_vectorizer=True
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    model = joblib.load(model_path)
    
    if with_vectorizer:
        vectorizer_path = os.path.join(MODELS_DIR, f"{model_name}_vectorizer.pkl")
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    
    return model


def evaluate_model(model, X_test, y_test, vectorizer=None, model_name=None):
    """
    Αξιολόγηση της απόδοσης του μοντέλου
    
    Args:
        model: Το μοντέλο προς αξιολόγηση
        X_test: Το σύνολο δεδομένων δοκιμής
        y_test: Οι πραγματικές ετικέτες
        vectorizer: Ο vectorizer που χρησιμοποιήθηκε (προαιρετικό)
        model_name (str): Το όνομα του μοντέλου (προαιρετικό)
        
    Returns:
        dict: Λεξικό με τις μετρικές αξιολόγησης
    """
    # Πρόβλεψη
    if vectorizer is not None and 'processed_review' in X_test.columns:
        X_text = vectorizer.transform(X_test['processed_review'])
        other_features = X_test.drop('processed_review', axis=1).values
        
        if other_features.shape[1] > 0:
            X_transformed = np.hstack((X_text.toarray(), other_features))
        else:
            X_transformed = X_text
    else:
        X_transformed = X_test
    
    y_pred = model.predict(X_transformed)
    
    # Υπολογισμός μετρικών
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Εμφάνιση αποτελεσμάτων
    print(f"Αποτελέσματα αξιολόγησης" + (f" για το μοντέλο {model_name}" if model_name else ""))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nΛεπτομερής αναφορά:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Πίνακας σύγχυσης
    cm = confusion_matrix(y_test, y_pred)
    
    # Αποθήκευση αποτελεσμάτων
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Αν είναι δυνατόν, υπολογισμός και της ROC καμπύλης
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_transformed)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        results['roc_auc'] = roc_auc
        results['fpr'] = fpr
        results['tpr'] = tpr
        
        print(f"AUC: {roc_auc:.4f}")
    
    return results


def plot_confusion_matrix(confusion_matrix, title=None, save_path=None):
    """
    Σχεδιασμός πίνακα σύγχυσης
    
    Args:
        confusion_matrix: Ο πίνακας σύγχυσης προς απεικόνιση
        title (str): Τίτλος γραφήματος (προαιρετικό)
        save_path (str): Διαδρομή όπου θα αποθηκευτεί το γράφημα (προαιρετικό)
    """
    plt.figure(figsize=(8, 6))
    
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    
    classes = ['Αρνητικό', 'Θετικό']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Προσθήκη αριθμών στα κελιά
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Πραγματική Ετικέτα')
    plt.xlabel('Προβλεπόμενη Ετικέτα')
    
    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, title=None, save_path=None):
    """
    Σχεδιασμός ROC καμπύλης
    
    Args:
        fpr: False Positive Rate
        tpr: True Positive Rate
        roc_auc: Τιμή AUC
        title (str): Τίτλος γραφήματος (προαιρετικό)
        save_path (str): Διαδρομή όπου θα αποθηκευτεί το γράφημα (προαιρετικό)
    """
    plt.figure(figsize=(8, 6))
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC καμπύλη (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    if title:
        plt.title(title)
    else:
        plt.title('Receiver Operating Characteristic')
    
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def create_lexicon_from_data(data, label_column='label', text_column='review', min_freq=5):
    """
    Δημιουργία λεξικού συναισθημάτων από τα δεδομένα
    
    Args:
        data (pd.DataFrame): Τα δεδομένα
        label_column (str): Το όνομα της στήλης με τις ετικέτες
        text_column (str): Το όνομα της στήλης με το κείμενο
        min_freq (int): Ελάχιστη συχνότητα εμφάνισης λέξης
        
    Returns:
        dict: Λεξικό με λέξεις και βαθμολογίες συναισθήματος
    """
    # Διαχωρισμός των δεδομένων σε θετικά και αρνητικά
    positive_texts = data[data[label_column] == 'Positive'][text_column].tolist()
    negative_texts = data[data[label_column] == 'Negative'][text_column].tolist()
    
    # Tokenization
    positive_tokens = []
    for text in positive_texts:
        tokens = text.lower().split()
        positive_tokens.extend(tokens)
    
    negative_tokens = []
    for text in negative_texts:
        tokens = text.lower().split()
        negative_tokens.extend(tokens)
    
    # Υπολογισμός συχνοτήτων
    positive_freq = {}
    for token in positive_tokens:
        positive_freq[token] = positive_freq.get(token, 0) + 1
    
    negative_freq = {}
    for token in negative_tokens:
        negative_freq[token] = negative_freq.get(token, 0) + 1
    
    # Υπολογισμός βαθμολογιών συναισθήματος
    lexicon = {}
    all_tokens = set(list(positive_freq.keys()) + list(negative_freq.keys()))
    
    for token in all_tokens:
        pos_count = positive_freq.get(token, 0)
        neg_count = negative_freq.get(token, 0)
        
        # Έλεγχος συχνότητας
        if pos_count + neg_count >= min_freq:
            total = pos_count + neg_count
            lexicon[token] = (pos_count - neg_count) / total
    
    return lexicon 