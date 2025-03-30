"""
Προχωρημένες οπτικοποιήσεις μετρικών για το fine-tuned BERT μοντέλο 3 κλάσεων (Αρνητικό/Θετικό/Ουδέτερο).
Δημιουργεί:
1. ROC Curves (One-vs-Rest)
2. Precision-Recall Curves (One-vs-Rest)
3. Confusion Matrix
4. Λεπτομερή αναφορά μετρικών ανά κλάση
"""

import os
import numpy as np
import pandas as pd
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                           average_precision_score, confusion_matrix, 
                           classification_report, ConfusionMatrixDisplay)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import warnings

# Αγνόηση warnings για καθαρότερη έξοδο
warnings.filterwarnings('ignore')

# --- Διαδρομές ---
MODEL_DIR = os.path.join('models', 'fine_tuned_bert_3class')
TEST_DATA_PATH = os.path.join('data', 'processed', 'skroutz_3class_dataset.csv')
RESULTS_DIR = 'results'

# Χρώματα και ονόματα για τις κλάσεις - Διατηρώ τα ονόματα των κλάσεων στα ελληνικά
CLASS_NAMES = ['Negative', 'Positive', 'Neutral']
CLASS_COLORS = ['#EF5350', '#66BB6A', '#FFAB40']  # Κόκκινο, Πράσινο, Πορτοκαλί

# --- Βοηθητικές συναρτήσεις ---
def load_model_and_tokenizer(model_dir=MODEL_DIR):
    """Φορτώνει το μοντέλο και τον tokenizer."""
    print(f"Φόρτωση μοντέλου από τον φάκελο: {model_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        print(f"Μοντέλο με {model.config.num_labels} κλάσεις φορτώθηκε επιτυχώς.")
        return model, tokenizer
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση του μοντέλου: {e}")
        return None, None

def load_test_data(test_path=TEST_DATA_PATH, test_size=0.3, random_state=42):
    """Φορτώνει δεδομένα δοκιμής από CSV."""
    try:
        print(f"Φόρτωση δεδομένων δοκιμής από: {test_path}")
        if not os.path.exists(test_path):
            print(f"Το αρχείο δεδομένων δοκιμής δεν βρέθηκε: {test_path}")
            return None
        
        df = pd.read_csv(test_path)
        if 'label' not in df.columns:
            print("Λείπει η στήλη 'label' από τα δεδομένα.")
            return None
        
        if 'text' not in df.columns:
            print("Λείπει η στήλη 'text' από τα δεδομένα.")
            return None
        
        # Διαχωρισμός σε train/test
        from sklearn.model_selection import train_test_split
        _, df_test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['label']
        )
        
        print(f"Φορτώθηκαν {len(df_test)} δείγματα δοκιμής.")
        
        # Κατανομή κλάσεων στο test set
        class_dist = df_test['label'].value_counts().sort_index()
        print(f"Κατανομή κλάσεων (test): {class_dist.to_dict()}")
        
        return df_test
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση των δεδομένων δοκιμής: {e}")
        return None

def predict_proba(model, tokenizer, texts, device=-1):
    """Επιστρέφει τις προβλέψεις πιθανότητας του μοντέλου για τα κείμενα."""
    print(f"Πρόβλεψη πιθανοτήτων για {len(texts)} κείμενα...")
    
    # Δημιουργία του pipeline
    classifier = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_all_scores=True
    )
    
    # Πρόβλεψη
    try:
        # Προσθήκη ρητού περιορισμού μήκους σε 512 tokens (μέγιστο για BERT)
        results = classifier(texts, batch_size=16, truncation=True, max_length=512)
        
        # Μετατροπή των αποτελεσμάτων σε πίνακα Ν δειγμάτων x 3 κλάσεις
        probs = np.zeros((len(texts), model.config.num_labels))
        for i, result in enumerate(results):
            for class_result in result:
                label_id = int(class_result['label'].split('_')[1])
                probs[i, label_id] = class_result['score']
                
        print(f"Οι προβλέψεις ολοκληρώθηκαν επιτυχώς.")
        return probs
    except Exception as e:
        print(f"Σφάλμα κατά την πρόβλεψη: {e}")
        return None

def create_roc_curves(y_true, y_proba, class_names=CLASS_NAMES, colors=CLASS_COLORS):
    """Δημιουργεί καμπύλες ROC (one-vs-rest) για κάθε κλάση."""
    print("Δημιουργία καμπυλών ROC...")
    
    plt.figure(figsize=(12, 8))
    
    # Δημιουργία καμπύλης ROC για κάθε κλάση
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        # Δημιουργία one-hot encoded y_true για τη συγκεκριμένη κλάση
        y_true_binary = (y_true == i).astype(int)
        
        # Υπολογισμός ROC
        fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Σχεδίαση της καμπύλης
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})', color=color)
    
    # Προσθήκη της γραμμής αναφοράς (τυχαίος ταξινομητής)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Προσαρμογή του γραφήματος - Αγγλικοί όροι στις ετικέτες
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Αποθήκευση
    output_path = os.path.join(RESULTS_DIR, 'roc_curves.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Η καμπύλη ROC αποθηκεύτηκε στο: {output_path}")
    plt.close()

def create_pr_curves(y_true, y_proba, class_names=CLASS_NAMES, colors=CLASS_COLORS):
    """Δημιουργεί καμπύλες Precision-Recall (one-vs-rest) για κάθε κλάση."""
    print("Δημιουργία καμπυλών Precision-Recall...")
    
    plt.figure(figsize=(12, 8))
    
    # Δημιουργία καμπύλης P-R για κάθε κλάση
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        # Δημιουργία one-hot encoded y_true για τη συγκεκριμένη κλάση
        y_true_binary = (y_true == i).astype(int)
        
        # Υπολογισμός P-R
        precision, recall, _ = precision_recall_curve(y_true_binary, y_proba[:, i])
        average_precision = average_precision_score(y_true_binary, y_proba[:, i])
        
        # Σχεδίαση της καμπύλης
        plt.plot(recall, precision, lw=2, 
                 label=f'{class_name} (AP = {average_precision:.3f})', 
                 color=color)
    
    # Προσαρμογή του γραφήματος - Αγγλικοί όροι στις ετικέτες
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (One-vs-Rest)')
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Αποθήκευση
    output_path = os.path.join(RESULTS_DIR, 'precision_recall_curves.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Η καμπύλη Precision-Recall αποθηκεύτηκε στο: {output_path}")
    plt.close()

def create_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES):
    """Δημιουργεί πίνακα σύγχυσης (confusion matrix)."""
    print("Δημιουργία πίνακα σύγχυσης...")
    
    # Υπολογισμός πίνακα σύγχυσης
    cm = confusion_matrix(y_true, y_pred)
    
    # Κανονικοποίηση για ποσοστά
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Δημιουργία γραφήματος με το seaborn
    plt.figure(figsize=(10, 8))
    
    # Πίνακας σύγχυσης με ποσοστά
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    
    # Αγγλικοί όροι στις ετικέτες
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix (Normalized)')
    
    # Αποθήκευση
    output_path = os.path.join(RESULTS_DIR, 'confusion_matrix_normalized.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Ο κανονικοποιημένος πίνακας σύγχυσης αποθηκεύτηκε στο: {output_path}")
    
    # Δημιουργία πίνακα σύγχυσης με απόλυτους αριθμούς
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    
    # Αγγλικοί όροι στις ετικέτες
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix (Absolute)')
    
    # Αποθήκευση
    output_path = os.path.join(RESULTS_DIR, 'confusion_matrix_absolute.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Ο πίνακας σύγχυσης σε απόλυτους αριθμούς αποθηκεύτηκε στο: {output_path}")
    plt.close()

def create_class_metrics_report(y_true, y_pred, class_names=CLASS_NAMES):
    """Δημιουργεί αναφορά μετρικών ανά κλάση και την αποθηκεύει σε αρχείο."""
    print("Δημιουργία αναφοράς μετρικών ανά κλάση...")
    
    # Δημιουργία του classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Μετατροπή σε DataFrame για ευκολότερο χειρισμό
    df_report = pd.DataFrame(report).transpose()
    
    # Αποθήκευση σε CSV
    output_path_csv = os.path.join(RESULTS_DIR, 'classification_report.csv')
    df_report.to_csv(output_path_csv)
    print(f"Η αναφορά μετρικών αποθηκεύτηκε στο: {output_path_csv}")
    
    # Αποθήκευση σε μορφή κειμένου
    with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    
    # Δημιουργία οπτικοποίησης για precision, recall, f1-score
    metrics = ['precision', 'recall', 'f1-score']
    df_plot = df_report.loc[class_names, metrics]
    
    plt.figure(figsize=(12, 6))
    df_plot.plot(kind='bar', rot=0, colormap='viridis')
    plt.title('Metrics per Class')
    plt.ylabel('Value')
    plt.xlabel('Class')
    plt.ylim(0, 1.0)
    plt.legend(title='Metric')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Αποθήκευση
    output_path_plot = os.path.join(RESULTS_DIR, 'class_metrics.png')
    plt.tight_layout()
    plt.savefig(output_path_plot, dpi=300)
    print(f"Το γράφημα μετρικών ανά κλάση αποθηκεύτηκε στο: {output_path_plot}")
    plt.close()

def create_prediction_distribution(y_proba, class_names=CLASS_NAMES, colors=CLASS_COLORS):
    """Δημιουργεί γράφημα κατανομής των προβλέψεων πιθανότητας."""
    print("Δημιουργία γραφήματος κατανομής προβλέψεων...")
    
    plt.figure(figsize=(12, 6))
    
    # Δημιουργία subplots για κάθε κλάση
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for i, (class_name, color, ax) in enumerate(zip(class_names, colors, axes)):
        # Ιστόγραμμα των πιθανοτήτων για τη συγκεκριμένη κλάση
        ax.hist(y_proba[:, i], bins=20, alpha=0.7, color=color)
        ax.set_title(f"Probability Distribution: {class_name}")
        ax.set_xlabel("Probability")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Προσθήκη κατακόρυφης γραμμής στο 0.5
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    
    # Ο πρώτος άξονας έχει και ετικέτα Y
    axes[0].set_ylabel("Frequency")
    
    # Αποθήκευση
    output_path = os.path.join(RESULTS_DIR, 'prediction_distribution.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Το γράφημα κατανομής προβλέψεων αποθηκεύτηκε στο: {output_path}")
    plt.close()

def create_learning_curve_visualization():
    """Δημιουργεί οπτικοποίηση της καμπύλης μάθησης βάσει του trainer_state.json."""
    print("Δημιουργία οπτικοποίησης καμπύλης μάθησης...")
    
    state_file = os.path.join(MODEL_DIR, 'trainer_state.json')
    
    if not os.path.exists(state_file):
        print(f"Το αρχείο {state_file} δεν βρέθηκε. Παραλείπεται η οπτικοποίηση της καμπύλης μάθησης.")
        return
    
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
            
        if 'log_history' not in state:
            print("Δεν βρέθηκε ιστορικό καταγραφής στο αρχείο κατάστασης.")
            return
            
        log_history = state['log_history']
        
        # Συλλογή δεδομένων
        train_steps = []
        train_loss = []
        eval_steps = []
        eval_loss = []
        eval_accuracy = []
        
        for log in log_history:
            if 'loss' in log and 'step' in log:
                train_steps.append(log['step'])
                train_loss.append(log['loss'])
            elif 'eval_loss' in log and 'step' in log:
                eval_steps.append(log['step'])
                eval_loss.append(log['eval_loss'])
                if 'eval_accuracy' in log:
                    eval_accuracy.append(log['eval_accuracy'])
        
        # Δημιουργία γραφήματος καμπύλης μάθησης
        plt.figure(figsize=(14, 8))
        
        # Training loss
        if train_steps and train_loss:
            plt.plot(train_steps, train_loss, 'b-', label='Training Loss', alpha=0.7)
            
        # Validation loss
        if eval_steps and eval_loss:
            plt.plot(eval_steps, eval_loss, 'r-', label='Validation Loss', alpha=0.7)
            
        # Accuracy
        if eval_steps and eval_accuracy:
            ax2 = plt.twinx()
            ax2.plot(eval_steps, eval_accuracy, 'g-', label='Validation Accuracy', alpha=0.7)
            ax2.set_ylim([0, 1.1])
            ax2.set_ylabel('Accuracy', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            
            # Προσθήκη των ετικετών του accuracy στο legend του κύριου γραφήματος
            lines, labels = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(lines + lines2, labels + labels2, loc='best')
        else:
            plt.legend(loc='best')
            
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title('Learning Curve: Loss and Accuracy over Training Steps')
        
        # Αποθήκευση
        output_path = os.path.join(RESULTS_DIR, 'learning_curve.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Η καμπύλη μάθησης αποθηκεύτηκε στο: {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"Σφάλμα κατά την οπτικοποίηση της καμπύλης μάθησης: {e}")

def main():
    """Κύρια συνάρτηση που εκτελεί τη συνολική ανάλυση."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Έλεγχος για CUDA
    device = 0 if torch.cuda.is_available() else -1
    print(f"Χρήση {'GPU' if device == 0 else 'CPU'} για υπολογισμούς.")
    
    # 1. Φόρτωση μοντέλου
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        return
    
    # 2. Φόρτωση δεδομένων δοκιμής
    df_test = load_test_data()
    if df_test is None:
        return
    
    X_test = df_test['text'].tolist()
    y_test = df_test['label'].astype(int).values
    
    # 3. Πρόβλεψη πιθανοτήτων
    y_proba = predict_proba(model, tokenizer, X_test, device)
    if y_proba is None:
        return
    
    # 4. Υπολογισμός των προβλέψεων κλάσεων
    y_pred = np.argmax(y_proba, axis=1)
    
    # 5. Δημιουργία οπτικοποιήσεων
    create_roc_curves(y_test, y_proba)
    create_pr_curves(y_test, y_proba)
    create_confusion_matrix(y_test, y_pred)
    create_class_metrics_report(y_test, y_pred)
    create_prediction_distribution(y_proba)
    create_learning_curve_visualization()
    
    print("\nΗ ανάλυση ολοκληρώθηκε. Όλα τα αποτελέσματα αποθηκεύτηκαν στον φάκελο:", RESULTS_DIR)

if __name__ == "__main__":
    # Έλεγχος εγκατάστασης απαραίτητων βιβλιοθηκών
    required_packages = ['torch', 'transformers', 'pandas', 'numpy', 
                         'matplotlib', 'sklearn', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            # Ειδική μετατροπή για το scikit-learn
            if package == 'sklearn':
                missing_packages.append('scikit-learn')
            else:
                missing_packages.append(package)
    
    if missing_packages:
        print("Λείπουν τα ακόλουθα πακέτα:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nΠαρακαλώ εγκαταστήστε τα με:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        main() 