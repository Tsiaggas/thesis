"""
Εκπαίδευση και αξιολόγηση μοντέλων ανάλυσης συναισθημάτων
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from src.models.sentiment_model import SentimentClassifier
from src.utils.config import MODELS_DIR
from src.utils.utils import plot_confusion_matrix, plot_roc_curve


def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Εκπαίδευση και αξιολόγηση πολλαπλών μοντέλων
    
    Args:
        X_train: Δεδομένα εκπαίδευσης
        X_val: Δεδομένα επικύρωσης
        X_test: Δεδομένα δοκιμής
        y_train: Ετικέτες εκπαίδευσης
        y_val: Ετικέτες επικύρωσης
        y_test: Ετικέτες δοκιμής
    
    Returns:
        dict: Αποτελέσματα αξιολόγησης για κάθε μοντέλο
    """
    model_types = ['logistic', 'svm', 'random_forest', 'naive_bayes']
    emotion_options = [True, False]
    
    results = {}
    best_model = None
    best_f1 = 0
    
    for model_type in model_types:
        for use_emotions in emotion_options:
            print(f"\n{'='*80}")
            print(f"Εκπαίδευση μοντέλου: {model_type} {'με' if use_emotions else 'χωρίς'} χαρακτηριστικά συναισθημάτων")
            print(f"{'='*80}")
            
            # Δημιουργία και εκπαίδευση του μοντέλου
            model = SentimentClassifier(model_type=model_type, use_emotions=use_emotions)
            model.fit(X_train, y_train)
            
            # Αξιολόγηση στο validation set
            val_results = model.evaluate(X_val, y_val)
            
            # Αποθήκευση αποτελεσμάτων
            model_key = f"{model_type}_{'with' if use_emotions else 'without'}_emotions"
            results[model_key] = val_results
            
            # Έλεγχος αν είναι το καλύτερο μοντέλο
            if val_results['f1'] > best_f1:
                best_f1 = val_results['f1']
                best_model = model
                print(f"Νέο καλύτερο μοντέλο: {model_key} με F1 score: {best_f1:.4f}")
            
            # Αποθήκευση του μοντέλου
            model.save()
    
    # Αξιολόγηση του καλύτερου μοντέλου στο test set
    if best_model:
        print(f"\n{'='*80}")
        print(f"Αξιολόγηση του καλύτερου μοντέλου στο test set")
        print(f"{'='*80}")
        
        test_results = best_model.evaluate(X_test, y_test)
        results['best_model_test'] = test_results
        
        # Αποθήκευση του καλύτερου μοντέλου
        best_model.save('best_model')
        
        # Σχεδιασμός πίνακα σύγχυσης
        if 'confusion_matrix' in test_results:
            plot_confusion_matrix(
                test_results['confusion_matrix'],
                title=f"Πίνακας Σύγχυσης - Καλύτερο Μοντέλο",
                save_path=os.path.join(MODELS_DIR, 'best_model_confusion_matrix.png')
            )
        
        # Σχεδιασμός ROC καμπύλης
        if 'roc_auc' in test_results:
            plot_roc_curve(
                test_results['fpr'],
                test_results['tpr'],
                test_results['roc_auc'],
                title=f"ROC Καμπύλη - Καλύτερο Μοντέλο",
                save_path=os.path.join(MODELS_DIR, 'best_model_roc_curve.png')
            )
    
    return results, best_model


def optimize_hyperparameters(X_train, y_train, X_val, y_val, model_type='logistic', use_emotions=True):
    """
    Βελτιστοποίηση υπερπαραμέτρων για το επιλεγμένο μοντέλο
    
    Args:
        X_train: Δεδομένα εκπαίδευσης
        y_train: Ετικέτες εκπαίδευσης
        X_val: Δεδομένα επικύρωσης
        y_val: Ετικέτες επικύρωσης
        model_type (str): Ο τύπος του μοντέλου
        use_emotions (bool): Αν θα χρησιμοποιηθούν τα χαρακτηριστικά συναισθημάτων
    
    Returns:
        SentimentClassifier: Το βελτιστοποιημένο μοντέλο
    """
    print(f"\n{'='*80}")
    print(f"Βελτιστοποίηση υπερπαραμέτρων για το μοντέλο: {model_type}")
    print(f"{'='*80}")
    
    # Αρχικοποίηση του μοντέλου
    model = SentimentClassifier(model_type=model_type, use_emotions=use_emotions)
    
    # Εκπαίδευση του αρχικού μοντέλου για αναφορά
    model.fit(X_train, y_train)
    initial_results = model.evaluate(X_val, y_val)
    print(f"Απόδοση αρχικού μοντέλου - F1 Score: {initial_results['f1']:.4f}")
    
    # Ορισμός πλέγματος υπερπαραμέτρων με βάση τον τύπο του μοντέλου
    if model_type == 'logistic':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'class_weight': [None, 'balanced'],
            'solver': ['liblinear', 'saga']
        }
    elif model_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
            'class_weight': [None, 'balanced']
        }
    elif model_type == 'random_forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        }
    elif model_type == 'naive_bayes':
        param_grid = {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
        }
    else:
        raise ValueError(f"Μη υποστηριζόμενος τύπος μοντέλου: {model_type}")
    
    # Εκπαίδευση μοντέλων με διαφορετικές υπερπαραμέτρους
    best_params = None
    best_score = 0
    best_model = None
    
    # Δημιουργία συνδυασμών υπερπαραμέτρων
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    print(f"Δοκιμή {len(param_combinations)} συνδυασμών υπερπαραμέτρων...")
    
    for i, combination in enumerate(param_combinations):
        params = dict(zip(param_names, combination))
        print(f"\nΔοκιμή συνδυασμού {i+1}/{len(param_combinations)}: {params}")
        
        # Δημιουργία μοντέλου με τις τρέχουσες υπερπαραμέτρους
        current_model = SentimentClassifier(model_type=model_type, use_emotions=use_emotions)
        
        # Προσαρμογή των υπερπαραμέτρων του μοντέλου
        if model_type == 'logistic':
            current_model._create_model = lambda: LogisticRegression(
                C=params['C'],
                class_weight=params['class_weight'],
                solver=params['solver'],
                random_state=42,
                max_iter=1000
            )
        elif model_type == 'svm':
            current_model._create_model = lambda: SVC(
                C=params['C'],
                kernel=params['kernel'],
                gamma=params['gamma'],
                class_weight=params['class_weight'],
                random_state=42,
                probability=True
            )
        elif model_type == 'random_forest':
            current_model._create_model = lambda: RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                class_weight=params['class_weight'],
                random_state=42
            )
        elif model_type == 'naive_bayes':
            current_model._create_model = lambda: MultinomialNB(
                alpha=params['alpha']
            )
        
        # Εκπαίδευση και αξιολόγηση
        try:
            current_model.fit(X_train, y_train)
            results = current_model.evaluate(X_val, y_val)
            
            # Έλεγχος για καλύτερο σκορ
            if results['f1'] > best_score:
                best_score = results['f1']
                best_params = params
                best_model = current_model
                print(f"Νέο καλύτερο μοντέλο! F1 Score: {best_score:.4f}")
        except Exception as e:
            print(f"Σφάλμα κατά την εκπαίδευση με τις υπερπαραμέτρους {params}: {str(e)}")
    
    print(f"\nΒέλτιστες υπερπαράμετροι: {best_params}")
    print(f"Καλύτερο F1 Score: {best_score:.4f}")
    
    # Αποθήκευση του βελτιστοποιημένου μοντέλου
    if best_model:
        best_model.save(f"optimized_{model_type}")
    
    return best_model


def compare_models_performance(results):
    """
    Σύγκριση της απόδοσης διαφορετικών μοντέλων
    
    Args:
        results (dict): Αποτελέσματα αξιολόγησης για κάθε μοντέλο
    """
    # Εξαγωγή μετρικών απόδοσης
    model_names = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for model_name, model_results in results.items():
        if model_name != 'best_model_test':  # Εξαίρεση του test set του καλύτερου μοντέλου
            model_names.append(model_name)
            accuracy_scores.append(model_results['accuracy'])
            precision_scores.append(model_results['precision'])
            recall_scores.append(model_results['recall'])
            f1_scores.append(model_results['f1'])
    
    # Δημιουργία γραφήματος
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bar_width = 0.2
    index = np.arange(len(model_names))
    
    # Σχεδιασμός των ράβδων
    ax.bar(index, accuracy_scores, bar_width, label='Accuracy')
    ax.bar(index + bar_width, precision_scores, bar_width, label='Precision')
    ax.bar(index + 2*bar_width, recall_scores, bar_width, label='Recall')
    ax.bar(index + 3*bar_width, f1_scores, bar_width, label='F1 Score')
    
    # Προσθήκη ετικετών και τίτλου
    ax.set_xlabel('Μοντέλο')
    ax.set_ylabel('Βαθμολογία')
    ax.set_title('Σύγκριση Απόδοσης Μοντέλων')
    ax.set_xticks(index + 1.5*bar_width)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'model_comparison.png'))
    plt.show()


if __name__ == "__main__":
    from src.preprocessing.data_preparation import load_or_prepare_data
    
    # Φόρτωση δεδομένων
    X_train, X_val, X_test, y_train, y_val, y_test = load_or_prepare_data()
    
    # Εκπαίδευση και αξιολόγηση μοντέλων
    results, best_model = train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Σύγκριση απόδοσης μοντέλων
    compare_models_performance(results)
    
    # Βελτιστοποίηση του καλύτερου μοντέλου
    if best_model:
        optimized_model = optimize_hyperparameters(
            X_train, y_train, X_val, y_val, 
            model_type=best_model.model_type, 
            use_emotions=best_model.use_emotions
        ) 