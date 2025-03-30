"""
Μοντέλα ανάλυσης συναισθημάτων για ελληνικές κριτικές
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

from src.utils.config import MODELS_DIR, MAX_FEATURES, MIN_DF
from src.utils.utils import save_model, load_model, evaluate_model
from src.preprocessing.text_preprocessing import count_emotion_words


class SentimentClassifier:
    """Ταξινομητής συναισθημάτων για ελληνικές κριτικές"""
    
    def __init__(self, model_type='logistic', use_emotions=True):
        """
        Αρχικοποίηση του ταξινομητή
        
        Args:
            model_type (str): Ο τύπος του μοντέλου ('logistic', 'svm', 'random_forest', 'naive_bayes')
            use_emotions (bool): Αν θα χρησιμοποιηθούν τα χαρακτηριστικά συναισθημάτων
        """
        self.model_type = model_type
        self.use_emotions = use_emotions
        self.model = None
        self.vectorizer = None
        self.emotion_columns = None
    
    def _create_model(self):
        """
        Δημιουργία του μοντέλου με βάση τον επιλεγμένο τύπο
        
        Returns:
            model: Το μοντέλο που δημιουργήθηκε
        """
        if self.model_type == 'logistic':
            return LogisticRegression(
                C=10, 
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        elif self.model_type == 'svm':
            return SVC(
                C=10, 
                kernel='linear', 
                class_weight='balanced',
                random_state=42,
                probability=True
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'naive_bayes':
            return MultinomialNB(alpha=0.1)
        else:
            raise ValueError(f"Μη υποστηριζόμενος τύπος μοντέλου: {self.model_type}")
    
    def _create_vectorizer(self):
        """
        Δημιουργία του TF-IDF vectorizer
        
        Returns:
            vectorizer: Ο vectorizer που δημιουργήθηκε
        """
        return TfidfVectorizer(
            max_features=MAX_FEATURES,
            min_df=MIN_DF,
            ngram_range=(1, 2),
            stop_words=None,  # Χρησιμοποιούμε τη δική μας λίστα stopwords στο preprocessing
            use_idf=True,
            sublinear_tf=True
        )
    
    def fit(self, X_train, y_train):
        """
        Εκπαίδευση του μοντέλου
        
        Args:
            X_train: Τα δεδομένα εκπαίδευσης
            y_train: Οι ετικέτες εκπαίδευσης
        
        Returns:
            self: Το εκπαιδευμένο μοντέλο
        """
        print(f"Εκπαίδευση μοντέλου τύπου: {self.model_type}")
        
        # Δημιουργία του vectorizer και μετατροπή του κειμένου
        self.vectorizer = self._create_vectorizer()
        X_text = self.vectorizer.fit_transform(X_train['processed_review'])
        
        # Χρήση χαρακτηριστικών συναισθημάτων αν έχει οριστεί
        if self.use_emotions:
            # Εύρεση των στηλών με τα χαρακτηριστικά συναισθημάτων
            self.emotion_columns = [col for col in X_train.columns if col.startswith('emotion_')]
            
            if self.emotion_columns:
                # Συνδυασμός χαρακτηριστικών κειμένου με χαρακτηριστικά συναισθημάτων
                X_emotions = X_train[self.emotion_columns].values
                X_train_features = np.hstack((X_text.toarray(), X_emotions))
            else:
                X_train_features = X_text
        else:
            X_train_features = X_text
        
        # Δημιουργία και εκπαίδευση του μοντέλου
        self.model = self._create_model()
        self.model.fit(X_train_features, y_train)
        
        return self
    
    def predict(self, X):
        """
        Πρόβλεψη συναισθήματος για νέα δεδομένα
        
        Args:
            X: Τα δεδομένα για πρόβλεψη
        
        Returns:
            np.array: Οι προβλέψεις
        """
        if self.model is None:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί ακόμα. Καλέστε πρώτα τη μέθοδο fit().")
        
        # Μετατροπή του κειμένου
        X_text = self.vectorizer.transform(X['processed_review'])
        
        # Χρήση χαρακτηριστικών συναισθημάτων αν έχει οριστεί
        if self.use_emotions and self.emotion_columns:
            X_emotions = X[self.emotion_columns].values
            X_features = np.hstack((X_text.toarray(), X_emotions))
        else:
            X_features = X_text
        
        return self.model.predict(X_features)
        
    def predict_proba(self, X):
        """
        Πρόβλεψη πιθανοτήτων συναισθήματος για νέα δεδομένα
        
        Args:
            X: Τα δεδομένα για πρόβλεψη
        
        Returns:
            np.array: Οι πιθανότητες προβλέψεων
        """
        if self.model is None:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί ακόμα. Καλέστε πρώτα τη μέθοδο fit().")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Το μοντέλο τύπου {self.model_type} δεν υποστηρίζει πιθανοτικές προβλέψεις.")
        
        # Μετατροπή του κειμένου
        X_text = self.vectorizer.transform(X['processed_review'])
        
        # Χρήση χαρακτηριστικών συναισθημάτων αν έχει οριστεί
        if self.use_emotions and self.emotion_columns:
            X_emotions = X[self.emotion_columns].values
            X_features = np.hstack((X_text.toarray(), X_emotions))
        else:
            X_features = X_text
        
        return self.model.predict_proba(X_features)
    
    def evaluate(self, X_test, y_test):
        """
        Αξιολόγηση του μοντέλου
        
        Args:
            X_test: Τα δεδομένα αξιολόγησης
            y_test: Οι ετικέτες αξιολόγησης
        
        Returns:
            dict: Αποτελέσματα αξιολόγησης
        """
        model_name = f"{self.model_type}_{'with' if self.use_emotions else 'without'}_emotions"
        results = evaluate_model(self.model, X_test, y_test, self.vectorizer, model_name)
        return results
    
    def save(self, file_prefix=None):
        """
        Αποθήκευση του μοντέλου
        
        Args:
            file_prefix (str): Πρόθεμα για το όνομα του αρχείου (προαιρετικό)
        """
        if self.model is None:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί ακόμα. Καλέστε πρώτα τη μέθοδο fit().")
        
        prefix = file_prefix if file_prefix else self.model_type
        model_name = f"{prefix}_{'with' if self.use_emotions else 'without'}_emotions"
        
        save_model(self.model, model_name, self.vectorizer)
        
        # Αποθήκευση των στηλών συναισθημάτων αν υπάρχουν
        if self.use_emotions and self.emotion_columns:
            emotion_columns_path = os.path.join(MODELS_DIR, f"{model_name}_emotion_columns.txt")
            with open(emotion_columns_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.emotion_columns))
    
    def load(self, file_prefix=None):
        """
        Φόρτωση του μοντέλου
        
        Args:
            file_prefix (str): Πρόθεμα για το όνομα του αρχείου (προαιρετικό)
        
        Returns:
            self: Το φορτωμένο μοντέλο
        """
        prefix = file_prefix if file_prefix else self.model_type
        model_name = f"{prefix}_{'with' if self.use_emotions else 'without'}_emotions"
        
        try:
            self.model, self.vectorizer = load_model(model_name, with_vectorizer=True)
            
            # Φόρτωση των στηλών συναισθημάτων αν υπάρχουν
            if self.use_emotions:
                emotion_columns_path = os.path.join(MODELS_DIR, f"{model_name}_emotion_columns.txt")
                if os.path.exists(emotion_columns_path):
                    with open(emotion_columns_path, 'r', encoding='utf-8') as f:
                        self.emotion_columns = f.read().splitlines()
            
            return self
        except:
            raise ValueError(f"Δεν ήταν δυνατή η φόρτωση του μοντέλου {model_name}")
    
    def predict_text(self, text):
        """
        Πρόβλεψη συναισθήματος για νέο κείμενο
        
        Args:
            text (str): Το κείμενο προς ανάλυση
        
        Returns:
            tuple: (prediction, probabilities, emotions)
        """
        from src.preprocessing.text_preprocessing import process_text_for_analysis, count_emotion_words
        
        if self.model is None:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί ακόμα. Καλέστε πρώτα τη μέθοδο fit().")
        
        # Προεπεξεργασία κειμένου
        processed_text = process_text_for_analysis(text)
        
        # Δημιουργία DataFrame
        data = pd.DataFrame({
            'processed_review': [processed_text]
        })
        
        # Προσθήκη χαρακτηριστικών συναισθημάτων αν χρειάζεται
        emotions = {}
        if self.use_emotions and self.emotion_columns:
            emotions = count_emotion_words(text)
            for emotion, count in emotions.items():
                data[f'emotion_{emotion}'] = count
        
        # Πρόβλεψη
        prediction = self.predict(data)[0]
        
        # Πιθανότητες
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.predict_proba(data)[0]
        
        return prediction, probabilities, emotions 