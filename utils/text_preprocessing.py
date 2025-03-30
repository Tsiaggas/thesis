#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Αυτό το module περιέχει συναρτήσεις για την προεπεξεργασία κειμένου
που χρησιμοποιούνται στην ανάλυση συναισθημάτων.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Φορτώνουμε τα απαραίτητα δεδομένα NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.download('punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Λίστα από stop words για τα ελληνικά
GREEK_STOP_WORDS = set([
    'ο', 'η', 'το', 'οι', 'τα', 'του', 'της', 'των', 'τον', 'την', 'και',
    'κι', 'κ', 'ή', 'είτε', 'να', 'θα', 'που', 'πως', 'όταν', 'μη', 'μην',
    'μα', 'αλλά', 'όμως', 'ωστόσο', 'αν', 'εάν', 'και', 'κι', 'ούτε', 'μήτε',
    'δεν', 'μη', 'από', 'για', 'σε', 'με', 'ως', 'παρά', 'αντί', 'κατά',
    'μετά', 'μέχρι', 'ώσπου', 'προς', 'επί', 'περί', 'όπως', 'έτσι',
    'είμαι', 'είσαι', 'είναι', 'είμαστε', 'είστε', 'ήταν', 'ήμουν',
    'έχω', 'έχεις', 'έχει', 'έχουμε', 'έχετε', 'έχουν',
    'αυτός', 'αυτή', 'αυτό', 'αυτοί', 'αυτές', 'αυτά',
    'εγώ', 'εσύ', 'εμείς', 'εσείς', 'αυτοί', 'αυτές',
    'τι', 'ποιος', 'ποια', 'ποιο', 'ποιοι', 'ποιες', 'ποια',
    'πολύ', 'λίγο', 'κάπως'
])

def preprocess_text(text):
    """
    Εφαρμόζει προεπεξεργασία σε ένα κείμενο.
    
    Η προεπεξεργασία περιλαμβάνει:
    1. Αφαίρεση ειδικών χαρακτήρων και αριθμών
    2. Μετατροπή σε πεζά γράμματα
    3. Αφαίρεση σημείων στίξης
    4. Αφαίρεση stop words
    5. Tokenization
    
    Args:
        text (str): Το κείμενο προς προεπεξεργασία
        
    Returns:
        str: Το προεπεξεργασμένο κείμενο
    """
    if not isinstance(text, str):
        return ""
    
    # Μετατροπή σε πεζά
    text = text.lower()
    
    # Αφαίρεση URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Αφαίρεση HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Αφαίρεση αριθμών
    text = re.sub(r'\d+', '', text)
    
    # Αφαίρεση σημείων στίξης
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # Χειροκίνητη tokenization για να αποφύγουμε το punkt
    tokens = text.split()
    
    # Αφαίρεση stop words (Ελληνικά και Αγγλικά)
    english_stopwords = set(stopwords.words('english'))
    stop_words = GREEK_STOP_WORDS.union(english_stopwords)
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization (μόνο για αγγλικό κείμενο)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Επανένωση tokens σε ένα string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text 