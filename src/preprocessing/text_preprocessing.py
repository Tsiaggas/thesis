"""
Συναρτήσεις προεπεξεργασίας κειμένου για ελληνικές κριτικές
"""

import re
import unicodedata
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from src.utils.config import EMOTION_CATEGORIES

# Κατέβασμα των απαραίτητων resources από το NLTK
def download_nltk_resources():
    """Κατεβάζει τα απαραίτητα resources για το NLTK"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


def remove_accents(text):
    """
    Αφαίρεση τόνων από το κείμενο
    
    Args:
        text (str): Το κείμενο προς επεξεργασία
        
    Returns:
        str: Το κείμενο χωρίς τόνους
    """
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')


def clean_text(text):
    """
    Βασικός καθαρισμός κειμένου
    
    Args:
        text (str): Το κείμενο προς επεξεργασία
        
    Returns:
        str: Το καθαρισμένο κείμενο
    """
    if not isinstance(text, str):
        return ''
    
    # Μετατροπή σε πεζά
    text = text.lower()
    
    # Αφαίρεση URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Αφαίρεση HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Αφαίρεση ειδικών χαρακτήρων και αριθμών
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Αφαίρεση πολλαπλών κενών
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def create_greek_stopwords():
    """
    Δημιουργία λίστας ελληνικών stopwords
    
    Returns:
        list: Λίστα με ελληνικές stopwords
    """
    greek_stopwords = [
        'ο', 'η', 'το', 'οι', 'τα', 'του', 'της', 'των', 'τον', 'την', 'και',
        'κι', 'κ', 'ή', 'είτε', 'να', 'μη', 'μην', 'μεν', 'δε', 'δεν', 'όχι',
        'θα', 'ας', 'ως', 'αν', 'από', 'για', 'προς', 'με', 'σε', 'εκτός', 'μέχρι',
        'παρά', 'αντί', 'μετά', 'κατά', 'μετά', 'πριν', 'εγώ', 'εσύ', 'αυτός', 'αυτή',
        'αυτό', 'εμείς', 'εσείς', 'αυτοί', 'αυτές', 'αυτά', 'που', 'πως', 'ποιος',
        'ποια', 'ποιο', 'ποιοι', 'ποιες', 'ποια', 'έτσι', 'αλλά', 'όμως', 'ενώ', 'μα',
        'επειδή', 'γιατί', 'έχω', 'έχεις', 'έχει', 'έχουμε', 'έχετε', 'έχουν', 'είμαι', 
        'είσαι', 'είναι', 'είμαστε', 'είστε', 'ήμουν', 'ήσουν', 'ήταν', 'ήμαστε', 'ήσαστε',
        'ήταν', 'θα', 'να', 'δεν', 'δε', 'μη', 'μην', 'επίσης', 'ακόμα', 'ακόμη', 'καλά',
        'καλό', 'καλή', 'καλοί', 'καλές', 'πολύ', 'πολλά', 'πολλοί', 'πολλές', 'λίγο',
        'λίγα', 'λίγοι', 'λίγες', 'κάθε', 'μερικοί', 'μερικές', 'μερικά', 'όλος', 'όλη',
        'όλο', 'όλοι', 'όλες', 'όλα', 'ένας', 'μία', 'ένα', 'πρώτος', 'πρώτη', 'πρώτο',
        'πρώτοι', 'πρώτες', 'πρώτα', 'δεύτερος', 'δεύτερη', 'δεύτερο', 'δεύτεροι', 'δεύτερες',
        'δεύτερα', 'τρίτος', 'τρίτη', 'τρίτο', 'τρίτοι', 'τρίτες', 'τρίτα', 'τέταρτος', 
        'τέταρτη', 'τέταρτο', 'αυτούς', 'αυτήν', 'αυτόν', 'αυτών', 'αυτές', 'αυτά', 'εκείνο',
        'εκείνη', 'εκείνος', 'εκείνοι', 'εκείνες', 'εκείνα', 'εκείνων', 'εδώ', 'εκεί', 'τώρα', 
        'αύριο', 'χθες', 'σήμερα', 'απόψε', 'χθες', 'τότε', 'πάνω', 'κάτω', 'μπροστά', 'πίσω',
        'μέσα', 'έξω', 'ναι', 'όχι', 'μάλλον', 'ίσως', 'πιθανόν', 'ξανά', 'πάλι', 'ακόμα',
        'ακόμη', 'μόνο', 'περισσότερο', 'λιγότερο', 'αρκετά', 'απλώς'
    ]
    return greek_stopwords


def remove_stopwords(text, stopwords=None):
    """
    Αφαίρεση stopwords από το κείμενο
    
    Args:
        text (str): Το κείμενο προς επεξεργασία
        stopwords (list, optional): Λίστα με stopwords. 
                                  Αν δεν δοθεί, χρησιμοποιούνται οι προκαθορισμένες.
        
    Returns:
        str: Το κείμενο χωρίς stopwords
    """
    if stopwords is None:
        stopwords = create_greek_stopwords()
    
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stopwords]
    return ' '.join(filtered_tokens)


def process_text_for_analysis(text):
    """
    Πλήρης προεπεξεργασία κειμένου για ανάλυση
    
    Args:
        text (str): Το κείμενο προς επεξεργασία
        
    Returns:
        str: Το επεξεργασμένο κείμενο
    """
    # Καθαρισμός κειμένου
    text = clean_text(text)
    
    # Αφαίρεση stopwords
    text = remove_stopwords(text)
    
    return text


def load_and_preprocess_data(file_path):
    """
    Φόρτωση και προεπεξεργασία των δεδομένων από το αρχικό αρχείο
    
    Args:
        file_path (str): Η διαδρομή προς το αρχείο δεδομένων
        
    Returns:
        pd.DataFrame: Το προεπεξεργασμένο DataFrame
    """
    # Φόρτωση των δεδομένων
    data = pd.read_excel(file_path)
    
    # Αφαίρεση διπλότυπων εγγραφών
    data = data.drop_duplicates(subset=['review'])
    
    # Αφαίρεση κενών τιμών
    data = data.dropna(subset=['review'])
    
    # Προεπεξεργασία κειμένου
    data['processed_review'] = data['review'].apply(process_text_for_analysis)
    
    # Προσθήκη στήλης με το μήκος κάθε κριτικής
    data['length'] = data['review'].apply(len)
    
    return data


def count_emotion_words(text, emotion_categories=EMOTION_CATEGORIES):
    """
    Μέτρηση λέξεων συναισθημάτων σε κείμενο
    
    Args:
        text (str): Το κείμενο προς ανάλυση
        emotion_categories (dict): Λεξικό με κατηγορίες συναισθημάτων και λέξεις-κλειδιά
        
    Returns:
        dict: Λεξικό με το πλήθος των λέξεων ανά κατηγορία συναισθήματος
    """
    result = {emotion: 0 for emotion in emotion_categories}
    
    # Μετατροπή σε πεζά για ομοιόμορφη σύγκριση
    text = text.lower()
    
    # Αφαίρεση τόνων
    text_without_accents = remove_accents(text)
    
    # Μέτρηση λέξεων ανά κατηγορία συναισθήματος
    for emotion, words in emotion_categories.items():
        # Προσθήκη εκδοχών χωρίς τόνους
        words_without_accents = [remove_accents(word) for word in words]
        all_words = set(words + words_without_accents)
        
        # Μέτρηση εμφανίσεων
        for word in all_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            result[emotion] += len(re.findall(pattern, text)) + len(re.findall(pattern, text_without_accents))
    
    return result


if __name__ == "__main__":
    # Δοκιμαστική εκτέλεση
    download_nltk_resources()
    test_text = "Είμαι πολύ ευχαριστημένος με την αγορά. Εξαιρετική ποιότητα και άψογη εξυπηρέτηση!"
    processed = process_text_for_analysis(test_text)
    print(f"Αρχικό κείμενο: {test_text}")
    print(f"Επεξεργασμένο κείμενο: {processed}")
    
    emotions = count_emotion_words(test_text)
    print(f"Συναισθήματα: {emotions}") 