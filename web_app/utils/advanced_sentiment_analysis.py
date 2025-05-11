#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Προηγμένες μέθοδοι ανάλυσης συναισθημάτων
Περιλαμβάνει:
1. VADER - Λεξικό συναισθημάτων (valence-based)
2. NRCLex - Λεξικό με συγκεκριμένα συναισθήματα (χαρά, εμπιστοσύνη, κλπ)
3. LSA - Λανθάνουσα Σημασιολογική Ανάλυση
"""

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
import re
import nltk
from googletrans import Translator
from textblob import TextBlob

# Κατέβασμα απαραίτητων δεδομένων από NLTK αν χρειαστεί
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Αγνόηση μη σημαντικών προειδοποιήσεων
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class AdvancedSentimentAnalyzer:
    """
    Κλάση που συνδυάζει διαφορετικές μεθόδους ανάλυσης συναισθημάτων
    όπως αναφέρονται στο proposal της πτυχιακής εργασίας.
    """
    
    def __init__(self):
        """Αρχικοποίηση των απαραίτητων αντικειμένων"""
        # VADER analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Μεταφραστής
        try:
            self.translator = Translator()
        except:
            self.translator = None
        
        # Λεξικό θετικών/αρνητικών λέξεων στα ελληνικά
        self.greek_sentiment_words = {
            'positive': [
                'καλό', 'άριστο', 'εξαιρετικό', 'υπέροχο', 'φανταστικό', 'ευτυχισμένος', 'χαρούμενος',
                'ικανοποιημένος', 'ευχαριστημένος', 'ενθουσιασμένος', 'τέλειο', 'αγαπημένο', 'ωραίο',
                'όμορφο', 'φιλικό', 'ευγενικό', 'εύκολο', 'βολικό', 'γρήγορο', 'αποτελεσματικό',
                'προτείνω', 'συνιστώ', 'επιτυχία', 'επιτυχημένο', 'ευχάριστο', 'διασκεδαστικό',
                'εντυπωσιακό', 'αξιόπιστο', 'αξιόλογο', 'δυνατό', 'φιλικό', 'εξυπηρετικό',
                'χρήσιμο', 'πρακτικό', 'λειτουργικό', 'άψογο', 'αξιοθαύμαστο', 'θετικό',
                'συμφέρον', 'οικονομικό', 'αξίζει', 'δίκαιο', 'δωρεάν', 'έκπτωση', 
                'προσφορά', 'υποστήριξη', 'βοήθεια', 'φροντίδα'
            ],
            'negative': [
                'κακό', 'χάλια', 'απαίσιο', 'απογοητευτικό', 'άσχημο', 'λυπημένος', 'θυμωμένος',
                'αγχωμένος', 'αρνητικό', 'δύσκολο', 'προβληματικό', 'δυσαρεστημένος', 'απογοήτευση',
                'αποτυχία', 'αποτυχημένο', 'αργό', 'ακριβό', 'ενοχλητικό', 'κουραστικό', 'βαρετό',
                'μετανιώνω', 'λάθος', 'πρόβλημα', 'δυσκολία', 'ελαττωματικό', 'χειρότερο',
                # Επέκταση αρνητικών λέξεων
                'μπούλινγκ', 'bullying', 'κόμπλεξ', 'απωθημένα', 'επιφανειακοί', 'επιφανειακό',
                'ψεύτικο', 'ψέμα', 'ψεύτης', 'απάτη', 'εξαπάτηση', 'κοροϊδία', 'κοροϊδεύω',
                'εκμετάλλευση', 'εκμεταλλεύονται', 'άσχημη', 'άσχημες', 'άσχημος', 'απαράδεκτο',
                'απαράδεκτη', 'απαράδεκτος', 'άθλιο', 'άθλια', 'άθλιος', 'αηδία', 'αηδιαστικό',
                'αγενής', 'αγένεια', 'αγενείς', 'κακομεταχείριση', 'κακοποίηση', 'επίθεση',
                'εχθρικός', 'εχθρικό', 'εχθρικότητα', 'εκφοβισμός', 'τραγικό', 'τραγική',
                'άχρηστο', 'άχρηστος', 'άχρηστη', 'άχρηστοι', 'σκουπίδι', 'σκουπίδια',
                'προδοσία', 'προδότης', 'προδοτικό', 'προδοτική', 'προδοτικός', 'προδίδω',
                'αποτυχημένος', 'αποτυχημένη', 'αποτυχημένο', 'αποτυχία', 'αποτυχίες',
                'ανίκανος', 'ανίκανη', 'ανίκανο', 'ανικανότητα', 'χειριστικός', 'χειρισμός',
                'μακριά', 'φύγε', 'φεύγω', 'παραιτούμαι', 'παραίτηση', 'εγκαταλείπω',
                'αηδιάζω', 'αηδιασμένος', 'αηδιασμένη', 'αδικία', 'άδικο', 'άδικος', 'άδικη'
            ]
        }
        
        # Λεξικό συναισθημάτων στα ελληνικά
        self.greek_emotion_words = {
            'fear': [
                'φόβος', 'φοβάμαι', 'τρομακτικό', 'τρομακτικά', 'φοβερό', 'τρομαγμένος', 
                'ανησυχώ', 'ανησυχία', 'άγχος', 'αγχωμένος', 'τρόμος', 'πανικός',
                'φοβισμένος', 'φοβισμένη', 'τρομοκρατημένος', 'τρομοκρατία', 'φρίκη',
                'ανασφάλεια', 'ανασφαλής', 'κίνδυνος', 'επικίνδυνος', 'επικίνδυνη',
                'απειλή', 'απειλητικό', 'απειλητική', 'απειλητικός'
            ],
            'anger': [
                'θυμός', 'θυμωμένος', 'εκνευρισμένος', 'οργή', 'οργισμένος', 'εξοργιστικό',
                'εκνευρίζομαι', 'τσαντίζομαι', 'τσαντισμένος', 'αγανάκτηση',
                'εξαγριωμένος', 'εξαγριώνομαι', 'μανία', 'έξαλλος', 'θύμωσα', 'θύμωσε',
                'νεύρα', 'νευριασμένος', 'νευριασμένη', 'νευρικός', 'νευρική',
                'εκρήγνυμαι', 'έκρηξη', 'βρίζω', 'βρισιές', 'κατάρες', 'μίσος', 'μισώ'
            ],
            'anticipation': [
                'προσδοκία', 'αναμονή', 'προσμένω', 'ελπίζω', 'ελπίδα', 'αδημονία',
                'αδημον', 'περιμένω', 'προσβλέπω', 'ανυπομονώ', 'ανυπομονησία',
                'αναμένω', 'προσμονή', 'προσδοκώ', 'αναμονή', 'περίμενε', 'περιμένει',
                'επιθυμία', 'επιθυμώ', 'επιθυμητό', 'επιθυμητή', 'επιθυμητός',
                'λαχτάρα', 'λαχταρώ', 'πόθος', 'ποθώ'
            ],
            'trust': [
                'εμπιστοσύνη', 'εμπιστεύομαι', 'αξιόπιστος', 'αξιόπιστο', 'αξιοπιστία',
                'πιστεύω', 'πίστη', 'εμπιστεύομαι', 'αξιόπιστη', 'εμπιστευτικός',
                'πιστός', 'πιστή', 'πιστό', 'αφοσίωση', 'αφοσιωμένος', 'αφοσιωμένη',
                'υποστήριξη', 'υποστηρίζω', 'υποστηρικτής', 'υποστηρικτική', 'υποστηρικτικός',
                'αξιοπιστία', 'εγγύηση', 'εγγυημένο', 'εγγυημένη', 'εγγυημένος'
            ],
            'surprise': [
                'έκπληξη', 'εκπλήσσομαι', 'έκπληκτος', 'αναπάντεχο', 'απροσδόκητο',
                'εντυπωσιακό', 'σοκαριστικό', 'σοκ', 'έκπληκτη', 'αναπάντεχα',
                'σοκαρισμένος', 'σοκαρισμένη', 'έκπληκτος', 'έκπληκτη', 'έκπληκτο',
                'αιφνιδιασμός', 'αιφνιδιασμένος', 'αιφνιδιασμένη', 'αιφνιδιαστικό',
                'αναπάντεχος', 'αναπάντεχη', 'αναπάντεχα', 'απρόσμενο', 'απρόσμενος', 'απρόσμενη'
            ],
            'sadness': [
                'λύπη', 'λυπημένος', 'στεναχώρια', 'στεναχωρημένος', 'θλίψη',
                'θλιμμένος', 'απογοήτευση', 'απογοητευμένος', 'μελαγχολία', 'θλιβερό',
                'πένθος', 'πενθώ', 'κλαίω', 'δάκρυα', 'δακρύζω', 'δακρυσμένος', 'δακρυσμένη',
                'πόνος', 'πονάω', 'πληγώθηκα', 'πληγωμένος', 'πληγωμένη', 'πληγωμένο',
                'χαμένος', 'χαμένη', 'χαμένο', 'απώλεια', 'απελπισία', 'απελπισμένος', 'απελπισμένη'
            ],
            'disgust': [
                'αηδία', 'αποστροφή', 'σιχαίνομαι', 'αποκρουστικό', 'απωθητικό',
                'αηδιαστικό', 'αποκρουστικός', 'απεχθής', 'απαίσιο', 'δυσάρεστο',
                'αποστρέφομαι', 'σιχαμερό', 'σιχαμένο', 'σιχαμένος', 'σιχαμένη',
                'αηδιασμένος', 'αηδιασμένη', 'αηδιαστικός', 'αηδιαστική', 'αηδιαστικά',
                'βδελυρός', 'βδελυρή', 'βδελυρό', 'αποτροπιασμός', 'αποτρόπαιος', 'αποτρόπαιο', 'αποτρόπαια'
            ],
            'joy': [
                'χαρά', 'χαρούμενος', 'ευτυχισμένος', 'ευτυχία', 'χαίρομαι', 
                'απολαμβάνω', 'ευχαρίστηση', 'ενθουσιασμός', 'ενθουσιασμένος', 'διασκέδαση',
                'ευτυχής', 'ευχάριστος', 'ευχάριστη', 'ευχάριστο', 'γέλιο', 'γελάω',
                'διασκεδάζω', 'κέφι', 'κεφάτος', 'κεφάτη', 'κεφάτο', 'αγαλλίαση',
                'απόλαυση', 'απολαυστικός', 'απολαυστική', 'απολαυστικό', 'ευφορία',
                'ευφορικός', 'ευφορική', 'ευφορικό', 'έκσταση', 'εκστασιασμένος', 'εκστασιασμένη'
            ]
        }
                
        # LSA components
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.svd = TruncatedSVD(n_components=10, random_state=42)
        self.lsa_model_trained = False
        
    def analyze_text(self, text):
        """
        Αναλύει ένα κείμενο χρησιμοποιώντας όλες τις διαθέσιμες μεθόδους.
        
        Args:
            text (str): Το κείμενο προς ανάλυση
            
        Returns:
            dict: Αποτελέσματα ανάλυσης από όλες τις μεθόδους
        """
        results = {}
        
        # Δοκιμή μετάφρασης του κειμένου στα αγγλικά για καλύτερα αποτελέσματα
        translated_text = self._translate_to_english(text)
        
        # 1. VADER ανάλυση - Θετική/Αρνητική/Ουδέτερη κατηγοριοποίηση
        vader_scores = self.get_vader_scores(translated_text if translated_text else text)
        
        # 2. Χρησιμοποιούμε επίσης το TextBlob για ένα δεύτερο σκορ συναισθήματος
        textblob_scores = self._get_textblob_scores(translated_text if translated_text else text)
        
        # 3. Συνδυασμός αποτελεσμάτων από VADER και TextBlob με το δικό μας λεξικό
        vader_scores = self._enhance_with_greek_lexicon(vader_scores, text)
        
        results['vader'] = vader_scores
        
        # 4. NRCLex ανάλυση - Εξαγωγή συγκεκριμένων συναισθημάτων
        # Χρησιμοποιούμε μεταφρασμένο κείμενο για καλύτερα αποτελέσματα
        emotion_scores = self.get_emotion_scores(translated_text if translated_text else text)
        # Ενισχύουμε με ελληνικό λεξικό συναισθημάτων
        emotion_scores = self._enhance_with_greek_emotions(emotion_scores, text)
        results['emotions'] = emotion_scores
        
        # 5. Προσθήκη σημασιολογικής ανάλυσης (χωρίς εκπαίδευση για ένα μεμονωμένο κείμενο)
        results['lsa'] = {
            "available": False,
            "message": "Η λανθάνουσα σημασιολογική ανάλυση (LSA) είναι διαθέσιμη μόνο για ανάλυση πολλαπλών κειμένων."
        }
        
        return results
    
    def _translate_to_english(self, text):
        """
        Μεταφράζει το κείμενο από ελληνικά σε αγγλικά για καλύτερη ανάλυση.
        
        Args:
            text (str): Το ελληνικό κείμενο
            
        Returns:
            str: Το μεταφρασμένο κείμενο (ή το αρχικό αν αποτύχει η μετάφραση)
        """
        if not text:
            return text
            
        try:
            # Δοκιμή μετάφρασης με googletrans
            if self.translator:
                translation = self.translator.translate(text, src='el', dest='en')
                if translation and translation.text:
                    return translation.text
            
            # Αν αποτύχει, δοκιμή με TextBlob
            blob = TextBlob(text)
            translated = str(blob.translate(from_lang='el', to='en'))
            return translated
        except Exception as e:
            print(f"Σφάλμα κατά τη μετάφραση: {e}")
            # Αν αποτύχει η μετάφραση, επιστρέφουμε το αρχικό κείμενο
            return text
    
    def get_vader_scores(self, text):
        """
        Χρησιμοποιεί το VADER για την ανάλυση του συναισθήματος.
        
        Args:
            text (str): Το κείμενο προς ανάλυση
            
        Returns:
            dict: Τα scores από το VADER (neg, neu, pos, compound)
        """
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Προσθήκη κατηγοριοποίησης συναισθήματος με βάση το compound score
        if scores['compound'] >= 0.05:
            sentiment = "Θετικό"
        elif scores['compound'] <= -0.05:
            sentiment = "Αρνητικό"
        else:
            sentiment = "Ουδέτερο"
            
        scores['sentiment'] = sentiment
        
        # Μετατροπή σε ποσοστά για καλύτερη κατανόηση
        scores['neg_percent'] = round(scores['neg'] * 100, 1)
        scores['neu_percent'] = round(scores['neu'] * 100, 1)
        scores['pos_percent'] = round(scores['pos'] * 100, 1)
        scores['compound_scaled'] = round((scores['compound'] + 1) / 2 * 100, 1)  # Κλίμακα 0-100
        
        return scores
    
    def get_emotion_scores(self, text):
        """
        Χρησιμοποιεί την βιβλιοθήκη NRCLex για την εξαγωγή συγκεκριμένων συναισθημάτων.
        
        Args:
            text (str): Το κείμενο προς ανάλυση
            
        Returns:
            dict: Οι τιμές για κάθε συναίσθημα (χαρά, φόβος, θυμός, κλπ)
        """
        try:
            # Αντιμετώπιση κενού κειμένου
            if not text or text.strip() == "":
                return {}
                
            # Ανάλυση με το NRCLex
            emotion_analyzer = NRCLex(text)
            
            # Λήψη των συναισθηματικών τιμών
            raw_scores = emotion_analyzer.affect_frequencies
            
            # Επεξεργασία αποτελεσμάτων για την τελική έξοδο
            emotions = {}
            for emotion, score in raw_scores.items():
                # Μετατροπή στα ελληνικά (για τα βασικά συναισθήματα)
                emotion_el = self._translate_emotion(emotion)
                emotions[emotion_el] = round(score * 100, 1)  # Σε ποσοστό %
            
            # Εύρεση του επικρατέστερου συναισθήματος (εξαιρώντας positive/negative)
            specific_emotions = {e: s for e, s in emotions.items() 
                               if e not in ['Θετικό', 'Αρνητικό', 'Ουδέτερο']}
            
            if specific_emotions:
                dominant_emotion = max(specific_emotions.items(), key=lambda x: x[1])
                emotions['dominant_emotion'] = dominant_emotion[0]
                emotions['dominant_score'] = dominant_emotion[1]
            else:
                emotions['dominant_emotion'] = "Δεν εντοπίστηκε"
                emotions['dominant_score'] = 0
                
            return emotions
            
        except Exception as e:
            print(f"Σφάλμα κατά την ανάλυση συναισθημάτων με NRCLex: {e}")
            return {
                "error": str(e),
                "dominant_emotion": "Σφάλμα",
                "dominant_score": 0
            }
    
    def _translate_emotion(self, emotion):
        """
        Μετάφραση των συναισθημάτων από αγγλικά σε ελληνικά.
        
        Args:
            emotion (str): Το συναίσθημα στα αγγλικά
            
        Returns:
            str: Το συναίσθημα στα ελληνικά
        """
        emotion_map = {
            'fear': 'Φόβος',
            'anger': 'Θυμός',
            'anticipation': 'Προσδοκία',
            'trust': 'Εμπιστοσύνη',
            'surprise': 'Έκπληξη',
            'sadness': 'Λύπη',
            'disgust': 'Αποστροφή',
            'joy': 'Χαρά',
            'positive': 'Θετικό',
            'negative': 'Αρνητικό',
            'neutral': 'Ουδέτερο',
        }
        return emotion_map.get(emotion, emotion)
    
    def analyze_batch(self, texts):
        """
        Ανάλυση πολλαπλών κειμένων συνδυάζοντας όλες τις μεθόδους.
        
        Args:
            texts (list): Λίστα με κείμενα προς ανάλυση
            
        Returns:
            dict: Αποτελέσματα από όλες τις μεθόδους και στατιστικά
        """
        results = {
            'individual_results': [],
            'statistics': {},
            'lsa_topics': []
        }
        
        # Συλλογή αποτελεσμάτων για κάθε κείμενο
        vader_sentiments = {'Θετικό': 0, 'Αρνητικό': 0, 'Ουδέτερο': 0}
        emotion_counts = {}
        
        for i, text in enumerate(texts):
            # Ανάλυση με όλες τις μεθόδους (εκτός LSA)
            text_analysis = {}
            
            # VADER
            vader_scores = self.get_vader_scores(text)
            text_analysis['vader'] = vader_scores
            
            # Ενημέρωση στατιστικών VADER
            sentiment = vader_scores['sentiment']
            vader_sentiments[sentiment] += 1
            
            # NRCLex
            emotion_scores = self.get_emotion_scores(text)
            text_analysis['emotions'] = emotion_scores
            
            # Ενημέρωση στατιστικών συναισθημάτων
            if 'dominant_emotion' in emotion_scores and emotion_scores['dominant_emotion'] != 'Δεν εντοπίστηκε':
                dominant = emotion_scores['dominant_emotion']
                if dominant in emotion_counts:
                    emotion_counts[dominant] += 1
                else:
                    emotion_counts[dominant] = 1
            
            # Προσθήκη του κειμένου στα αποτελέσματα
            text_analysis['text'] = text[:100] + "..." if len(text) > 100 else text
            text_analysis['id'] = i + 1
            
            results['individual_results'].append(text_analysis)
        
        # Στατιστικά για τα αποτελέσματα του VADER
        results['statistics']['vader_sentiments'] = vader_sentiments
        
        # Στατιστικά για τα συναισθήματα
        results['statistics']['emotion_counts'] = emotion_counts
        
        # Λανθάνουσα Σημασιολογική Ανάλυση (LSA)
        if len(texts) > 1:
            lsa_results = self._perform_lsa(texts)
            results['lsa_topics'] = lsa_results
        
        return results
    
    def _perform_lsa(self, texts):
        """
        Εκτελεί Λανθάνουσα Σημασιολογική Ανάλυση στα δοσμένα κείμενα.
        
        Args:
            texts (list): Λίστα με κείμενα
            
        Returns:
            list: Λίστα με τα κύρια θέματα και τις λέξεις που τα αποτελούν
        """
        try:
            # Μετατροπή κειμένων σε TF-IDF αναπαράσταση
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Εφαρμογή SVD για εύρεση των λανθανουσών διαστάσεων
            lsa_matrix = self.svd.fit_transform(tfidf_matrix)
            
            # Εξαγωγή των κυριότερων λέξεων για κάθε θέμα
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Προετοιμασία αποτελεσμάτων
            lsa_topics = []
            for topic_idx, topic in enumerate(self.svd.components_):
                # Λήψη των 10 κορυφαίων λέξεων για το συγκεκριμένο θέμα
                top_words_idx = topic.argsort()[:-11:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                
                # Προσθήκη θέματος με τις λέξεις του και τα βάρη τους
                topic_info = {
                    "topic_id": topic_idx + 1,
                    "top_words": top_words,
                    "weights": [round(topic[i], 3) for i in top_words_idx]
                }
                lsa_topics.append(topic_info)
            
            self.lsa_model_trained = True
            return lsa_topics
            
        except Exception as e:
            print(f"Σφάλμα κατά την εκτέλεση LSA: {e}")
            return [{"error": str(e)}]

    def _get_textblob_scores(self, text):
        """
        Χρησιμοποιεί το TextBlob για ανάλυση συναισθήματος.
        
        Args:
            text (str): Το κείμενο προς ανάλυση
            
        Returns:
            dict: Τα scores από το TextBlob (polarity, subjectivity)
        """
        try:
            blob = TextBlob(text)
            # Το TextBlob επιστρέφει polarity από -1 έως 1
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Μετατροπή σε κατηγορίες όπως το VADER
            if polarity >= 0.05:
                sentiment = "Θετικό"
            elif polarity <= -0.05:
                sentiment = "Αρνητικό"
            else:
                sentiment = "Ουδέτερο"
                
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment,
                'polarity_percent': round((polarity + 1) / 2 * 100, 1)  # Κλίμακα 0-100
            }
        except Exception as e:
            print(f"Σφάλμα κατά την ανάλυση με TextBlob: {e}")
            return {
                'polarity': 0,
                'subjectivity': 0,
                'sentiment': "Ουδέτερο",
                'polarity_percent': 50
            }
            
    def _enhance_with_greek_lexicon(self, vader_scores, text):
        """
        Ενισχύει τα αποτελέσματα του VADER με λέξεις-κλειδιά στα ελληνικά.
        
        Args:
            vader_scores (dict): Τα αρχικά scores από το VADER
            text (str): Το αρχικό ελληνικό κείμενο
            
        Returns:
            dict: Τα τροποποιημένα scores
        """
        if not text:
            return vader_scores
            
        text_lower = text.lower()
        pos_count = 0
        neg_count = 0
        
        # Αναζήτηση για θετικές λέξεις
        for word in self.greek_sentiment_words['positive']:
            if re.search(r'\b' + word + r'\b', text_lower):
                pos_count += 1
                
        # Αναζήτηση για αρνητικές λέξεις
        for word in self.greek_sentiment_words['negative']:
            if re.search(r'\b' + word + r'\b', text_lower):
                neg_count += 2  # Διπλό βάρος στις αρνητικές λέξεις
                
        # Αν βρήκαμε σημαντικές ενδείξεις στο ελληνικό κείμενο, τροποποιούμε τα scores
        total_words = max(len(text_lower.split()), 1)
        
        # Υπολογισμός ποσοστών
        pos_ratio = min(pos_count / total_words * 1.5, 1.0)  # Αύξηση κατά 50% με μέγιστο το 1.0
        neg_ratio = min(neg_count / total_words * 2.0, 1.0)  # Αύξηση κατά 100% με μέγιστο το 1.0
        
        # Έλεγχος για συγκεκριμένες σαρκαστικές φράσεις που υποδηλώνουν αρνητικότητα
        sarcastic_patterns = [
            r'το παίζ\w+ .{1,30} (άνθρωπο|καλ)',
            r'δήθεν .{1,20}',
            r'τάχα .{1,20}',
            r'ότι να( |\')?ναι',
            r'επιφανεια\w+'
        ]
        
        for pattern in sarcastic_patterns:
            if re.search(pattern, text_lower):
                neg_ratio += 0.2  # Προσθήκη επιπλέον αρνητικής βαρύτητας
                neg_ratio = min(neg_ratio, 1.0)  # Διασφάλιση ότι δεν ξεπερνά το 1.0
        
        # Τροποποίηση των scores με βάση τις ελληνικές λέξεις 
        # και έμφαση στο αρνητικό συναίσθημα αν υπάρχει
        if neg_count > 0 or pos_count > 0:
            # Αν υπάρχουν περισσότερες αρνητικές λέξεις ή σαρκασμός
            if neg_ratio > pos_ratio * 0.7:  # Χαμηλότερο όριο για αρνητικά
                vader_scores['neg'] = max(vader_scores['neg'], neg_ratio)
                vader_scores['compound'] = -neg_ratio  # Προσδίδουμε περισσότερο βάρος
                vader_scores['sentiment'] = "Αρνητικό"
            # Αν υπάρχουν περισσότερες θετικές λέξεις και όχι πολλές αρνητικές
            elif pos_ratio > neg_ratio * 1.5:  # Υψηλότερο όριο για θετικά
                vader_scores['pos'] = max(vader_scores['pos'], pos_ratio)
                vader_scores['compound'] = pos_ratio
                vader_scores['sentiment'] = "Θετικό"
            # Αλλιώς θεωρούμε ότι είναι ουδέτερο ή μικτό συναίσθημα
            else:
                vader_scores['neu'] = max(vader_scores['neu'], 0.6)
                vader_scores['compound'] = (pos_ratio - neg_ratio) * 0.7
                if vader_scores['compound'] > 0.05:
                    vader_scores['sentiment'] = "Θετικό"
                elif vader_scores['compound'] < -0.05:
                    vader_scores['sentiment'] = "Αρνητικό"
                else:
                    vader_scores['sentiment'] = "Ουδέτερο"
            
            # Ενημέρωση των ποσοστών
            vader_scores['neg_percent'] = round(vader_scores['neg'] * 100, 1)
            vader_scores['neu_percent'] = round(vader_scores['neu'] * 100, 1)
            vader_scores['pos_percent'] = round(vader_scores['pos'] * 100, 1)
            vader_scores['compound_scaled'] = round((vader_scores['compound'] + 1) / 2 * 100, 1)
        
        return vader_scores
    
    def _enhance_with_greek_emotions(self, emotion_scores, text):
        """
        Ενισχύει τα αποτελέσματα συναισθημάτων με λέξεις-κλειδιά στα ελληνικά.
        
        Args:
            emotion_scores (dict): Τα αρχικά scores συναισθημάτων
            text (str): Το αρχικό ελληνικό κείμενο
            
        Returns:
            dict: Τα τροποποιημένα scores συναισθημάτων
        """
        if not text or not isinstance(emotion_scores, dict):
            return emotion_scores
            
        text_lower = text.lower()
        emotion_counts = {}
        
        # Αναζήτηση για λέξεις συναισθημάτων
        for emotion, words in self.greek_emotion_words.items():
            count = 0
            for word in words:
                if re.search(r'\b' + word + r'\b', text_lower):
                    count += 1
            
            if count > 0:
                emotion_counts[emotion] = count
                
        # Αν βρήκαμε συναισθήματα, τροποποιούμε τα scores
        if emotion_counts:
            total_words = len(text_lower.split())
            dominant_emotion = None
            dominant_score = 0
            
            # Υπολογισμός ποσοστών και εύρεση του επικρατέστερου συναισθήματος
            for emotion, count in emotion_counts.items():
                emotion_el = self._translate_emotion(emotion)
                # Αύξηση του πολλαπλασιαστή για να δώσουμε μεγαλύτερη έμφαση
                # στα συναισθήματα που εντοπίστηκαν στο ελληνικό κείμενο
                score = min(count / total_words * 100 * 7, 100)  # Αύξηση πολλαπλασιαστή από 5 σε 7
                
                # Ειδική μεταχείριση για συγκεκριμένα συναισθήματα που συχνά εμφανίζονται στις κριτικές
                if emotion in ['anger', 'disgust', 'fear']:
                    score *= 1.3  # Αύξηση βαρύτητας για αρνητικά συναισθήματα
                
                # Έλεγχος για πολλαπλή εμφάνιση της ίδιας λέξης (ένταση συναισθήματος)
                intensity_bonus = 0
                for word in self.greek_emotion_words[emotion]:
                    matches = re.findall(r'\b' + word + r'\b', text_lower)
                    if len(matches) > 1:
                        intensity_bonus += (len(matches) - 1) * 5  # 5% επιπλέον για κάθε επανάληψη
                
                score = min(score + intensity_bonus, 100)  # Προσθήκη του μπόνους με μέγιστο το 100
                
                # Προσθήκη ή ενημέρωση του score
                if emotion_el in emotion_scores:
                    emotion_scores[emotion_el] = max(emotion_scores[emotion_el], score)
                else:
                    emotion_scores[emotion_el] = score
                
                # Έλεγχος αν είναι το επικρατέστερο
                if emotion_scores[emotion_el] > dominant_score:
                    dominant_emotion = emotion_el
                    dominant_score = emotion_scores[emotion_el]
            
            # Ενημέρωση του επικρατέστερου συναισθήματος
            if dominant_emotion and dominant_score > 0:
                emotion_scores['dominant_emotion'] = dominant_emotion
                emotion_scores['dominant_score'] = round(dominant_score, 1)
        
        # Επιπρόσθετοι έλεγχοι για ειδικές περιπτώσεις κειμένων
        # Αρνητικά συναισθήματα σε σαρκαστικές εκφράσεις
        sarcasm_patterns = [
            r'το παίζ\w+ .{1,30}',
            r'δήθεν',
            r'τάχα',
            r'μακριά από',
            r'επιφανεια\w+'
        ]
        
        for pattern in sarcasm_patterns:
            if re.search(pattern, text_lower):
                # Αύξηση συναισθημάτων θυμού/αποστροφής σε περίπτωση σαρκασμού
                emotion_scores['Θυμός'] = emotion_scores.get('Θυμός', 0) + 15
                emotion_scores['Αποστροφή'] = emotion_scores.get('Αποστροφή', 0) + 20
                
                # Επανέλεγχος για το επικρατέστερο συναίσθημα
                for emotion_name, score in emotion_scores.items():
                    if emotion_name not in ['dominant_emotion', 'dominant_score'] and score > dominant_score:
                        dominant_emotion = emotion_name
                        dominant_score = score
                
                # Ενημέρωση αν άλλαξε το επικρατέστερο συναίσθημα
                if dominant_emotion != emotion_scores.get('dominant_emotion'):
                    emotion_scores['dominant_emotion'] = dominant_emotion
                    emotion_scores['dominant_score'] = round(dominant_score, 1)
        
        return emotion_scores

# Για δοκιμές αν το script εκτελεστεί απευθείας
if __name__ == "__main__":
    analyzer = AdvancedSentimentAnalyzer()
    
    # Δοκιμή με ένα κείμενο
    test_text = "Είμαι πολύ χαρούμενος με την εξυπηρέτηση. Το προϊόν ήταν εξαιρετικό!"
    results = analyzer.analyze_text(test_text)
    print(f"Ανάλυση κειμένου: '{test_text}'")
    print("VADER:", results['vader'])
    print("Συναισθήματα:", results['emotions'])
    
    # Δοκιμή με πολλαπλά κείμενα
    test_texts = [
        "Είμαι πολύ χαρούμενος με την εξυπηρέτηση. Το προϊόν ήταν εξαιρετικό!",
        "Δεν είμαι καθόλου ευχαριστημένος με το προϊόν που αγόρασα.",
        "Το προϊόν ήταν εντάξει, αλλά τίποτα το ιδιαίτερο.",
        "Φοβάμαι ότι δεν θα ολοκληρωθεί η παραγγελία στην ώρα της.",
        "Εμπιστεύομαι απόλυτα την εταιρεία και τα προϊόντα της."
    ]
    batch_results = analyzer.analyze_batch(test_texts)
    print("\nΣτατιστικά Ανάλυσης Πολλαπλών Κειμένων:")
    print("VADER Κατανομή:", batch_results['statistics']['vader_sentiments'])
    print("Κατανομή Συναισθημάτων:", batch_results['statistics']['emotion_counts'])
    print("\nLSA Θέματα:", batch_results['lsa_topics']) 