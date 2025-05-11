#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Εφαρμογή ιστού για την ανάλυση συναισθημάτων σε κείμενα πελατών.
Υποστηρίζει την ανάλυση μεμονωμένων κειμένων και αρχείων CSV.
Χρησιμοποιεί το fine-tuned BERT μοντέλο **3 κλάσεων (Αρνητικό/Θετικό/Ουδέτερο)**.
Με υποστήριξη API για το React frontend.
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify, Response, send_from_directory
from flask_cors import CORS  # Νέα εισαγωγή για υποστήριξη CORS
import plotly
import plotly.express as px
import plotly.graph_objects as go
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Προσθήκη του φακέλου του έργου στο path για την εισαγωγή των modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Εισαγωγή του module για προηγμένη ανάλυση συναισθημάτων
from utils.advanced_sentiment_analysis import AdvancedSentimentAnalyzer

app = Flask(__name__)
CORS(app)  # Ενεργοποίηση CORS για όλα τα endpoints

# Δημιουργία των απαραίτητων φακέλων αν δεν υπάρχουν
os.makedirs(os.path.join(app.root_path, 'templates'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'css'), exist_ok=True)

# --- Φόρτωση του Fine-tuned Μοντέλου 3 Κλάσεων ---
sentiment_pipeline = None
model_loaded_successfully = False
# Αντιστοίχιση ετικετών από το μοντέλο σε κείμενο και κωδικό
LABEL_MAP = {
    'LABEL_0': {'name': 'Αρνητικό', 'code': 0},
    'LABEL_1': {'name': 'Θετικό', 'code': 1},
    'LABEL_2': {'name': 'Ουδέτερο', 'code': 2}
}
# Χρώματα για τα γραφήματα
COLOR_MAP = {
    'Αρνητικό': '#EF5350', # Κόκκινο
    'Θετικό': '#66BB6A',   # Πράσινο
    'Ουδέτερο': '#FFAB40'  # Πορτοκαλί 
}

# Αρχικοποίηση advanced sentiment analyzer
advanced_analyzer = None
advanced_analysis_enabled = True

def load_model():
    """Φορτώνει το μοντέλο BERT και ρυθμίζει το pipeline για ανάλυση συναισθημάτων."""
    global model_loaded_successfully, sentiment_pipeline, advanced_analyzer
    
    try:
        print("Φόρτωση μοντέλου BERT για ανάλυση συναισθημάτων...")
        
        # Χρήση του σωστού μοντέλου που ζήτησε ο χρήστης
        model_name = "tsiaggas/fine-tuned-for-sentiment-3class"
        
        print(f"Προσπάθεια φόρτωσης του μοντέλου: {model_name}")
        
        try:
            # Χρησιμοποιούμε το pipeline για sentiment analysis
            sentiment_pipeline = pipeline('text-classification', 
                                       model=model_name, 
                                       tokenizer=model_name)
            print("Pipeline δημιουργήθηκε επιτυχώς")
        except Exception as pipeline_error:
            print(f"Σφάλμα κατά τη δημιουργία του pipeline: {str(pipeline_error)}")
            # Εναλλακτικό μοντέλο - δοκιμή με το nlpaueb/bert-base-greek-uncased-v1
            model_name = "nlpaueb/bert-base-greek-uncased-v1"
            print(f"Δοκιμή με εναλλακτικό μοντέλο: {model_name}")
            sentiment_pipeline = pipeline('text-classification', 
                                       model=model_name)
        
        # Αρχικοποίηση του προηγμένου αναλυτή συναισθημάτων
        print("Αρχικοποίηση προηγμένου αναλυτή συναισθημάτων...")
        advanced_analyzer = AdvancedSentimentAnalyzer()
        
        model_loaded_successfully = True
        print("Το μοντέλο φορτώθηκε με επιτυχία!")
        
    except Exception as e:
        import traceback
        print(f"Σφάλμα κατά τη φόρτωση του μοντέλου: {str(e)}")
        print("Λεπτομέρειες σφάλματος:")
        traceback.print_exc()
        model_loaded_successfully = False
        sentiment_pipeline = None
        advanced_analyzer = None

# Φόρτωση του μοντέλου κατά την εκκίνηση
load_model()

# --- Διαδρομή για το React frontend ---
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path and os.path.exists(os.path.join(app.root_path, 'react-frontend', 'build', path)):
        return send_from_directory(os.path.join(app.root_path, 'react-frontend', 'build'), path)
    else:
        return send_from_directory(os.path.join(app.root_path, 'react-frontend', 'build'), 'index.html')

@app.route('/api/status')
def status():
    """API endpoint για την κατάσταση του server και του μοντέλου."""
    return jsonify({
        'status': 'online',
        'model_loaded': model_loaded_successfully,
        'version': '1.0.0'
    })

# --- Βοηθητική συνάρτηση για επεξεργασία αποτελεσμάτων pipeline (3 κλάσεις) ---
def process_pipeline_output(output):
    """
    Επεξεργάζεται την έξοδο του pipeline για ένα μεμονωμένο κείμενο.
    
    Υποστηρίζει δύο μορφές:
    1. Λίστα αντικειμένων: [{'label': 'LABEL_0', 'score': 0.1}, ...] 
    2. Μεμονωμένο αντικείμενο: {'label': 'LABEL_0', 'score': 0.1}
    """
    scores = {}
    highest_score = -1.0
    predicted_label_internal = None
    
    # Χειρισμός αποτελεσμάτων από το pipeline
    items_to_process = output
    
    # Διαπέρασε όλα τα αντικείμενα στο output
    for item in items_to_process:
        label_internal = item.get('label', '')
        score = item.get('score', 0.0)
        
        # Για τα μοντέλα του Hugging Face, τα labels μπορεί να είναι 
        # είτε 'LABEL_X' είτε πιο περιγραφικά όπως 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
        if label_internal in LABEL_MAP:
            # Αν είναι στον LABEL_MAP, χρησιμοποιούμε την αντιστοίχιση
            label_name = LABEL_MAP[label_internal]['name']
        else:
            # Προσπαθούμε να αντιστοιχίσουμε σε άλλες συνηθισμένες ετικέτες
            label_name = map_common_labels(label_internal)
            
        scores[label_name] = score
        
        if score > highest_score:
            highest_score = score
            predicted_label_internal = label_internal
    
    if predicted_label_internal is None or not scores:
        # Fallback σε περίπτωση που δεν βρέθηκε έγκυρη ετικέτα
        return {
            'sentiment_label': 'Ουδέτερο',  # Προεπιλογή σε ουδέτερο
            'sentiment_code': 2,
            'probability': 1.0,
            'scores': {'Ουδέτερο': 1.0, 'Θετικό': 0.0, 'Αρνητικό': 0.0}
        }
    
    # Λήψη τελικής ετικέτας και κωδικού
    if predicted_label_internal in LABEL_MAP:
        final_label_info = LABEL_MAP[predicted_label_internal]
        sentiment_label = final_label_info['name']
        sentiment_code = final_label_info['code']
    else:
        # Αν δεν είναι στο LABEL_MAP, χρησιμοποιούμε την αντιστοίχιση
        sentiment_label = map_common_labels(predicted_label_internal)
        # Αντιστοίχιση κωδικού με βάση την ετικέτα
        if sentiment_label == 'Θετικό':
            sentiment_code = 1
        elif sentiment_label == 'Αρνητικό':
            sentiment_code = 0
        else:
            sentiment_code = 2  # Ουδέτερο
    
    probability = highest_score
    
    # Εξασφάλιση ότι υπάρχουν σκορ για όλες τις αναμενόμενες κλάσεις (έστω 0)
    for expected_label in COLOR_MAP.keys():
        if expected_label not in scores:
            scores[expected_label] = 0.0
            
    return {
        'sentiment_label': sentiment_label,
        'sentiment_code': sentiment_code,
        'probability': probability,
        'scores': scores # Επιστρέφουμε τα σκορ για Αρνητικό, Θετικό, Ουδέτερο
    }

def map_common_labels(label):
    """
    Αντιστοιχίζει συνηθισμένες ετικέτες στη δική μας ονοματολογία.
    """
    label = label.lower()
    if 'positive' in label or 'pos' in label:
        return 'Θετικό'
    elif 'negative' in label or 'neg' in label:
        return 'Αρνητικό'
    else:
        return 'Ουδέτερο'  # Προεπιλογή

# --- Βοηθητική συνάρτηση για δημιουργία γραφήματος πιθανοτήτων (3 κλάσεις) ---
def create_probability_chart(scores):
    """
    Δημιουργεί ένα γράφημα Plotly bar chart για τις πιθανότητες των 3 κλάσεων.
    scores: Dictionary π.χ. {'Αρνητικό': 0.1, 'Θετικό': 0.2, 'Ουδέτερο': 0.7}
    """
    labels = list(scores.keys())
    values = [round(v * 100, 2) for v in scores.values()] # Μετατροπή σε %
    colors = [COLOR_MAP.get(label, '#808080') for label in labels] # Χρήση γκρι αν δεν βρεθεί χρώμα

    fig = go.Figure([go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition='auto')])
    fig.update_layout(
        title='Πιθανότητες Συναισθήματος',
        yaxis_title='Πιθανότητα (%)',
        xaxis_title='Συναίσθημα',
        yaxis=dict(range=[0, 100]), # Άξονας Υ από 0 έως 100
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    return fig

# Δημιουργία γραφήματος για τα συγκεκριμένα συναισθήματα
def create_emotions_chart(emotions_data):
    """
    Δημιουργεί ένα γράφημα για συγκεκριμένα συναισθήματα.
    emotions_data: Dictionary π.χ. {'Χαρά': 45.2, 'Εμπιστοσύνη': 32.1, ...}
    """
    # Φιλτράρουμε μόνο τα πραγματικά συναισθήματα (όχι τα θετικό/αρνητικό/ουδέτερο)
    specific_emotions = {k: v for k, v in emotions_data.items() 
                        if k not in ['Θετικό', 'Αρνητικό', 'Ουδέτερο', 'dominant_emotion', 'dominant_score']}
    
    if not specific_emotions:
        return None
    
    # Χρώματα για τα συναισθήματα
    emotion_colors = {
        'Χαρά': '#FFEB3B',       # Κίτρινο
        'Εμπιστοσύνη': '#4CAF50',  # Πράσινο
        'Φόβος': '#9C27B0',      # Μοβ
        'Έκπληξη': '#FF9800',    # Πορτοκαλί
        'Λύπη': '#2196F3',       # Μπλε
        'Αποστροφή': '#795548',  # Καφέ
        'Θυμός': '#F44336',      # Κόκκινο
        'Προσδοκία': '#03A9F4',  # Ανοικτό μπλε
    }
    
    labels = list(specific_emotions.keys())
    values = list(specific_emotions.values())
    colors = [emotion_colors.get(label, '#9E9E9E') for label in labels]
    
    fig = go.Figure([go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition='auto')])
    fig.update_layout(
        title='Ανάλυση Συγκεκριμένων Συναισθημάτων',
        yaxis_title='Ένταση (%)',
        xaxis_title='Συναίσθημα',
        yaxis=dict(range=[0, 100]),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    return fig

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """API endpoint για ανάλυση μεμονωμένου κειμένου με το μοντέλο 3 κλάσεων."""
    if not model_loaded_successfully or sentiment_pipeline is None:
        return jsonify({
            'error': 'Το μοντέλο ανάλυσης συναισθημάτων δεν είναι διαθέσιμο.',
            'message': 'Παρακαλώ δοκιμάστε αργότερα ή ελέγξτε τη φόρτωση του μοντέλου.'
        }), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        use_advanced = data.get('use_advanced', True)  # Προεπιλογή: χρήση προηγμένης ανάλυσης
        
        if not text.strip():
            return jsonify({
                'error': 'Κενό κείμενο', 
                'message': 'Παρακαλώ εισάγετε κείμενο για ανάλυση.'
            }), 400
        
        # BERT Ανάλυση (βασική)
        output = sentiment_pipeline(text)
        
        # Έλεγχος δομής του output και επεξεργασία ανάλογα
        if isinstance(output, list):
            # Παλιά μορφή: [{'label': 'LABEL_0', 'score': 0.1}, ...]
            result = process_pipeline_output(output)
        else:
            # Νέα μορφή: {'label': 'LABEL_x', 'score': 0.x}
            result = process_pipeline_output([output])
        
        # Δημιουργία γραφήματος πιθανοτήτων BERT
        bert_fig = create_probability_chart(result['scores'])
        bert_chart_json = json.dumps(bert_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Προετοιμασία της απάντησης
        response = {
            'text': text,
            'bert': {
                'sentiment': result['sentiment_label'],
                'sentiment_code': result['sentiment_code'],
                'probability': round(result['probability'] * 100, 2),
                'scores': {k: round(v * 100, 2) for k, v in result['scores'].items()},
                'chart': bert_chart_json
            },
            'advanced_analysis_available': advanced_analysis_enabled
        }
        
        # Προσθήκη προηγμένης ανάλυσης αν ζητηθεί και είναι διαθέσιμη
        if use_advanced and advanced_analysis_enabled:
            advanced_results = advanced_analyzer.analyze_text(text)
            
            # VADER αποτελέσματα
            vader_results = advanced_results.get('vader', {})
            
            # Συγκεκριμένα συναισθήματα
            emotions_results = advanced_results.get('emotions', {})
            emotions_chart = None
            if emotions_results:
                emotions_chart = create_emotions_chart(emotions_results)
            
            response['advanced'] = {
                'vader': vader_results,
                'emotions': emotions_results
            }
            
            # Προσθήκη γραφήματος συναισθημάτων αν υπάρχει
            if emotions_chart:
                response['advanced']['emotions_chart'] = json.dumps(emotions_chart, cls=plotly.utils.PlotlyJSONEncoder)
            
        return jsonify(response)
    
    except Exception as e:
        print(f"Σφάλμα κατά την ανάλυση: {e}")
        return jsonify({
            'error': 'Σφάλμα επεξεργασίας', 
            'message': str(e)
        }), 500

@app.route('/api/analyze_batch', methods=['POST'])
def analyze_batch():
    """API endpoint για ανάλυση πολλαπλών κειμένων (CSV) με το μοντέλο 3 κλάσεων."""
    if not model_loaded_successfully or sentiment_pipeline is None:
        return jsonify({
            'error': 'Το μοντέλο ανάλυσης συναισθημάτων δεν είναι διαθέσιμο.',
            'message': 'Παρακαλώ δοκιμάστε αργότερα ή ελέγξτε τη φόρτωση του μοντέλου.'
        }), 500
    
    try:
        # Έλεγχος αν υπάρχει αρχείο
        if 'file' not in request.files:
            return jsonify({
                'error': 'Δεν βρέθηκε αρχείο', 
                'message': 'Παρακαλώ επιλέξτε ένα αρχείο CSV για ανάλυση.'
            }), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'Κανένα αρχείο δεν επιλέχθηκε', 
                'message': 'Παρακαλώ επιλέξτε ένα αρχείο CSV για ανάλυση.'
            }), 400
            
        # Έλεγχος τύπου αρχείου - αποδεκτά μόνο CSV
        if not file.filename.endswith('.csv'):
            return jsonify({
                'error': 'Μη αποδεκτός τύπος αρχείου', 
                'message': 'Παρακαλώ επιλέξτε ένα αρχείο CSV (.csv).'
            }), 400
        
        # Διαβάζουμε το CSV με το pandas
        text_column = request.form.get('text_column', None)
        df = pd.read_csv(file)
        
        # Έλεγχος αν υπάρχει η στήλη κειμένου
        if text_column and text_column not in df.columns:
            return jsonify({
                'error': 'Η στήλη κειμένου δεν βρέθηκε', 
                'message': f'Η στήλη "{text_column}" δεν υπάρχει στο αρχείο CSV.'
            }), 400
        
        # Αν δεν έχει οριστεί στήλη κειμένου, χρησιμοποιούμε την πρώτη στήλη
        if not text_column:
            text_column = df.columns[0]
            
        # Περιορισμός μεγέθους για επεξεργασία (προαιρετικό)
        max_rows = 100  # Μέγιστος αριθμός γραμμών για επεξεργασία
        if len(df) > max_rows:
            df = df.head(max_rows)
            
        # Επεξεργασία κάθε κειμένου στη στήλη
        sentiment_results = []
        sentiment_counts = {'Θετικό': 0, 'Αρνητικό': 0, 'Ουδέτερο': 0}
        
        for i, row in df.iterrows():
            try:
                text = str(row[text_column])
                if not text or text == 'nan':
                    continue
                    
                # Ανάλυση συναισθήματος
                output = sentiment_pipeline(text)
                
                # Έλεγχος δομής του output και επεξεργασία ανάλογα
                if isinstance(output, list):
                    # Παλιά μορφή: [{'label': 'LABEL_0', 'score': 0.1}, ...]
                    result = process_pipeline_output(output)
                else:
                    # Νέα μορφή: {'label': 'LABEL_x', 'score': 0.x}
                    result = process_pipeline_output([output])
                
                # Προσθήκη αποτελέσματος
                sentiment_label = result['sentiment_label']
                sentiment_counts[sentiment_label] += 1
                
                sentiment_results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': sentiment_label,
                    'probability': round(result['probability'] * 100, 2),
                    'row_number': i + 1  # +1 για αρίθμηση από 1
                })
            except Exception as e:
                print(f"Σφάλμα στην επεξεργασία της γραμμής {i}: {e}")
        
        # Δημιουργία γραφήματος συναισθημάτων
        chart_data = create_batch_sentiment_chart(sentiment_counts)
        
        # Επιστροφή αποτελεσμάτων σε JSON
        response = {
            'results': sentiment_results,
            'count': len(sentiment_results),
            'sentiment_counts': sentiment_counts,
            'chart': json.dumps(chart_data, cls=plotly.utils.PlotlyJSONEncoder)
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Σφάλμα κατά την ανάλυση του CSV: {e}")
        return jsonify({
            'error': 'Σφάλμα επεξεργασίας CSV', 
            'message': str(e)
        }), 500

def create_batch_sentiment_chart(sentiment_counts):
    """
    Δημιουργεί ένα γράφημα πίτας για την κατανομή συναισθημάτων.
    sentiment_counts: Dictionary {'Θετικό': 10, 'Αρνητικό': 5, 'Ουδέτερο': 3}
    """
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    colors = [COLOR_MAP.get(label, '#808080') for label in labels]
    
    fig = go.Figure(data=[go.Pie(labels=labels, 
                                values=values, 
                                marker_colors=colors,
                                textinfo='percent+label',
                                hole=.4)])
    fig.update_layout(
        title='Κατανομή Συναισθημάτων',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

if __name__ == '__main__':
    # Χρήση waitress για production deployment αντί για το Flask development server
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=5000)
    
    # Για ανάπτυξη (development)
    app.run(debug=True, host='0.0.0.0', port=5000) 