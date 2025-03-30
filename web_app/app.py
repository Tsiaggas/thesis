#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Εφαρμογή ιστού για την ανάλυση συναισθημάτων σε κείμενα πελατών.
Υποστηρίζει την ανάλυση μεμονωμένων κειμένων και αρχείων CSV.
Χρησιμοποιεί το fine-tuned BERT μοντέλο **3 κλάσεων (Αρνητικό/Θετικό/Ουδέτερο)**.
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify, Response
import plotly
import plotly.express as px
import plotly.graph_objects as go
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Προσθήκη του φακέλου του έργου στο path για την εισαγωγή των modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Αφαίρεση της εισαγωγής του παλιού μοντέλου
# from utils.bert_model import BERTSentimentAnalyzer

app = Flask(__name__)

# Δημιουργία των απαραίτητων φακέλων αν δεν υπάρχουν
os.makedirs(os.path.join(app.root_path, 'templates'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'css'), exist_ok=True)

# Διαδρομές για τα μοντέλα - επέλεξε μία από τις δύο μεθόδους
USE_HUGGINGFACE_HUB = True  # Άλλαξε σε False για να χρησιμοποιήσεις τοπικό μοντέλο
LOCAL_MODEL_PATH = os.path.join('..', 'models', 'fine_tuned_bert_3class')
HUGGINGFACE_MODEL_PATH = "tsiaggas/fine-tuned-for-sentiment-3class"  # Αντικατέστησε με το δικό σου username/repo

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

def load_model():
    try:
        if USE_HUGGINGFACE_HUB:
            # Φόρτωση από το Hugging Face Hub
            print(f"Φόρτωση μοντέλου από το Hugging Face Hub: {HUGGINGFACE_MODEL_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(HUGGINGFACE_MODEL_PATH)
        else:
            # Φόρτωση από τοπικό φάκελο
            print(f"Φόρτωση μοντέλου από τοπικό φάκελο: {LOCAL_MODEL_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
        
        # Ορισμός του μοντέλου σε λειτουργία αξιολόγησης
        model.eval()
        
        # Μεταφορά στη GPU αν είναι διαθέσιμη
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        return tokenizer, model, device
    except Exception as e:
        print(f"Σφάλμα κατά τη φόρτωση του μοντέλου: {e}")
        return None, None, None

try:
    print("Φόρτωση fine-tuned μοντέλου BERT 3 κλάσεων (Αρνητικό/Θετικό/Ουδέτερο)...")
    # Αλλαγή διαδρομής στο μοντέλο 3 κλάσεων
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'models', 'fine_tuned_bert_3class') 

    if os.path.exists(model_dir):
        tokenizer, model, device = load_model()

        # Έλεγχος αν το μοντέλο έχει όντως 3 ετικέτες
        if model.config.num_labels != 3:
             print(f"Προειδοποίηση: Το μοντέλο στο {model_dir} έχει {model.config.num_labels} ετικέτες, αλλά αναμένονται 3.")

        # Έλεγχος για GPU
        print(f"Χρήση συσκευής για pipeline: {'GPU' if device == 0 else 'CPU'}")

        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True # Επιστρέφει σκορ για όλες τις ετικέτες (LABEL_0, LABEL_1, LABEL_2)
        )
        model_loaded_successfully = True
        print("Το fine-tuned μοντέλο BERT 3 κλάσεων φορτώθηκε επιτυχώς ως pipeline!")
    else:
         print(f"Σφάλμα: Ο φάκελος του μοντέλου δεν βρέθηκε στη διαδρομή: {model_dir}")

except Exception as e:
    print(f"Σφάλμα κατά τη φόρτωση του fine-tuned μοντέλου 3 κλάσεων: {e}")
    sentiment_pipeline = None

@app.route('/')
def home():
    """Αρχική σελίδα της εφαρμογής."""
    return render_template('home.html', model_loaded=model_loaded_successfully)

# --- Βοηθητική συνάρτηση για επεξεργασία αποτελεσμάτων pipeline (3 κλάσεις) ---
def process_pipeline_output(output):
    """
    Επεξεργάζεται την έξοδο του pipeline 3 κλάσεων για ένα μεμονωμένο κείμενο.
    Η έξοδος είναι της μορφής: 
    [{'label': 'LABEL_0', 'score': 0.1}, {'label': 'LABEL_1', 'score': 0.2}, {'label': 'LABEL_2', 'score': 0.7}]
    """
    scores = {}
    highest_score = -1.0
    predicted_label_internal = None

    # Συλλογή σκορ και εύρεση της επικρατέστερης ετικέτας
    for item in output:
        label_internal = item['label']
        score = item['score']
        if label_internal in LABEL_MAP:
             label_name = LABEL_MAP[label_internal]['name']
             scores[label_name] = score
             if score > highest_score:
                 highest_score = score
                 predicted_label_internal = label_internal
        else:
            print(f"Άγνωστη ετικέτα από το pipeline: {label_internal}")
    
    if predicted_label_internal is None:
         # Fallback σε περίπτωση που δεν βρέθηκε έγκυρη ετικέτα
         return {
            'sentiment_label': 'Άγνωστο',
            'sentiment_code': -1,
            'probability': 0.0,
            'scores': scores
        }
        
    # Λήψη τελικής ετικέτας, κωδικού και πιθανότητας
    final_label_info = LABEL_MAP[predicted_label_internal]
    sentiment_label = final_label_info['name']
    sentiment_code = final_label_info['code']
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

@app.route('/analyze', methods=['POST'])
def analyze():
    """Ανάλυση μεμονωμένου κειμένου με το μοντέλο 3 κλάσεων."""
    if not model_loaded_successfully or sentiment_pipeline is None:
        return jsonify({
            'error': 'Το μοντέλο ανάλυσης συναισθημάτων δεν είναι διαθέσιμο.',
            'message': 'Παρακαλώ δοκιμάστε αργότερα ή ελέγξτε τη φόρτωση του μοντέλου.'
        }), 500
    
    try:
        text = request.form.get('text', '')
        if not text.strip():
            return jsonify({
                'error': 'Το κείμενο είναι κενό.',
                'message': 'Παρακαλώ εισάγετε ένα κείμενο για ανάλυση.'
            }), 400
        
        # Πρόβλεψη συναισθήματος με το pipeline
        pipeline_output = sentiment_pipeline(text)[0] 
        
        # Επεξεργασία της εξόδου του pipeline
        result = process_pipeline_output(pipeline_output)
                
        # Δημιουργία του γραφήματος Plotly
        prob_chart_fig = create_probability_chart(result['scores'])

        # Προετοιμασία της απόκρισης
        response_dict = {
            'text': text,
            'sentiment_label': result['sentiment_label'],
            'sentiment_code': result['sentiment_code'], 
            'probability': round(result['probability'] * 100, 2),
            'scores': {k: round(v * 100, 2) for k, v in result['scores'].items()},
            'probability_chart': prob_chart_fig 
        }
        
        # Χειροκίνητη μετατροπή σε JSON με Plotly encoder
        json_response = json.dumps(response_dict, cls=plotly.utils.PlotlyJSONEncoder)
        return Response(json_response, mimetype='application/json')
    
    except Exception as e:
        print(f"Σφάλμα στο /analyze: {e}") 
        # Επιστρέφουμε ακόμα JSON για τα σφάλματα με τον κλασικό τρόπο
        return jsonify({
            'error': 'Σφάλμα κατά την ανάλυση του κειμένου.',
            'message': str(e)
        }), 500

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    """Ανάλυση batch από αρχείο CSV με το μοντέλο 3 κλάσεων."""
    if not model_loaded_successfully or sentiment_pipeline is None:
        return jsonify({
            'error': 'Το μοντέλο ανάλυσης συναισθημάτων δεν είναι διαθέσιμο.',
            'message': 'Παρακαλώ δοκιμάστε αργότερα ή ελέγξτε τη φόρτωση του μοντέλου.'
        }), 500
    
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({
                'error': 'Δεν υποβλήθηκε αρχείο.',
                'message': 'Παρακαλώ επιλέξτε ένα αρχείο CSV για ανάλυση.'
            }), 400
        
        # Διάβασμα του αρχείου CSV
        try:
            df = pd.read_csv(file)
        except Exception as read_e:
             return jsonify({
                'error': 'Σφάλμα ανάγνωσης αρχείου CSV.',
                'message': f'Βεβαιωθείτε ότι το αρχείο είναι έγκυρο CSV. Σφάλμα: {read_e}'
            }), 400
            
        # Έλεγχος για την ύπαρξη της στήλης 'text'
        if 'text' not in df.columns:
            # Δοκιμή για εναλλακτικές στήλες (π.χ. 'review')
            found_col = None
            for col in ['review', 'comment', 'content', 'text_column']:
                if col in df.columns:
                    df.rename(columns={col: 'text'}, inplace=True)
                    found_col = col
                    print(f"Χρησιμοποιήθηκε η στήλη '{col}' ως στήλη κειμένου.")
                    break
            if not found_col:
                 return jsonify({
                    'error': 'Μη έγκυρο αρχείο CSV.',
                    'message': 'Το αρχείο CSV πρέπει να περιέχει μια στήλη με κείμενο (π.χ., "text", "review").'
                }), 400
        
        # Αφαίρεση κενών κειμένων και μετατροπή σε string
        df = df.dropna(subset=['text'])
        df['text'] = df['text'].astype(str)
        df = df[df['text'].str.strip() != '']
        texts_to_analyze = df['text'].tolist()
        total_texts_in_file = len(texts_to_analyze)
        
        if not texts_to_analyze:
            return jsonify({
                'error': 'Δεν βρέθηκαν έγκυρα κείμενα στο αρχείο.',
                'message': 'Το αρχείο CSV δεν περιέχει έγκυρα κείμενα για ανάλυση μετά τον καθαρισμό.'
            }), 400
        
        print(f"Ανάλυση {len(texts_to_analyze)} κειμένων από το αρχείο...")
        # Πρόβλεψη συναισθημάτων για όλα τα κείμενα με το pipeline
        # Χρήση batching του pipeline για ταχύτητα, αν είναι δυνατόν (εξαρτάται από πόρους)
        try:
             # Προσπάθεια για batching, μπορεί να χρειαστεί προσαρμογή batch_size
             results_pipeline = sentiment_pipeline(texts_to_analyze, batch_size=8, truncation=True) 
        except Exception as pipe_err:
             print(f"Σφάλμα κατά το batching στο pipeline: {pipe_err}. Δοκιμή χωρίς batching.")
             # Fallback χωρίς batching αν το batching αποτύχει
             results_pipeline = [sentiment_pipeline(text)[0] for text in texts_to_analyze]

        print("Η ανάλυση ολοκληρώθηκε. Επεξεργασία αποτελεσμάτων...")
        
        # Επεξεργασία αποτελεσμάτων και συλλογή στατιστικών για 3 κλάσεις
        analysis_results = []
        sentiment_counts = {'Αρνητικό': 0, 'Θετικό': 0, 'Ουδέτερο': 0}
        
        for i, output in enumerate(results_pipeline):
            original_text = texts_to_analyze[i]
            # Επεξεργασία εξόδου για το κάθε κείμενο (χρησιμοποιεί τη νέα process_pipeline_output)
            processed_result = process_pipeline_output(output)
            
            analysis_results.append({
                'text': original_text,
                'sentiment_label': processed_result['sentiment_label'],
                'probability': round(processed_result['probability'] * 100, 2)
            })
            
            # Μέτρηση κατανομής συναισθημάτων
            if processed_result['sentiment_label'] in sentiment_counts:
                sentiment_counts[processed_result['sentiment_label']] += 1
        
        analyzed_texts_count = len(analysis_results)
        
        # Δημιουργία γραφήματος κατανομής (χρησιμοποιεί τη νέα create_batch_sentiment_chart)
        sentiment_chart_fig = create_batch_sentiment_chart(sentiment_counts)

        # Προετοιμασία απόκρισης
        response_dict = {
            'total_texts_in_file': total_texts_in_file,
            'analyzed_texts': analyzed_texts_count,
            'sentiment_counts': sentiment_counts,
            'results': analysis_results,
            'sentiment_distribution_chart': sentiment_chart_fig
        }
        
        # Χειροκίνητη μετατροπή σε JSON με Plotly encoder
        json_response = json.dumps(response_dict, cls=plotly.utils.PlotlyJSONEncoder)
        return Response(json_response, mimetype='application/json')

    except Exception as e:
        print(f"Σφάλμα στο /analyze_batch: {e}")
        # Επιστρέφουμε ακόμα JSON για τα σφάλματα με τον κλασικό τρόπο
        return jsonify({
            'error': 'Σφάλμα κατά την ανάλυση του αρχείου.',
            'message': str(e)
        }), 500

@app.route('/about')
def about():
    """Σελίδα πληροφοριών για την εφαρμογή."""
    return render_template('about.html')

# --- Βοηθητική συνάρτηση για δημιουργία γραφήματος κατανομής για batch (3 κλάσεις) ---
def create_batch_sentiment_chart(sentiment_counts):
    """
    Δημιουργεί ένα γράφημα πίτας Plotly για την κατανομή των 3 συναισθημάτων σε ένα batch.
    sentiment_counts: Dictionary π.χ. {'Αρνητικό': 50, 'Θετικό': 100, 'Ουδέτερο': 20}
    """
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())
    colors = [COLOR_MAP.get(label, '#808080') for label in labels]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker_colors=colors, hole=.3)])
    fig.update_layout(
        title_text='Κατανομή Συναισθημάτων στο Αρχείο',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    return fig

if __name__ == '__main__':
    # Χρήση waitress για production deployment αντί για το Flask development server
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=5000)
    
    # Για ανάπτυξη (development)
    app.run(debug=True, host='0.0.0.0', port=5000) 