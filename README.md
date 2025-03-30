# Ανάλυση Συναισθήματος Κριτικών Skroutz με Deep Learning

Το παρόν repository περιέχει την υλοποίηση ενός συστήματος ανάλυσης συναισθήματος 3 κλάσεων (Αρνητικό, Θετικό, Ουδέτερο) για κριτικές προϊόντων από το Skroutz, χρησιμοποιώντας τεχνικές Deep Learning και μοντέλα BERT.

## Περιγραφή

Η εργασία αυτή στοχεύει στην ανάπτυξη ενός αποτελεσματικού συστήματος ανάλυσης συναισθήματος για κείμενα στα Ελληνικά, εστιάζοντας στις κριτικές προϊόντων από το Skroutz. Χρησιμοποιεί state-of-the-art τεχνικές Natural Language Processing (NLP) και συγκεκριμένα fine-tuning του προ-εκπαιδευμένου μοντέλου BERT για ελληνικά κείμενα.

## Πρόσβαση στο Μοντέλο

Το εκπαιδευμένο μοντέλο είναι διαθέσιμο στο Hugging Face Hub:
https://huggingface.co/tsiaggas/fine-tuned-for-sentiment-3class

Μπορεί να χρησιμοποιηθεί απευθείας με τον παρακάτω κώδικα Python:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Φόρτωση μοντέλου και tokenizer
model = AutoModelForSequenceClassification.from_pretrained("tsiaggas/fine-tuned-for-sentiment-3class")
tokenizer = AutoTokenizer.from_pretrained("tsiaggas/fine-tuned-for-sentiment-3class")
```

## Δομή Repository

```
├── data/                         # Δεδομένα και datasets
│   ├── processed/                # Επεξεργασμένα datasets
│   │   ├── augmented_skroutz_dataset.csv    # Αυξημένο dataset (θετικά/αρνητικά)
│   │   └── skroutz_3class_dataset.csv       # Τελικό dataset 3 κλάσεων
│   └── neutral_texts.csv         # Dataset ουδέτερων κειμένων
├── models/                       # Μοντέλα (τοπικά)
├── results/                      # Αποτελέσματα και μετρικές
│   └── advanced_metrics_description.txt  # Επεξήγηση των μετρικών
├── utils/                        # Βοηθητικά scripts
│   ├── advanced_metrics.py       # Script για προηγμένες οπτικοποιήσεις
│   ├── combine_datasets.py       # Script για συνδυασμό datasets
│   ├── plot_metrics.py           # Script για απλές οπτικοποιήσεις
│   └── upload_to_hf.py           # Script για ανέβασμα του μοντέλου στο Hugging Face
├── web_app/                      # Web εφαρμογή για επίδειξη του μοντέλου
├── README.md                     # Το αρχείο που διαβάζετε
└── requirements.txt              # Απαιτούμενες βιβλιοθήκες Python
```

## Εγκατάσταση και Εκτέλεση

### Προαπαιτούμενα

- Python 3.8+
- CUDA (προαιρετικά για GPU acceleration)

### Εγκατάσταση

1. Κλωνοποιήστε το repository:

```bash
git clone https://github.com/your-username/thesis.git
cd thesis
```

2. Δημιουργήστε και ενεργοποιήστε ένα εικονικό περιβάλλον Python:

```bash
# Δημιουργία περιβάλλοντος
python -m venv .venv

# Ενεργοποίηση στα Windows
.\.venv\Scripts\activate

# Ενεργοποίηση σε Linux/Mac
source .venv/bin/activate
```

3. Εγκαταστήστε τις απαιτούμενες βιβλιοθήκες:

```bash
pip install -r requirements.txt
```

### Εκτέλεση Web Εφαρμογής

Η web εφαρμογή μπορεί να χρησιμοποιήσει είτε το τοπικό μοντέλο (αν υπάρχει) είτε το μοντέλο από το Hugging Face Hub:

```bash
cd web_app
python app.py
```

Η ρύθμιση γίνεται στην παράμετρο `USE_HUGGINGFACE_HUB` στο αρχείο `web_app/app.py`.

### Εκτέλεση Οπτικοποιήσεων

Για τη δημιουργία προηγμένων οπτικοποιήσεων μετρικών:

```bash
python utils/advanced_metrics.py
```

## Βασικές Λειτουργίες

- **Fine-tuning μοντέλου BERT**: Προσαρμογή του BERT για την ταξινόμηση συναισθήματος σε 3 κλάσεις
- **Επεξεργασία και συνδυασμός datasets**: Συνδυασμός δεδομένων για τη δημιουργία ενός ισορροπημένου dataset 3 κλάσεων
- **Αξιολόγηση και οπτικοποίηση**: Υπολογισμός και οπτικοποίηση προηγμένων μετρικών αξιολόγησης
- **Web εφαρμογή επίδειξης**: Διαδραστική εφαρμογή για την επίδειξη της λειτουργίας του μοντέλου
- **Ανέβασμα στο Hugging Face Hub**: Δυνατότητα διαμοιρασμού του μοντέλου μέσω του Hugging Face Hub

## Αποτελέσματα

Το μοντέλο επιτυγχάνει υψηλή ακρίβεια στην ταξινόμηση συναισθήματος ελληνικών κριτικών προϊόντων:

- **Accuracy**: ~97.3%
- **F1 Score (Weighted)**: ~97.3%
- **Precision (Weighted)**: ~97.3%
- **Recall (Weighted)**: ~97.3%

Λεπτομερή αποτελέσματα και οπτικοποιήσεις παράγονται από το script `advanced_metrics.py`.

## Άδεια Χρήσης

[Προσθέστε την άδεια χρήσης που επιθυμείτε]

---

Αναπτύχθηκε ως μέρος πτυχιακής εργασίας για το [Όνομα Πανεπιστημίου/Τμήματος]. 