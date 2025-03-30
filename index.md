# Ευρετήριο Project: Ανάλυση Συναισθήματος σε Ελληνικά Κείμενα

Αυτό το έγγραφο αποτελεί τον οδηγό πλοήγησης για το project ανάλυσης συναισθήματος σε ελληνικά κείμενα με τη χρήση του fine-tuned BERT μοντέλου.

## Δομή του Project

```
/
├── README.md                  # Γενικές πληροφορίες για το project
├── requirements.txt           # Απαιτούμενα πακέτα Python 
├── index.md                   # Αυτό το αρχείο - ευρετήριο του project
├── analyze_dataset.py         # Script για ανάλυση δεδομένων
├── analyze_length.py          # Script για ανάλυση μήκους κειμένων
│
├── data/                      # Φάκελος δεδομένων
│   ├── Skroutz_dataset.xlsx   # Αρχικό dataset από Skroutz
│   └── processed/             # Επεξεργασμένα datasets
│
├── models/                    # Φάκελος μοντέλων
│   ├── fine_tuned_bert/       # Το fine-tuned BERT μοντέλο
│   ├── train_bert_finetune.py # Script εκπαίδευσης του BERT
│   └── train_model.py         # Γενικό script εκπαίδευσης μοντέλων
│
├── utils/                     # Βοηθητικές λειτουργίες
│   ├── evaluate_model.py      # Script αξιολόγησης μοντέλου
│   └── text_preprocessing.py  # Λειτουργίες προεπεξεργασίας κειμένου
│
├── results/                   # Αποτελέσματα αξιολόγησης
│
├── web_app/                   # Διαδικτυακή εφαρμογή
│   ├── app.py                 # Κύριο αρχείο Flask εφαρμογής
│   ├── templates/             # HTML templates
│   ├── static/                # CSS, JS, εικόνες
│   ├── uploads/               # Προσωρινός φάκελος για uploaded αρχεία
│   ├── utils/                 # Βοηθητικά scripts για την εφαρμογή
│   └── models/                # Αντίγραφα μοντέλων για την εφαρμογή
│
├── tests/                     # Unit tests
│
├── src/                       # Πηγαίος κώδικας (legacy)
│
├── notebooks/                 # Jupyter notebooks
│
└── .venv/                     # Virtual environment (δεν περιλαμβάνεται στο Git)
```

## Βασικές Λειτουργίες

### 1. Εκπαίδευση Μοντέλου

Το project χρησιμοποιεί το προ-εκπαιδευμένο μοντέλο `nlpaueb/bert-base-greek-uncased-v1` fine-tuned σε δεδομένα από το Skroutz. Το script εκπαίδευσης βρίσκεται στο:

```
models/train_bert_finetune.py
```

### 2. Αξιολόγηση Μοντέλου

Η αξιολόγηση του μοντέλου γίνεται με το script:

```
utils/evaluate_model.py
```

Αυτό το script παράγει αναφορές απόδοσης (accuracy, precision, recall, F1-score) και αποθηκεύει τα αποτελέσματα στον φάκελο `results/`.

### 3. Διαδικτυακή Εφαρμογή

Η διαδικτυακή εφαρμογή Flask επιτρέπει:
- Ανάλυση μεμονωμένων κειμένων
- Μαζική ανάλυση από αρχεία CSV

Το κύριο αρχείο της εφαρμογής είναι:

```
web_app/app.py
```

## Σημαντικές Ροές Εργασίας

### Πλήρης Ροή Εργασίας

1. **Προεπεξεργασία Δεδομένων**: Χρήση του `utils/text_preprocessing.py`
2. **Εκπαίδευση Μοντέλου**: Εκτέλεση του `models/train_bert_finetune.py`
3. **Αξιολόγηση Μοντέλου**: Εκτέλεση του `utils/evaluate_model.py`
4. **Εκτέλεση Εφαρμογής**: Εκτέλεση του `web_app/app.py`

### Χρήση Διαδικτυακής Εφαρμογής

1. Ενεργοποίηση virtual environment: `.\.venv\Scripts\activate`
2. Εκτέλεση: `python web_app/app.py`
3. Πρόσβαση στη διεύθυνση: http://127.0.0.1:5000

## Τεχνικές Λεπτομέρειες

### Περιβάλλον

- Python 3.12
- PyTorch
- Transformers
- Flask
- Plotly

### Fine-tuned Μοντέλο

Το μοντέλο fine-tuned BERT επιτυγχάνει ακρίβεια >96% στην ανάλυση συναισθήματος για ελληνικά κείμενα, με δύο κλάσεις (Θετικό/Αρνητικό).

## Ανάπτυξη και Συντήρηση

### Προσθήκη Νέων Λειτουργιών

Για την προσθήκη νέων λειτουργιών, συνιστάται:

1. Προσθήκη νέων scripts στον κατάλληλο φάκελο (models/, utils/)
2. Ενημέρωση του αρχείου `requirements.txt` αν απαιτούνται νέα πακέτα
3. Ενημέρωση του `index.md` και του `README.md`

### Αντιμετώπιση Προβλημάτων

Συνήθη προβλήματα και λύσεις:

1. **Σφάλμα φόρτωσης μοντέλου**: Βεβαιωθείτε ότι υπάρχει ο φάκελος `models/fine_tuned_bert/`
2. **Σφάλματα διαδικτυακής εφαρμογής**: Ελέγξτε τα logs στο terminal όπου τρέχει η εφαρμογή 