# Τεκμηρίωση Μοντέλου BERT για Ανάλυση Συναισθήματος

Αυτό το έγγραφο περιγράφει το fine-tuned BERT μοντέλο που χρησιμοποιείται για την ανάλυση συναισθήματος ελληνικών κειμένων στο project.

## Βασικό Μοντέλο

Το βασικό μοντέλο που χρησιμοποιήθηκε είναι το [nlpaueb/bert-base-greek-uncased-v1](https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1), το οποίο είναι ένα μοντέλο BERT προ-εκπαιδευμένο σε μεγάλο corpus ελληνικών κειμένων.

Χαρακτηριστικά του βασικού μοντέλου:
- Αρχιτεκτονική: BERT-Base
- Γλώσσα: Ελληνικά
- Uncased (μη ευαίσθητο σε κεφαλαία/πεζά)
- 12 επίπεδα (layers)
- 768 διαστάσεις στο κρυφό επίπεδο (hidden)
- 12 κεφαλές προσοχής (attention heads)
- 110M παραμέτρους

## Fine-tuning

Το βασικό μοντέλο υποβλήθηκε σε fine-tuning για την εργασία ανάλυσης συναισθήματος (sentiment analysis) χρησιμοποιώντας ένα dataset από κριτικές προϊόντων από το Skroutz.

### Dataset Fine-tuning

- **Προέλευση**: Κριτικές προϊόντων από το Skroutz.gr
- **Μέγεθος**: ~XXXXX δείγματα
- **Κλάσεις**: Δυαδική κατηγοριοποίηση (Θετικό/Αρνητικό)
- **Διαχωρισμός**: 80% εκπαίδευση, 10% επικύρωση, 10% δοκιμή
- **Προεπεξεργασία**: Αφαίρεση ειδικών χαρακτήρων, κανονικοποίηση κειμένου

### Παράμετροι Fine-tuning

Οι βασικές παράμετροι που χρησιμοποιήθηκαν για το fine-tuning του μοντέλου είναι:

- **Ρυθμός μάθησης (learning rate)**: 2e-5
- **Batch size**: 16
- **Εποχές (epochs)**: 4
- **Μέγιστο μήκος ακολουθίας**: 128 tokens
- **Optimizer**: AdamW
- **Weight decay**: 0.01
- **Warmup steps**: 500

Η διαδικασία fine-tuning εκτελείται με το script `models/train_bert_finetune.py`.

## Απόδοση Μοντέλου

Το fine-tuned μοντέλο πέτυχε τις εξής μετρικές στο test set:

- **Ακρίβεια (Accuracy)**: >96%
- **Precision**: Θετικό: 0.97, Αρνητικό: 0.95
- **Recall**: Θετικό: 0.96, Αρνητικό: 0.96
- **F1-score**: Θετικό: 0.96, Αρνητικό: 0.95

## Χρήση του Μοντέλου

### Φόρτωση του Μοντέλου

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Φόρτωση του tokenizer και του μοντέλου
model_dir = "models/fine_tuned_bert"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Δημιουργία pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)
```

### Χρήση για Πρόβλεψη

```python
# Πρόβλεψη για ένα κείμενο
text = "Το προϊόν είναι εξαιρετικό και λειτουργεί τέλεια!"
result = sentiment_pipeline(text)[0]

# Επεξεργασία αποτελέσματος
score_neg = next(item['score'] for item in result if item['label'] == 'LABEL_0')
score_pos = next(item['score'] for item in result if item['label'] == 'LABEL_1')

sentiment = "Θετικό" if score_pos > score_neg else "Αρνητικό"
print(f"Συναίσθημα: {sentiment}, Πιθανότητα: {max(score_pos, score_neg):.2f}")
```

## Περιορισμοί

- Το μοντέλο είναι βελτιστοποιημένο για κείμενα παρόμοια με κριτικές προϊόντων
- Έχει περιορισμό στο μήκος των κειμένων (max 512 tokens)
- Δεν λαμβάνει υπόψη το context πέρα από το ίδιο το κείμενο
- Καλύτερη απόδοση σε ξεκάθαρα θετικά ή αρνητικά κείμενα

## Επεκτάσεις και Μελλοντικές Βελτιώσεις

Πιθανές επεκτάσεις για το μοντέλο περιλαμβάνουν:

1. Fine-tuning σε μεγαλύτερο και πιο ποικίλο dataset
2. Επέκταση σε περισσότερες κλάσεις συναισθημάτων
3. Χρήση μεγαλύτερων εκδόσεων του BERT ή άλλων αρχιτεκτονικών (π.χ. RoBERTa)
4. Εφαρμογή τεχνικών όπως model distillation για βελτίωση απόδοσης 