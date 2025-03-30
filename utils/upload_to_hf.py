"""
Script για το ανέβασμα του μοντέλου ανάλυσης συναισθήματος 3 κλάσεων στο Hugging Face Hub.
Το μοντέλο θα είναι διαθέσιμο δημόσια και μπορεί να χρησιμοποιηθεί από οποιονδήποτε.
"""

import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login

def upload_model_to_huggingface(
    model_path="models/fine_tuned_bert_3class",
    repo_name=None,
    token=None
):
    """
    Ανεβάζει το μοντέλο και τον tokenizer στο Hugging Face Hub.
    
    Args:
        model_path (str): Τοπικό μονοπάτι του μοντέλου
        repo_name (str): Όνομα του repository στο Hugging Face (π.χ. "username/skroutz-sentiment-3class")
        token (str): Το API token του Hugging Face (προαιρετικό αν έχετε ήδη συνδεθεί)
    """
    print(f"Φόρτωση μοντέλου από: {model_path}")
    
    # Έλεγχος αν το μοντέλο υπάρχει
    if not os.path.exists(model_path):
        print(f"Σφάλμα: Το μοντέλο δεν βρέθηκε στο μονοπάτι {model_path}")
        return
    
    # Σύνδεση στο Hugging Face Hub αν δόθηκε token
    if token:
        print("Σύνδεση στο Hugging Face Hub...")
        login(token=token)
    
    # Προσδιορισμός repository name αν δεν δόθηκε
    if not repo_name:
        print("ΣΦΑΛΜΑ: Πρέπει να δώσετε ένα όνομα repository (username/repo-name)")
        print("Παράδειγμα: your-username/skroutz-sentiment-3class")
        return
    
    try:
        # Φόρτωση του μοντέλου και του tokenizer
        print("Φόρτωση μοντέλου και tokenizer...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ανέβασμα στο Hugging Face Hub
        print(f"Ανέβασμα μοντέλου στο {repo_name}...")
        model.push_to_hub(repo_name)
        
        print(f"Ανέβασμα tokenizer στο {repo_name}...")
        tokenizer.push_to_hub(repo_name)
        
        print("\n✅ Το μοντέλο ανέβηκε επιτυχώς στο Hugging Face Hub!")
        print(f"   Είναι πλέον διαθέσιμο στη διεύθυνση: https://huggingface.co/{repo_name}")
        print("\nΜπορείτε να το χρησιμοποιήσετε με τον παρακάτω κώδικα:")
        print("```python")
        print(f"from transformers import AutoModelForSequenceClassification, AutoTokenizer")
        print(f"")
        print(f"model = AutoModelForSequenceClassification.from_pretrained(\"{repo_name}\")")
        print(f"tokenizer = AutoTokenizer.from_pretrained(\"{repo_name}\")")
        print("```")
        
    except Exception as e:
        print(f"Σφάλμα κατά το ανέβασμα: {e}")
        print("\nΠιθανές αιτίες:")
        print("- Μη έγκυρο token")
        print("- Το όνομα repository υπάρχει ήδη αλλά δεν έχετε δικαιώματα σε αυτό")
        print("- Προβλήματα δικτύου")
        print("- Το μοντέλο είναι πολύ μεγάλο (> 10GB)")
        
        print("\nΣυμβουλή: Δοκιμάστε να συνδεθείτε ξανά με την εντολή:")
        print("huggingface-cli login")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ανέβασμα μοντέλου στο Hugging Face Hub")
    parser.add_argument("--model_path", default="models/fine_tuned_bert_3class", 
                        help="Μονοπάτι του μοντέλου (προεπιλογή: models/fine_tuned_bert_3class)")
    parser.add_argument("--repo_name", required=True,
                        help="Όνομα του repository στο Hugging Face (π.χ. username/skroutz-sentiment-3class)")
    parser.add_argument("--token", 
                        help="Hugging Face API token (προαιρετικό αν έχετε ήδη συνδεθεί)")
    
    args = parser.parse_args()
    
    upload_model_to_huggingface(
        model_path=args.model_path,
        repo_name=args.repo_name,
        token=args.token
    ) 