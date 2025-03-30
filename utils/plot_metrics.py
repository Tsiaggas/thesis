import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Διαδρομές ---
# Υποθέτουμε ότι το script τρέχει από τον κύριο φάκελο 'thesis'
STATE_FILE_PATH = os.path.join('models', 'fine_tuned_bert_3class', 'trainer_state.json')
RESULTS_DIR = 'results'

# --- Συνάρτηση για δημιουργία και αποθήκευση γραφήματος ---
def plot_and_save(x_values, y_values, title, xlabel, ylabel, filename, y_limit=None, x_is_step=True):
    """Δημιουργεί και αποθηκεύει ένα γράφημα."""
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    
    # Χρήση ακέραιων για τον άξονα x αν είναι βήματα ή εποχές
    if all(isinstance(x, (int, float)) and x == int(x) for x in x_values):
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout() # Προσαρμογή διάταξης για να χωράνε οι ετικέτες
    
    if y_limit:
        plt.ylim(y_limit) # Ορισμός ορίων άξονα y αν δίνονται
        
    output_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(output_path)
    print(f"Το γράφημα αποθηκεύτηκε στο: {output_path}")
    plt.close() # Κλείσιμο του plot για εξοικονόμηση μνήμης

# --- Συνάρτηση για δημιουργία και αποθήκευση γραφήματος με δύο γραμμές ---
def plot_and_save_two_lines(x1, y1, label1, x2, y2, label2, title, xlabel, ylabel, filename, y_limit=None):
    """Δημιουργεί και αποθηκεύει ένα γράφημα με δύο γραμμές."""
    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, marker='o', linestyle='-', label=label1)
    plt.plot(x2, y2, marker='s', linestyle='--', label=label2)

    # Χρήση ακέραιων για τον άξονα x
    if all(isinstance(x, (int, float)) and x == int(x) for x in x1 + x2):
         plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend() # Εμφάνιση legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    if y_limit:
        plt.ylim(y_limit)
        
    output_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(output_path)
    print(f"Το γράφημα αποθηκεύτηκε στο: {output_path}")
    plt.close()

# --- Κύρια Λειτουργία ---
def main():
    # Έλεγχος αν υπάρχει το αρχείο state
    if not os.path.exists(STATE_FILE_PATH):
        print(f"Σφάλμα: Το αρχείο '{STATE_FILE_PATH}' δεν βρέθηκε.")
        return

    # Δημιουργία φακέλου results αν δεν υπάρχει
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Φόρτωση του JSON state
    try:
        with open(STATE_FILE_PATH, 'r', encoding='utf-8') as f:
            state = json.load(f)
    except Exception as e:
        print(f"Σφάλμα κατά την ανάγνωση του αρχείου JSON: {e}")
        return
        
    # Έλεγχος αν υπάρχει το log_history
    if 'log_history' not in state or not state['log_history']:
        print("Σφάλμα: Δεν βρέθηκε ή είναι κενό το 'log_history' στο αρχείο JSON.")
        return
        
    log_history = state['log_history']
    
    # Εξαγωγή δεδομένων
    steps = []
    epochs = []
    training_loss = []
    eval_steps = []
    eval_epochs = []
    eval_loss = []
    eval_accuracy = []
    eval_f1 = []
    eval_precision = []
    eval_recall = []

    # Προσδιορισμός αν η καταγραφή γίνεται ανά βήμα ή εποχή κυρίως για τα eval metrics
    # Αν τα eval metrics έχουν 'epoch' και είναι διαφορετικά, τα χρησιμοποιούμε ως x-axis
    # Αλλιώς, χρησιμοποιούμε το 'step'
    eval_has_distinct_epochs = False
    temp_eval_epochs = [log.get('epoch') for log in log_history if 'eval_loss' in log]
    if len(set(e for e in temp_eval_epochs if e is not None)) > 1:
         eval_has_distinct_epochs = True

    # Το Training loss συνήθως καταγράφεται πιο συχνά (ανά logging_steps)
    train_steps = []
    train_epochs_approx = [] # Κατ' εκτίμηση εποχή για training loss
    
    for log in log_history:
        step = log.get('step')
        epoch = log.get('epoch')

        if 'loss' in log and step is not None: # Training log
            steps.append(step)
            epochs.append(epoch) # Μπορεί να είναι float
            training_loss.append(log['loss'])
            train_steps.append(step)
            train_epochs_approx.append(epoch)

        elif 'eval_loss' in log and step is not None: # Evaluation log
             eval_steps.append(step)
             eval_epochs.append(epoch) # Μπορεί να είναι float αλλά συνήθως ακέραιο στο τέλος εποχής
             eval_loss.append(log['eval_loss'])
             if 'eval_accuracy' in log: eval_accuracy.append(log['eval_accuracy'])
             if 'eval_f1_weighted' in log: eval_f1.append(log['eval_f1_weighted'])
             if 'eval_precision_weighted' in log: eval_precision.append(log['eval_precision_weighted'])
             if 'eval_recall_weighted' in log: eval_recall.append(log['eval_recall_weighted'])

    if not training_loss and not eval_loss:
        print("Δεν βρέθηκαν δεδομένα για loss στο log history.")
        return # Δεν υπάρχουν δεδομένα για σχεδίαση

    # Επιλογή άξονα Χ για τα γραφήματα αξιολόγησης
    eval_x_axis_label = "Epoch" if eval_has_distinct_epochs else "Step"
    eval_x_values = eval_epochs if eval_has_distinct_epochs else eval_steps
    
    # Επιλογή άξονα Χ για τα γραφήματα εκπαίδευσης (συνήθως Steps)
    train_x_axis_label = "Step"
    train_x_values = train_steps
    
    # --- Δημιουργία Γραφημάτων ---
    
    # 1. Loss (Training vs Validation)
    if training_loss and eval_loss:
        plot_and_save_two_lines(
            train_x_values, training_loss, "Training Loss",
            eval_x_values, eval_loss, "Validation Loss",
            title="Training vs Validation Loss",
            xlabel=eval_x_axis_label, # Κοινός άξονας (συνήθως Steps ή Epochs)
            ylabel="Loss",
            filename="loss_train_vs_validation.png",
            y_limit=[0, max(max(training_loss), max(eval_loss)) * 1.1] # Όριο y λίγο πάνω από το max loss
        )
    elif training_loss:
         plot_and_save(train_x_values, training_loss, "Training Loss", train_x_axis_label, "Loss", "training_loss.png")
    elif eval_loss:
         plot_and_save(eval_x_values, eval_loss, "Validation Loss", eval_x_axis_label, "Loss", "validation_loss.png")
         
    # 2. Accuracy (Validation)
    if eval_accuracy:
        plot_and_save(eval_x_values, eval_accuracy, "Validation Accuracy", eval_x_axis_label, "Accuracy", "validation_accuracy.png", y_limit=[min(eval_accuracy)*0.95, 1.0]) # Όριο y από λίγο κάτω από min accuracy έως 1

    # 3. F1 Score (Validation)
    if eval_f1:
        plot_and_save(eval_x_values, eval_f1, "Validation F1 Score (Weighted)", eval_x_axis_label, "F1 Score", "validation_f1_weighted.png", y_limit=[min(eval_f1)*0.95, 1.0])

    # 4. Precision (Validation)
    if eval_precision:
        plot_and_save(eval_x_values, eval_precision, "Validation Precision (Weighted)", eval_x_axis_label, "Precision", "validation_precision_weighted.png", y_limit=[min(eval_precision)*0.95, 1.0])
        
    # 5. Recall (Validation)
    if eval_recall:
        plot_and_save(eval_x_values, eval_recall, "Validation Recall (Weighted)", eval_x_axis_label, "Recall", "validation_recall_weighted.png", y_limit=[min(eval_recall)*0.95, 1.0])

    print("\nΌλα τα γραφήματα δημιουργήθηκαν στον φάκελο:", RESULTS_DIR)

if __name__ == "__main__":
    # Σιγουρευόμαστε ότι έχουμε το matplotlib
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("Σφάλμα: Η βιβλιοθήκη matplotlib δεν είναι εγκατεστημένη.")
        print("Παρακαλώ εκτελέστε: pip install matplotlib")
        exit()
        
    main() 