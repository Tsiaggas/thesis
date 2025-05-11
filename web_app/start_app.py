#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script για την εκκίνηση της εφαρμογής ανάλυσης συναισθημάτων.
Ξεκινά τον Flask server και το React frontend.
"""

import os
import subprocess
import sys
import time
import threading
import webbrowser
import platform

def print_banner():
    """Εμφανίζει ένα banner για την εφαρμογή."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║          ΕΦΑΡΜΟΓΗ ΑΝΑΛΥΣΗΣ ΣΥΝΑΙΣΘΗΜΑΤΩΝ                  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    
     * BERT μοντέλο 3 κλάσεων (Θετικό/Αρνητικό/Ουδέτερο)
     * VADER λεξικό συναισθημάτων
     * NRCLex ανάλυση συγκεκριμένων συναισθημάτων
     * LSA (Λανθάνουσα Σημασιολογική Ανάλυση)
     
    """
    print(banner)

def start_flask_server():
    """Εκκινεί τον Flask server στο port 5000."""
    print("Εκκίνηση του Flask API server...")
    # Εκτέλεση του Flask server σε ξεχωριστή διεργασία
    flask_cmd = [sys.executable, "app.py"]
    flask_process = subprocess.Popen(flask_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return flask_process

def start_react_frontend():
    """Εκκινεί το React frontend με την εντολή npm start."""
    print("Εκκίνηση του React frontend...")
    # Αλλαγή στον φάκελο του React frontend
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "react-frontend")
    
    # Εκτέλεση του npm start σε ξεχωριστή διεργασία
    if os.name == 'nt':  # Windows
        react_cmd = ["npm.cmd", "start"]
    else:  # Linux/Mac
        react_cmd = ["npm", "start"]
    
    react_process = subprocess.Popen(react_cmd, cwd=frontend_dir)
    return react_process

def open_browser():
    """Ανοίγει τον browser στην εφαρμογή μετά από καθυστέρηση."""
    time.sleep(5)  # Περιμένουμε λίγο για να ξεκινήσουν οι servers
    url = "http://localhost:3000"
    print(f"Άνοιγμα εφαρμογής στο browser: {url}")
    webbrowser.open(url)

def print_system_info():
    """Εμφανίζει πληροφορίες για το σύστημα."""
    print("\n--- Πληροφορίες Συστήματος ---")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Πληροφορίες για τις βιβλιοθήκες ανάλυσης συναισθημάτων
    try:
        import pkg_resources
        packages = ["transformers", "torch", "vaderSentiment", "nrclex", "scikit-learn"]
        for pkg in packages:
            try:
                version = pkg_resources.get_distribution(pkg).version
                print(f"{pkg}: v{version}")
            except pkg_resources.DistributionNotFound:
                print(f"{pkg}: Δεν βρέθηκε")
    except:
        print("Δεν ήταν δυνατή η ανάκτηση πληροφοριών για τα εγκατεστημένα πακέτα.")
    
    print("-------------------------\n")

if __name__ == "__main__":
    # Εμφάνιση banner και πληροφοριών συστήματος
    print_banner()
    print_system_info()
    
    # Εκκίνηση του Flask API server
    flask_process = start_flask_server()
    
    # Εκκίνηση του React frontend
    react_process = start_react_frontend()
    
    # Άνοιγμα του browser μετά από μικρή καθυστέρηση
    threading.Thread(target=open_browser).start()
    
    print("\nΗ εφαρμογή ξεκίνησε!")
    print("- Flask API server: http://localhost:5000")
    print("- React frontend: http://localhost:3000")
    print("\nΓια να τερματίσετε την εφαρμογή, πατήστε Ctrl+C.\n")
    
    try:
        # Περιμένουμε μέχρι να διακοπεί η εκτέλεση με Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nΤερματισμός της εφαρμογής...")
        # Τερματισμός των διεργασιών
        flask_process.terminate()
        react_process.terminate()
        print("Η εφαρμογή τερματίστηκε με επιτυχία.") 