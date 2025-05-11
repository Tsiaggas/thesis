#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script για την εγκατάσταση των απαραίτητων βιβλιοθηκών και δεδομένων
για την εφαρμογή ανάλυσης συναισθημάτων.
"""

import os
import sys
import subprocess
import importlib
import time

def print_banner():
    """Εμφανίζει banner εγκατάστασης"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║       ΕΓΚΑΤΑΣΤΑΣΗ ΒΙΒΛΙΟΘΗΚΩΝ ΑΝΑΛΥΣΗΣ ΣΥΝΑΙΣΘΗΜΑΤΩΝ     ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_package(package_name):
    """Ελέγχει αν μια βιβλιοθήκη είναι εγκατεστημένη"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name, version=None):
    """Εγκαθιστά ένα πακέτο Python"""
    package_spec = package_name
    if version:
        package_spec = f"{package_name}=={version}"
        
    print(f"Εγκατάσταση {package_spec}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        print(f"Το πακέτο {package_name} εγκαταστάθηκε επιτυχώς!")
        return True
    except subprocess.CalledProcessError:
        print(f"Σφάλμα κατά την εγκατάσταση του πακέτου {package_name}")
        return False

def download_corpora():
    """Κατεβάζει τα απαραίτητα corpora για το TextBlob"""
    try:
        print("Εγκατάσταση των απαραίτητων corpora για TextBlob...")
        # Προσπαθούμε να εισάγουμε το TextBlob
        import textblob.download_corpora
        
        # Κατέβασμα των corpora
        textblob.download_corpora.download_all()
        print("Τα corpora για το TextBlob εγκαταστάθηκαν επιτυχώς!")
        return True
    except Exception as e:
        print(f"Σφάλμα κατά την εγκατάσταση των corpora: {e}")
        return False

def download_nltk_data():
    """Κατεβάζει τα απαραίτητα δεδομένα για το NLTK"""
    try:
        print("Εγκατάσταση των απαραίτητων δεδομένων για NLTK...")
        import nltk
        
        # Κατέβασμα δεδομένων που χρειάζονται για το NRCLex και άλλες λειτουργίες
        nltk_packages = [
            'punkt',
            'stopwords',
            'wordnet',
            'vader_lexicon',
            'averaged_perceptron_tagger'
        ]
        
        for package in nltk_packages:
            try:
                print(f"Κατέβασμα πακέτου NLTK: {package}")
                nltk.download(package, quiet=True)
                print(f"Επιτυχής εγκατάσταση πακέτου NLTK: {package}")
            except Exception as e:
                print(f"Σφάλμα κατά την εγκατάσταση του πακέτου NLTK {package}: {e}")
        
        return True
    except Exception as e:
        print(f"Σφάλμα κατά την εγκατάσταση των NLTK δεδομένων: {e}")
        return False

def main():
    print_banner()
    
    # Λίστα με τις απαραίτητες βιβλιοθήκες και τις εκδόσεις τους
    required_packages = [
        ("vaderSentiment", "3.3.2"),
        ("nrclex", "3.0.0"),
        ("textblob", "0.17.1"),
        ("nltk", "3.9.1"),
        ("flask-cors", "4.0.0")
    ]
    
    # Έλεγχος και εγκατάσταση των πακέτων που λείπουν
    packages_installed = []
    for package_name, version in required_packages:
        if check_package(package_name.replace("-", "_").split(".")[0]):
            print(f"Το πακέτο {package_name} είναι ήδη εγκατεστημένο.")
            packages_installed.append(package_name)
        else:
            if install_package(package_name, version):
                packages_installed.append(package_name)
    
    # Εγκατάσταση των απαραίτητων δεδομένων
    if "textblob" in packages_installed:
        download_corpora()
    
    if "nltk" in packages_installed:
        download_nltk_data()
    
    # Τελική αναφορά
    total_packages = len(required_packages)
    installed_packages = len(packages_installed)
    
    print("\n" + "=" * 60)
    print(f"Εγκατάσταση ολοκληρώθηκε: {installed_packages}/{total_packages} πακέτα")
    print("=" * 60)
    
    if installed_packages == total_packages:
        print("Όλα τα απαραίτητα πακέτα εγκαταστάθηκαν επιτυχώς!")
        print("Μπορείτε τώρα να εκκινήσετε την εφαρμογή με την εντολή:")
        print("\npython web_app/start_app.py\n")
    else:
        print("ΠΡΟΣΟΧΗ: Ορισμένα πακέτα δεν εγκαταστάθηκαν.")
        print("Παρακαλώ εγκαταστήστε χειροκίνητα τα υπόλοιπα πακέτα:")
        for package_name, version in required_packages:
            if package_name not in packages_installed:
                print(f"pip install {package_name}=={version}")

if __name__ == "__main__":
    main() 