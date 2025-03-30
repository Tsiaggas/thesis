import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ανάγνωση του dataset
df = pd.read_excel('Skroutz_dataset.xlsx')

# Προσθήκη στήλης με το μήκος κάθε κριτικής
df['length'] = df['review'].apply(len)

# Στατιστικά στοιχεία για τα μήκη
print("Στατιστικά μήκους κριτικών:")
print(df['length'].describe())

# Στατιστικά ανά κατηγορία (θετικές/αρνητικές)
print("\nΜέσο μήκος ανά κατηγορία:")
print(df.groupby('label')['length'].mean())

# Εκτύπωση των 5 συντομότερων κριτικών
print("\n5 συντομότερες κριτικές:")
short_reviews = df.sort_values('length').head(5)
for i, row in short_reviews.iterrows():
    print(f"Κριτική {row['id']} (μήκος {row['length']}): {row['review']} - {row['label']}")

# Εκτύπωση των 5 μεγαλύτερων κριτικών
print("\n5 μεγαλύτερες κριτικές:")
long_reviews = df.sort_values('length', ascending=False).head(5)
for i, row in long_reviews.iterrows():
    print(f"Κριτική {row['id']} (μήκος {row['length']}): {row['review'][:50]}... - {row['label']}")

# Αποθήκευση των αποτελεσμάτων για μελλοντική χρήση
length_stats = {
    'mean': df['length'].mean(),
    'median': df['length'].median(),
    'min': df['length'].min(),
    'max': df['length'].max(),
    'positive_mean': df[df['label'] == 'Positive']['length'].mean(),
    'negative_mean': df[df['label'] == 'Negative']['length'].mean()
} 