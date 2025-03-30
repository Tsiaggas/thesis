import pandas as pd

# Ανάγνωση του dataset
df = pd.read_excel('Skroutz_dataset.xlsx')

# Εμφάνιση των πρώτων 5 γραμμών
print("Πρώτες 5 εγγραφές:")
print(df.head())

# Πληροφορίες για τις στήλες
print("\nΣτήλες:", df.columns.tolist())

# Μέγεθος του dataset
print("\nΜέγεθος dataset:", df.shape)

# Κατανομή ετικετών
print("\nΚατανομή ετικετών:")
print(df['label'].value_counts())

# Βασικά στατιστικά
print("\nΒασικά στατιστικά:")
print(df.describe(include='all'))

# Έλεγχος για κενές τιμές
print("\nΚενές τιμές:")
print(df.isnull().sum()) 