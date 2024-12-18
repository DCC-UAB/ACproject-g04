import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar els arxius separats
data_dir = "data/"

train_data = pd.read_csv(data_dir + "train_set.csv")
valid_data = pd.read_csv(data_dir + "val_set.csv")
test_data = pd.read_csv(data_dir + "test_set.csv")

# 2. Preparar les dades
X_train = train_data['description'].fillna("") # Ressenyes d'entrenament: Substituir NaN per cadena buida 
y_train = train_data['score']  # Etiquetes d'entrenament

X_valid = valid_data['description'].fillna("") # Ressenyes de validació: Substituir NaN per cadena buida 
y_valid = valid_data['score']  # Etiquetes de validació

X_test = test_data['description'] # Ressenyes de test: Substituir NaN per cadena buida
y_test = test_data['score']  # Etiquetes de test

# 3. Vectoritzar el text amb TF-IDF (ajustat amb dades d'entrenament)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)
X_test_tfidf = vectorizer.transform(X_test)

# Convertir el TF-IDF de X_train a un DataFrame llegible
feature_names = vectorizer.get_feature_names_out()
X_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names)

# Imprimir una part del TF-IDF per veure les característiques
print("Primers 5 vectors TF-IDF:")
print(X_train_df.head())

# 4. Entrenar el model SVM amb el conjunt d'entrenament
svm_model = LinearSVC() #classification
svm_model.fit(X_train_tfidf, y_train)

# 5. Validar el model amb el conjunt de validació
y_valid_pred = svm_model.predict(X_valid_tfidf)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print(f"Exactitud del model en validació: {valid_accuracy:.4f}")

# 6. Avaluar el model amb el conjunt de test
y_test_pred = svm_model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Exactitud del model en test: {test_accuracy:.4f}")

# 7. Informes finals
print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred))

# 8. Funció per a la matriu de confusió amb colors
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Percentatges per fila
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Matriu de Confusió (%)")
    plt.ylabel("Valor Real")
    plt.xlabel("Predicció")
    plt.show()

# 9. Matriu de Confusió
print("Matriu de Confusió (Test):")
unique_labels = sorted(set(y_test))  # Etiquetes úniques per eix
plot_confusion_matrix(y_test, y_test_pred, labels=unique_labels)
 
