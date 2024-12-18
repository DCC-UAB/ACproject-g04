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

# Eliminar les ressenyes amb score 3 (neutral) per fer classificació binària
train_data = train_data[train_data['score'] != 3]
valid_data = valid_data[valid_data['score'] != 3]
test_data = test_data[test_data['score'] != 3]

# 2. Preparar les dades
X_train = train_data['description'].fillna("")  # Substituir NaN per cadena buida
y_train = train_data['score']

X_valid = valid_data['description'].fillna("")
y_valid = valid_data['score']

X_test = test_data['description'].fillna("")
y_test = test_data['score']

# Reclassificar les etiquetes a binari: 0 = Negatiu, 1 = Positiu
y_train = y_train.replace({1: 0, 2: 0, 4: 1, 5: 1})
y_valid = y_valid.replace({1: 0, 2: 0, 4: 1, 5: 1})
y_test = y_test.replace({1: 0, 2: 0, 4: 1, 5: 1})

# 3. Vectoritzar el text amb TF-IDF (ajustat amb dades d'entrenament)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Entrenar el model SVM amb el conjunt d'entrenament
svm_model = LinearSVC()
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

# 8. Funció per plotejar la matriu de confusió
def plot_confusion_matrix(y_true, y_pred, title="Matriu de Confusió"):
    cm = confusion_matrix(y_true, y_pred)  # Matriu sense normalitzar
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Percentatges per fila
    labels = ["Negatiu", "Positiu"]  # Etiquetes per a classificació binària
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Prediccions")
    plt.ylabel("Valors Reals")
    plt.show()

# 9. Mostrar la matriu de confusió
print("\nMatriu de Confusió (Test):")
plot_confusion_matrix(y_test, y_test_pred, title="Matriu de Confusió - Test")
