import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar els arxius separats
data_dir = "data/comparacio_idiomes/"

# Espanyol
train_data_es = pd.read_csv(data_dir + "ressenyes_es_train.csv")
valid_data_es = pd.read_csv(data_dir + "ressenyes_es_val.csv")
test_data_es = pd.read_csv(data_dir + "ressenyes_es_test.csv")

# Anglès
train_data_en = pd.read_csv(data_dir + "ressenyes_en_train.csv")
valid_data_en = pd.read_csv(data_dir + "ressenyes_en_val.csv")
test_data_en = pd.read_csv(data_dir + "ressenyes_en_test.csv")

# 2. Funció per preparar les dades
def prepara_dades(train_data, valid_data, test_data):
    X_train = train_data['description'].fillna("")
    y_train = train_data['score']
    X_valid = valid_data['description'].fillna("")
    y_valid = valid_data['score']
    X_test = test_data['description'].fillna("")
    y_test = test_data['score']
    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Preparar dades espanyol
X_train_es, y_train_es, X_valid_es, y_valid_es, X_test_es, y_test_es = prepara_dades(
    train_data_es, valid_data_es, test_data_es
)

# Preparar dades anglès
X_train_en, y_train_en, X_valid_en, y_valid_en, X_test_en, y_test_en = prepara_dades(
    train_data_en, valid_data_en, test_data_en
)

# 3. Vectoritzar el text amb TF-IDF
vectorizer_es = TfidfVectorizer(max_features=5000)
X_train_tfidf_es = vectorizer_es.fit_transform(X_train_es)
X_valid_tfidf_es = vectorizer_es.transform(X_valid_es)
X_test_tfidf_es = vectorizer_es.transform(X_test_es)

vectorizer_en = TfidfVectorizer(max_features=5000)
X_train_tfidf_en = vectorizer_en.fit_transform(X_train_en)
X_valid_tfidf_en = vectorizer_en.transform(X_valid_en)
X_test_tfidf_en = vectorizer_en.transform(X_test_en)

# 4. Funció per plotejar la matriu de confusió
def plot_confusion_matrix(y_true, y_pred, idioma, title="Matriu de Confusió"):
    cm = confusion_matrix(y_true, y_pred)  # Matriu sense normalitzar
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Percentatges per fila
    labels = sorted(set(y_true))  # Etiquetes úniques ordenades
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{title} - {idioma}")
    plt.xlabel("Prediccions")
    plt.ylabel("Valors Reals")
    plt.show()

# 5. Entrenar i avaluar models
def entrena_i_avaluar(X_train_tfidf, y_train, X_valid_tfidf, y_valid, X_test_tfidf, y_test, idioma):
    # Entrenar SVM
    model = LinearSVC()
    model.fit(X_train_tfidf, y_train)
    
    # Validació
    y_valid_pred = model.predict(X_valid_tfidf)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    print(f"\nExactitud del model en validació ({idioma}): {valid_accuracy:.4f}")
    
    # Test
    y_test_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Exactitud del model en test ({idioma}): {test_accuracy:.4f}")
    
    # Informes
    print(f"\nClassification Report ({idioma} - Test):")
    print(classification_report(y_test, y_test_pred))
    
    # Matriu de confusió
    print(f"Matriu de Confusió ({idioma} - Test):")
    plot_confusion_matrix(y_test, y_test_pred, idioma, title="Matriu de Confusió")

# Entrenar i avaluar per a espanyol
entrena_i_avaluar(X_train_tfidf_es, y_train_es, X_valid_tfidf_es, y_valid_es, X_test_tfidf_es, y_test_es, "Espanyol")
print(train_data_es['score'].value_counts())

# Entrenar i avaluar per a anglès
entrena_i_avaluar(X_train_tfidf_en, y_train_en, X_valid_tfidf_en, y_valid_en, X_test_tfidf_en, y_test_en, "Anglès")
print(train_data_en['score'].value_counts())
