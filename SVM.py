import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
svm_model = LinearSVC() #regression
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

print("Matriu de Confusió (Test):")
print(confusion_matrix(y_test, y_test_pred))
 
