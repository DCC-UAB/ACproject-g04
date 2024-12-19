import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar els arxius separats
data_dir = "data/"

train_data = pd.read_csv(data_dir + "train_set.csv")
valid_data = pd.read_csv(data_dir + "val_set.csv")
test_data = pd.read_csv(data_dir + "test_set.csv")

# 2. Preparar les dades
X_train = train_data['description'].fillna("")  # Ressenyes d'entrenament: Substituir NaN per cadena buida 
y_train = train_data['score']  # Etiquetes d'entrenament

X_valid = valid_data['description'].fillna("")  # Ressenyes de validació: Substituir NaN per cadena buida 
y_valid = valid_data['score']  # Etiquetes de validació

X_test = test_data['description'].fillna("")  # Ressenyes de test: Substituir NaN per cadena buida
y_test = test_data['score']  # Etiquetes de test

# 3. Vectoritzar el text amb TF-IDF (ajustat amb dades d'entrenament)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)
X_test_tfidf = vectorizer.transform(X_test)

# Convertir el TF-IDF de X_train a un DataFrame llegible (només per inspecció)
feature_names = vectorizer.get_feature_names_out()
X_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names)

print("Primers 5 vectors TF-IDF:")
print(X_train_df.head())

# 4. Entrenar el model SVM per a regressió
svm_model = LinearSVR()
svm_model.fit(X_train_tfidf, y_train)

# 5. Validar el model amb el conjunt de validació
y_valid_pred = svm_model.predict(X_valid_tfidf)

mae_valid = mean_absolute_error(y_valid, y_valid_pred)
mse_valid = mean_squared_error(y_valid, y_valid_pred)
r2_valid = r2_score(y_valid, y_valid_pred)

print("\n--- Resultats de Validació ---")
print(f"MAE: {mae_valid:.4f}")
print(f"MSE: {mse_valid:.4f}")
print(f"R²: {r2_valid:.4f}")

# 6. Avaluar el model amb el conjunt de test
y_test_pred = svm_model.predict(X_test_tfidf)

mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("\n--- Resultats de Test ---")
print(f"MAE: {mae_test:.4f}")
print(f"MSE: {mse_test:.4f}")
print(f"R²: {r2_test:.4f}")

# 7. Visualització dels resultats
def plot_predictions(y_true, y_pred, title="Prediccions vs Valors Reals"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, color="blue", alpha=0.6, edgecolor="k")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.title(title)
    plt.xlabel("Valors Reals")
    plt.ylabel("Prediccions")
    plt.grid(True)
    plt.show()

# Visualitzar resultats per al conjunt de validació
print("\nVisualització: Validació")
plot_predictions(y_valid, y_valid_pred, title="Prediccions vs Valors Reals (Validació)")

# Visualitzar resultats per al conjunt de test
print("Visualització: Test")
plot_predictions(y_test, y_test_pred, title="Prediccions vs Valors Reals (Test)")
