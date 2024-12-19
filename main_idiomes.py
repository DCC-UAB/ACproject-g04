import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVR
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
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

# Funció per avaluar models de regressió
def avaluar_regressor(model, model_name, X_train, y_train, X_valid, y_valid, X_test, y_test, idioma):
    """
    Avaluar models de regressió amb mètriques de regressió (MSE, RMSE, MAE, R2)
    """
    print(f"\n### Model: {model_name} ({idioma}) ###")
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    # Mètriques de regressió
    val_mse = mean_squared_error(y_valid, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_valid, y_val_pred)
    val_r2 = r2_score(y_valid, y_val_pred)

    print(f"Validació ({idioma}) - MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Test ({idioma}) - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")

# Funció per avaluar models de classificació
def avaluar_classificador(model, model_name, X_train, y_train, X_valid, y_valid, X_test, y_test, idioma):
    """
    Avaluar models de classificació amb mètriques de classificació (exactitud, matriu de confusió)
    """
    print(f"\n### Model: {model_name} ({idioma}) ###")
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Informació detallada
    print(f"Exactitud en test: {test_accuracy:.4f}")
    print("\nClassification Report (Test):")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    print(classification_report(y_test, y_test_pred))

    # Matriu de confusió
    print("Matriu de Confusió (Test):")
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.title(f"Matriu de Confusió: {model_name} ({idioma})")
    plt.xlabel("Predicció")
    plt.ylabel("Valor Real")
    plt.show()

# Main
if __name__ == "__main__":
    # Exemple de models per utilitzar
    regressors = [
        (LinearSVR(), "SVM Regressor"),
        #(RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), "RandomForest Regressor"),
        (XGBRegressor(n_estimators=100, random_state=42), "XGBoost Regressor")
    ]

    classificadors = [
        (MultinomialNB(), "MultinomialNB"),
        (ComplementNB(), "ComplementNB")
    ]

    # Avaluar per al dataset en espanyol amb regressors
    for model, model_name in regressors:
        avaluar_regressor(model, model_name, X_train_tfidf_es, y_train_es, X_valid_tfidf_es, y_valid_es, X_test_tfidf_es, y_test_es, "Espanyol")

    # Avaluar per al dataset en espanyol amb classificadors
    for model, model_name in classificadors:
        avaluar_classificador(model, model_name, X_train_tfidf_es, y_train_es, X_valid_tfidf_es, y_valid_es, X_test_tfidf_es, y_test_es, "Espanyol")

    # Avaluar per al dataset en anglès amb regressors
    for model, model_name in regressors:
        avaluar_regressor(model, model_name, X_train_tfidf_en, y_train_en, X_valid_tfidf_en, y_valid_en, X_test_tfidf_en, y_test_en, "Anglès")

    # Avaluar per al dataset en anglès amb classificadors
    for model, model_name in classificadors:
        avaluar_classificador(model, model_name, X_train_tfidf_en, y_train_en, X_valid_tfidf_en, y_valid_en, X_test_tfidf_en, y_test_en, "Anglès")
