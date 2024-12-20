import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVR
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                             confusion_matrix, accuracy_score, classification_report)
import seaborn as sns
import matplotlib.pyplot as plt

# Funció per carregar i preparar les dades
def prepara_dades(data_dir, vectorizer_type="tfidf", max_features=5000):
    """
    Carrega, neteja i vectoritza les dades.
    """
    # Carregar dades
    train_data = pd.read_csv(data_dir + "train_set.csv")
    val_data = pd.read_csv(data_dir + "val_set.csv")
    test_data = pd.read_csv(data_dir + "test_set.csv")

    # Substituir NaN per cadenes buides i convertir a string
    for dataset in [train_data, val_data, test_data]:
        dataset['description'] = dataset['description'].fillna('').astype(str)

    # Etiquetes
    y_train = train_data['score']
    y_val = val_data['score']
    y_test = test_data['score']

    # Vectorització
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features)
    elif vectorizer_type == "count":
        vectorizer = CountVectorizer(max_features=max_features)
    else:
        raise ValueError("vectorizer_type ha de ser 'tfidf' o 'count'.")

    X_train = vectorizer.fit_transform(train_data['description'])
    X_val = vectorizer.transform(val_data['description'])
    X_test = vectorizer.transform(test_data['description'])

    return X_train, y_train, X_val, y_val, X_test, y_test, vectorizer

# Funció per entrenar i avaluar regressors
def entrenar_avaluar_regressor(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Entrena i avalua un model de regressió.
    """
    print(f"\n### Model: {model_name} ###")

    # Entrenar model
    model.fit(X_train, y_train)

    # Prediccions
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Mètriques de regressió
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"Validació - MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}")

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Test - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")

# Funció per entrenar i avaluar classificadors amb matriu de confusió
def entrenar_avaluar_classificador(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Entrena i avalua un model de classificació.
    """
    print(f"\n### Model: {model_name} ###")
    
    # Entrenar model
    model.fit(X_train, y_train)

    # Prediccions
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Informació detallada
    print(f"Exactitud en test: {test_accuracy:.4f}")
    print("\nClassification Report (Test):")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    print(classification_report(y_test, y_test_pred))

    # Mostrar mètriques específiques
    print(f"Precision: {report['weighted avg']['precision']:.4f}")
    print(f"Recall: {report['weighted avg']['recall']:.4f}")
    print(f"F1-Score: {report['weighted avg']['f1-score']:.4f}")

    # Matriu de confusió
    print("Matriu de Confusió (Test):")
    unique_labels = sorted(set(y_test))
    plot_confusion_matrix(y_test, y_test_pred, unique_labels, model_name)

# Funció per plotejar la matriu de confusió
def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    """
    Plotejar la matriu de confusió amb números i percentatges.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Anotacions combinant nombres i percentatges
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.2f}%)"

    # Plotejar
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=annotations, fmt="", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Matriu de Confusió: {model_name}")
    plt.ylabel("Valor Real")
    plt.xlabel("Predicció")
    plt.show()

# Main
if __name__ == "__main__":
    data_dir = "data/"

    # Preparar dades
    X_train, y_train, X_val, y_val, X_test, y_test, vectorizer = prepara_dades(data_dir, vectorizer_type="tfidf")

    # REGRESSORS
    regressors = [
        (LinearSVR(), "SVM Regressor"),
        #(RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), "RandomForest Regressor"),
        (XGBRegressor(n_estimators=100, random_state=42), "XGBoost Regressor")
    ]
    for model, model_name in regressors:
        entrenar_avaluar_regressor(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test)

    # CLASSIFICADORS
    classificadors = [
        (MultinomialNB(), "MultinomialNB"),
        (ComplementNB(), "ComplementNB")
    ]
    for model, model_name in classificadors:
        entrenar_avaluar_classificador(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test)
