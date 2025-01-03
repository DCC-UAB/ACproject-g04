import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVR
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                             confusion_matrix, accuracy_score, classification_report, roc_curve, auc, roc_auc_score)
import seaborn as sns
import matplotlib.pyplot as plt

# Funció per carregar i preparar les dades amb classificació binària
def prepara_dades_binari(data_dir, vectorizer_type="tfidf", max_features=5000, transformacio="versio1"):
    """
    Carrega, neteja i vectoritza les dades per a classificació binària.
    """
    # Carregar dades
    train_data = pd.read_csv(data_dir + "train_set.csv")
    val_data = pd.read_csv(data_dir + "val_set.csv")
    test_data = pd.read_csv(data_dir + "test_set.csv")

    # Substituir NaN per cadenes buides i convertir a string
    for dataset in [train_data, val_data, test_data]:
        dataset['description'] = dataset['description'].fillna('').astype(str)

    # Transformació binària de les etiquetes
    if transformacio == "versio1":
        train_data['score'] = train_data['score'].apply(lambda x: 0 if x in [1, 2] else 1)
        val_data['score'] = val_data['score'].apply(lambda x: 0 if x in [1, 2] else 1)
        test_data['score'] = test_data['score'].apply(lambda x: 0 if x in [1, 2] else 1)
    elif transformacio == "versio2":
        train_data['score'] = train_data['score'].apply(lambda x: 0 if x == 4 else 1 if x == 5 else np.nan)
        val_data['score'] = val_data['score'].apply(lambda x: 0 if x == 4 else 1 if x == 5 else np.nan)
        test_data['score'] = test_data['score'].apply(lambda x: 0 if x == 4 else 1 if x == 5 else np.nan)
        train_data = train_data.dropna()
        val_data = val_data.dropna()
        test_data = test_data.dropna()
    else:
        raise ValueError("transformacio ha de ser 'versio1' o 'versio2'.")

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

    # Curva ROC per a la classificació binària
    # Aquí tractem les prediccions com a probabilitats per a la classe positiva (1)
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(X_test)[:, 1]  # Probabilitats per a la classe 1
    elif hasattr(model, "decision_function"):
        y_test_prob = model.decision_function(X_test)  # Si el model té 'decision_function'
    else:
        y_test_prob = y_test_pred  # Si no tenim probabilitats, utilitzem les prediccions

    # Calcular la ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)

    # Plotejar la ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal de la classificació aleatòria
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positius (FPR)')
    plt.ylabel('Taxa de Veritables Positius (TPR)')
    plt.title(f'Curva ROC: {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# Funció per entrenar i avaluar classificadors amb matriu de confusió
def entrenar_avaluar_classificador(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, transformacio):
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
    plot_confusion_matrix(y_test, y_test_pred, unique_labels, model_name, transformacio)

    # Curva ROC per a la classificació binària
    # Aquí tractem les prediccions com a probabilitats per a la classe positiva (1)
    if hasattr(model, "predict_proba"):
        y_test_prob = model.predict_proba(X_test)[:, 1]  # Probabilitats per a la classe 1
    elif hasattr(model, "decision_function"):
        y_test_prob = model.decision_function(X_test)  # Si el model té 'decision_function'
    else:
        y_test_prob = y_test_pred  # Si no tenim probabilitats, utilitzem les prediccions

    # Calcular la ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)

    # Plotejar la ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal de la classificació aleatòria
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positius (FPR)')
    plt.ylabel('Taxa de Veritables Positius (TPR)')
    plt.title(f'Curva ROC: {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# Funció per plotejar la matriu de confusió
def plot_confusion_matrix(y_true, y_pred, labels, model_name, transformacio):
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
    versio = "Classificació Binària (1-2=0, 4-5=1)" if transformacio =="versio1" else "Classificació Binària (4=0, 5=1)"
    plt.title(f"Matriu de Confusió: {model_name}, {versio}")
    plt.ylabel("Valor Real")
    plt.xlabel("Predicció")
    plt.show()

# Main
if __name__ == "__main__":
    data_dir = "data/"

    # REGRESSORS
    regressors = [
        (LinearSVR(), "SVM Regressor"),
        (RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), "RandomForest Regressor"),
        (XGBRegressor(n_estimators=100, random_state=42), "XGBoost Regressor")
    ]

    # CLASSIFICADORS
    classificadors = [
        (MultinomialNB(), "MultinomialNB")
        (ComplementNB(), "ComplementNB")
    ]

    # Versió 1: 1 i 2 --> 0, 4 i 5 --> 1
    print("### Versió 1: Classificació Binària (1-2=0, 4-5=1) ###")
    X_train, y_train, X_val, y_val, X_test, y_test, vectorizer = prepara_dades_binari(data_dir, transformacio="versio1")

    for model, model_name in regressors:
        entrenar_avaluar_regressor(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test)

    for model, model_name in classificadors:
        entrenar_avaluar_classificador(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, "versio1")


    # Versió 2: 4 --> 0, 5 --> 1
    print("\n### Versió 2: Classificació Binària (4=0, 5=1) ###")
    X_train, y_train, X_val, y_val, X_test, y_test, vectorizer = prepara_dades_binari(data_dir, transformacio="versio2")

    for model, model_name in regressors:
        entrenar_avaluar_regressor(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test)

    for model, model_name in classificadors:
        entrenar_avaluar_classificador(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test, "versio2")

