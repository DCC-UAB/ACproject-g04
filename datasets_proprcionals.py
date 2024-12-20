import pandas as pd
from sklearn.model_selection import train_test_split
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import os

# Crear el directori 'comparacio_idiomes' dins de data
output_dir = os.path.join('data', 'comparacio_idiomes_proprocionals')
os.makedirs(output_dir, exist_ok=True)

# Descarregar recursos de NLTK si no estan disponibles
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')

# Funció per netejar el text
def clean_text(text):
    text = re.sub(r"[^\x20-\x7E]", "", text)  # Caràcters ASCII
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Només lletres i espais
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# **Pas 1: Neteja del dataset en espanyol**
print("Carregant i netejant 'ressenyes_es.csv'...")
df_es = pd.read_csv('data/ressenyes_es.csv')

# Deixar només les columnes 'review_body' i 'rating'
df_es = df_es[['review_body', 'rating']].rename(columns={'review_body': 'review_full', 'rating': 'rating_review'})

# Netejar els textos de la columna 'review_full'
df_es['review_full'] = df_es['review_full'].fillna("").astype(str).apply(clean_text)

# Guardar el dataset netejat
df_es_cleaned_path = 'data/ressenyes_es_cleaned.csv'
df_es.to_csv(df_es_cleaned_path, index=False)
print(f"Dataset en espanyol netejat guardat a {df_es_cleaned_path}")

# **Pas 2: Retallar el dataset en anglès segons la distribució del dataset espanyol**
print("Carregant i ajustant 'cleaned_dataset.csv' per igualar la distribució de puntuacions...")
df_en = pd.read_csv('data/cleaned_dataset.csv')

# Calcular la distribució de puntuacions al dataset en espanyol
es_distribution = df_es['rating_review'].value_counts(normalize=True)
print("Distribució de puntuacions en espanyol:")
print(es_distribution)

# Funció per seleccionar la mateixa proporció de puntuacions en anglès
def match_distribution(source_df, target_distribution):
    result_df = pd.DataFrame()
    for rating, proportion in target_distribution.items():
        # Trobar quantes files necessitem per aquesta puntuació
        n_rows = int(len(source_df) * proportion)
        subset = source_df[source_df['rating_review'] == rating]
        sampled_subset = subset.sample(n=min(n_rows, len(subset)), random_state=42)
        result_df = pd.concat([result_df, sampled_subset], ignore_index=True)
    return result_df

# Ajustar el dataset en anglès
df_en_trimmed = match_distribution(df_en, es_distribution)
df_en_trimmed_path = 'data/cleaned_dataset_trimmed.csv'
df_en_trimmed.to_csv(df_en_trimmed_path, index=False)
print(f"Dataset en anglès ajustat i guardat a {df_en_trimmed_path}")

# **Pas 3: Funció per dividir i desar els datasets**
def split_and_save(df, output_paths):
    """
    Divideix un dataset en entrenament, validació i test i guarda els resultats.
    Les columnes del fitxer de sortida seran 'description' i 'score'.
    """
    X = df['review_full']  # Text de les ressenyes
    y = df['rating_review']  # Puntuació

    # Dividir en 70% entrenament i 30% validació + test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    # Dividir 30% restant en 15% validació i 15% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp)

    # Crear DataFrames amb noms de columnes correctes
    train_df = pd.DataFrame({'description': X_train, 'score': y_train})
    val_df = pd.DataFrame({'description': X_val, 'score': y_val})
    test_df = pd.DataFrame({'description': X_test, 'score': y_test})

    # Guardar els fitxers
    train_df.to_csv(output_paths['train'], index=False)
    val_df.to_csv(output_paths['val'], index=False)
    test_df.to_csv(output_paths['test'], index=False)

    print(f"Conjunt d'entrenament: {len(X_train)} mostres")
    print(f"Conjunt de validació: {len(X_val)} mostres")
    print(f"Conjunt de test: {len(X_test)} mostres")
    print("Els subconjunts han estat desats amb èxit.")

# Definir els paths de sortida
output_paths_es = {
    'train': os.path.join(output_dir, 'ressenyes_es_train.csv'),
    'val': os.path.join(output_dir, 'ressenyes_es_val.csv'),
    'test': os.path.join(output_dir, 'ressenyes_es_test.csv')
}

output_paths_en = {
    'train': os.path.join(output_dir, 'ressenyes_en_train.csv'),
    'val': os.path.join(output_dir, 'ressenyes_en_val.csv'),
    'test': os.path.join(output_dir, 'ressenyes_en_test.csv')
}

# **Pas 4: Dividir i desar els datasets**
print("Dividint el dataset en espanyol...")
split_and_save(df_es, output_paths_es)

print("Dividint el dataset en anglès ajustat...")
split_and_save(df_en_trimmed, output_paths_en)

print("Procés complet!")
