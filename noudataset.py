import pandas as pd
from sklearn.model_selection import train_test_split
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
import nltk

# Descarregar les dades necessàries per NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/omw-1.4')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('words')


# Carregar el dataset
df = pd.read_csv("data/Hotel_Reviews.csv")

# Crear la columna 'Description' combinant les ressenyes positives i negatives
df['Description'] = df.apply(
    lambda row: f"{row['Negative_Review']} {row['Positive_Review']}" 
    if row['Negative_Review'] != 'No Negative' and row['Positive_Review'] != 'No Positive'
    else row['Negative_Review'] if row['Negative_Review'] != 'No Negative'
    else row['Positive_Review'] if row['Positive_Review'] != 'No Positive' else '',
    axis=1
)

# Eliminar les files que no tinguin descripció vàlida (quan les dues ressenyes són 'No Negative' i 'No Positive')
df = df[df['Description'] != '']

# Dividir el 'Reviewer_Score' per 2 per ajustar el rang a sobre 5 (els valors originals estan entre 0 i 10)
df['Reviewer_Score'] = df['Reviewer_Score'] / 2

# Arrodonir el 'Reviewer_Score' a un enter
df['Reviewer_Score'] = df['Reviewer_Score'].round().astype(int)

# Filtrar només les columnes de "Reviewer_Score" i "Description"
df_filtered = df[['Reviewer_Score', 'Description']]

# Limitar a un màxim de 500.000 files
df_filtered = df_filtered.head(500000)

print("DATASET PREPARAT PER NETEJAR")

#NETEJAR
df = df_filtered
# Obtenir el vocabulari en anglès de NLTK
english_vocab = set(words.words())

# Funció de neteja del text
def clean_text(text):
    # Eliminar caracters no ASCII (mantenir només lletres)
    text = re.sub(r"[^\x20-\x7E]", "", text)  # Caràcters ASCII imprimibles
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Mantenir només lletres i espais
    
    # Convertir a minúscules
    text = text.lower()
    
    # Tokenitzar el text --> UNITATS MÉS PETITES, SEPARA FRASES EN PARAULES, 
    tokens = word_tokenize(text)
    
    # Eliminar stop words --> SUPRIMIR PARAULES MOLT COMUNES (THE, IS, ETC)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Filtrar paraules que no estan al vocabulari anglès
    tokens = [word for word in tokens if word in english_vocab]
    
    # Lematitzar els tokens --> TRANSFORMAR EN L'ARREL
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Reconstruir el text net
    cleaned_text = " ".join(tokens)
    return cleaned_text

# Substituir valors no vàlids (NaN) per una cadena buida
df["Description"] = df["Description"].fillna("").astype(str)

# Aplicar la funció de neteja a la columna de ressenyes
df["cleaned_review"] = df["Description"].apply(clean_text)

# Renombrar la columna 'cleaned_review' a 'review_full' per substituir el contingut original
df["Description"] = df["cleaned_review"]

# Eliminar la columna 'cleaned_review' (ja no és necessària)
df.drop(columns=["cleaned_review"], inplace=True)

# Guardar el dataset netejat en un nou fitxer CSV (només amb les columnes 'rating_review' i 'review_full')
output_path = "data/resultat_filtratPROVES.csv"
df.to_csv(output_path, index=False, columns=["Reviewer_Score", "Description"])

# Mostrar un exemple de les dades netejades
print("DATASET NETEJAT")
print("Primeres files nou dataframe")
print(df.head())

# Comptar les ressenyes per cada puntuació
score_counts = df['Reviewer_Score'].value_counts().sort_index()

# Mostrar les comptabilitats
print("Comptabilitat de les ressenyes per puntuació:")
print(score_counts)


#SEPARAR EN TRAIN-TEST-VALIDATE
# Carregar les dades des del fitxer CSV
df = pd.read_csv("data/resultat_filtratPROVES.csv")

# ENs quedarà la columna de puntuació amb nom 'rating_review' i la de descripció com 'review_full'
X = df['Description']
y = df['Reviewer_Score']

# Dividim les dades en 70% entrenament i 30% per a validació + test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)

# Dividim les dades restants (30%) en 15% validació i 15% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

# Mostrem els resultats de la divisió
print(f"Conjunt d'entrenament: {len(X_train)} mostres")
print(f"Conjunt de validació: {len(X_val)} mostres")
print(f"Conjunt de test: {len(X_test)} mostres")

# Crear dataframes per a cada subconjunt
train_df = pd.DataFrame({'description': X_train, 'score': y_train})
val_df = pd.DataFrame({'description': X_val, 'score': y_val})
test_df = pd.DataFrame({'description': X_test, 'score': y_test})

# Guardar els subconjunts en fitxers CSV, a la carpeta data, (i no es pujaran al git)
train_df.to_csv("data/train_setPROVES.csv", index=False)
val_df.to_csv("data/val_setPROVES.csv", index=False)
test_df.to_csv("data/test_setPROVES.csv", index=False)

print("Els subconjunts han estat desats amb èxit.")





