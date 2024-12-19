import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Descarregar les dades necessàries per NLTK
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

# Carregar el dataset
data_path = "data/TripAdvisor_reviews.csv"  
df = pd.read_csv(data_path)
print(df.head())

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
    
    # Lematitzar els tokens --> TRANSFORMAR EN L'ARREL
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Reconstruir el text net
    cleaned_text = " ".join(tokens)
    return cleaned_text

# Substituir valors no vàlids (NaN) per una cadena buida
df["review_full"] = df["review_full"].fillna("").astype(str)

# Aplicar la funció de neteja a la columna de ressenyes
df["cleaned_review"] = df["review_full"].apply(clean_text)

# Renombrar la columna 'cleaned_review' a 'review_full' per substituir el contingut original
df["review_full"] = df["cleaned_review"]

# Eliminar la columna 'cleaned_review' (ja no és necessària)
df.drop(columns=["cleaned_review"], inplace=True)

# Guardar el dataset netejat en un nou fitxer CSV (només amb les columnes 'rating_review' i 'review_full')
output_path = "data/cleaned_dataset.csv"
df.to_csv(output_path, index=False, columns=["rating_review", "review_full"])

# Mostrar un exemple de les dades netejades
print(df.head())