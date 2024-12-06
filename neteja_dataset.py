import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')


data_path = "data/TripAdvisor_reviews.csv"  
df = pd.read_csv(data_path)
print(df.head())

def clean_text(text):
    # Eliminar caracters no ASCII (mantenir només lletres)
    text = re.sub(r"[^\x20-\x7E]", "", text)  # Caràcters ASCII imprimibles
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Mantenir només lletres i espais
    
    # Convertir a minúscules
    text = text.lower()
    
    # Tokenitzar el text
    tokens = word_tokenize(text)
    
    # Eliminar stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lematitzar els tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Reconstruir el text net
    cleaned_text = " ".join(tokens)
    return cleaned_text

# Aplicar la funció de neteja a la columna de ressenyes
df["cleaned_review"] = df["review_full"].apply(clean_text)

# Guardar el dataset netejat en un nou fitxer CSV (opcional)
output_path = "data/cleaned_dataset.csv"
df.to_csv(output_path, index=False)

# Mostrar un exemple de les dades netejades
print(df.head())
