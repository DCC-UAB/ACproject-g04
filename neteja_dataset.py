import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Descarregar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Carregar el dataset
df = pd.read_csv("data/TripAdvisor_reviews.csv")

# Inicialitzar el lematitzador i la llista de paraules buides (stop words)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Assegurar-nos que el text sigui una cadena (string)
    if not isinstance(text, str):
        text = str(text)  # Convertir a cadena si no ho és
    
    # Eliminar caràcters no ASCII, comes, punts i números
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Eliminar caràcters no ASCII
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar signes de puntuació
    text = re.sub(r'\d+', '', text)  # Eliminar números
    
    # Convertir a minúscules
    text = text.lower()
    
    # Tokenitzar el text
    tokens = word_tokenize(text)
    
    # Eliminar les paraules buides (stop words) i lematitzar
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Unir les paraules per formar el text net
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text

# Assegurar-nos que no hi hagi valors nuls a la columna 'review_full'
df['review_full'] = df['review_full'].fillna('')

# Aplicar la funció de neteja a la columna 'review_full'
df['review_full'] = df['review_full'].apply(clean_text)

# Guardar el dataset net en un nou fitxer CSV
df.to_csv("data/clean_dataset.csv", columns=['rating_review', 'review_full'], index=False)

print("Neteja completada i fitxer CSV guardat com 'dataset_limpio.csv'")
