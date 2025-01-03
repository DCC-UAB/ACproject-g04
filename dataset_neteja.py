import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
import matplotlib.pyplot as plt

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

# Funció per mostrar les paraules més comunes
def show_most_common_words(texts, title, num_words=10):
    all_words = " ".join(texts).split()
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(num_words)

    # Mostrar gràfic
    words, counts = zip(*common_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts, color='skyblue')
    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.show()

# Carregar el dataset
data_path = "data/TripAdvisor_reviews.csv"  
df = pd.read_csv(data_path)
print("Dataset original:")
print(df.head())

# Pas 1: Eliminar caràcters no ASCII i mantenir només lletres
def remove_non_ascii_and_keep_letters(text):
    return re.sub(r"[^a-zA-Z\s]", "", text)

df["step1_cleaned"] = df["review_full"].fillna("").astype(str).apply(remove_non_ascii_and_keep_letters)
print("\nDesprés de netejar caràcters no ASCII i mantenir només lletres:")
show_most_common_words(df["step1_cleaned"], "Pas 1: Words after removing non-ASCII characters")

# Pas 2: Convertir a minúscules
df["step2_cleaned"] = df["step1_cleaned"].str.lower()
print("\nDesprés de convertir a minúscules:")
show_most_common_words(df["step2_cleaned"], "Pas 2: Words after converting to lowercase")

# Pas 3: Tokenitzar el text
def tokenize_text(text):
    return " ".join(word_tokenize(text))

df["step3_cleaned"] = df["step2_cleaned"].apply(tokenize_text)
print("\nDesprés de tokenitzar:")
show_most_common_words(df["step3_cleaned"], "Pas 3: Words after tokenization")

# Pas 4: Eliminar stop words
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

df["step4_cleaned"] = df["step3_cleaned"].apply(remove_stop_words)
print("\nDesprés d'eliminar stop words:")
show_most_common_words(df["step4_cleaned"], "Pas 4: Words after removing stopwords")

# Pas 5: Lematitzar els tokens
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_tokens)

df["step5_cleaned"] = df["step4_cleaned"].apply(lemmatize_text)
print("\nDesprés de lematitzar:")
show_most_common_words(df["step5_cleaned"], "Pas 5: Words after lemmatization")

# Substituir la columna original amb el text netejat
df["review_full"] = df["step5_cleaned"]

# Guardar el dataset netejat
output_path = "data/cleaned_dataset.csv"
df.to_csv(output_path, index=False, columns=["rating_review", "review_full"])

print("\nExemple de dades netejades:")
print(df[["rating_review", "review_full"]].head())
