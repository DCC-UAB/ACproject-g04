import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# 1. Carregar els datasets
data_dir = "data/"
train_df = pd.read_csv(data_dir + "train_set.csv")
val_df = pd.read_csv(data_dir + "val_set.csv")
test_df = pd.read_csv(data_dir + "test_set.csv")

# Assegurar-nos que no hi ha valors nuls i convertir a cadenes de text
for df in [train_df, val_df, test_df]:
    df['description'] = df['description'].fillna("").astype(str)

print("Datasets carregats i preparats correctament.")

# 2. Funció per obtenir n-grams més comuns
def get_top_ngrams(corpus, n=2, top_k=10):
    """
    Obté els n-grams més comuns d'un corpus.

    Args:
        corpus (iterable): Llista o sèrie de textos.
        n (int): Longitud dels n-grams (2 per bigrames, 3 per trigrames, etc.).
        top_k (int): Nombre màxim d'n-grams més comuns a retornar.

    Returns:
        list: Llista de tuples amb els n-grams i les seves freqüències.
    """
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')  # Exclou stopwords
    ngram_counts = vectorizer.fit_transform(corpus)
    ngram_sums = ngram_counts.sum(axis=0)
    ngram_freq = [(word, ngram_sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    ngram_freq = sorted(ngram_freq, key=lambda x: x[1], reverse=True)
    return ngram_freq[:top_k]

# 3. Separar reviews positives (4 i 5) i negatives (1 i 2)
def filter_reviews(df):
    """
    Separa les reviews en positives (4, 5) i negatives (1, 2).
    """
    positive_reviews = df[df['score'].isin([4, 5])]['description']
    negative_reviews = df[df['score'].isin([1, 2])]['description']
    return positive_reviews, negative_reviews

train_pos, train_neg = filter_reviews(train_df)
val_pos, val_neg = filter_reviews(val_df)
test_pos, test_neg = filter_reviews(test_df)

# 4. Obtenir bigrames i trigrames
def print_ngrams(name, corpus, n, top_k=10):
    """
    Obté i imprimeix els n-grams més comuns per a un corpus donat.
    """
    print(f"\n{name} - {n}-grams més comuns:")
    ngrams = get_top_ngrams(corpus, n=n, top_k=top_k)
    for gram, freq in ngrams:
        print(f"{gram}: {freq}")

# Analitzar bigrames i trigrames en reviews positives i negatives
print("\n### TRAIN SET ###")
print_ngrams("Positives (Train)", train_pos, n=2)
print_ngrams("Negatives (Train)", train_neg, n=2)
print_ngrams("Positives (Train)", train_pos, n=3)
print_ngrams("Negatives (Train)", train_neg, n=3)

print("\n### VALIDATION SET ###")
print_ngrams("Positives (Validation)", val_pos, n=2)
print_ngrams("Negatives (Validation)", val_neg, n=2)
print_ngrams("Positives (Validation)", val_pos, n=3)
print_ngrams("Negatives (Validation)", val_neg, n=3)

print("\n### TEST SET ###")
print_ngrams("Positives (Test)", test_pos, n=2)
print_ngrams("Negatives (Test)", test_neg, n=2)
print_ngrams("Positives (Test)", test_pos, n=3)
print_ngrams("Negatives (Test)", test_neg, n=3)
