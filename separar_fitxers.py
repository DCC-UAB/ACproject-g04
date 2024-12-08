import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar les dades des del fitxer CSV
df = pd.read_csv("data/cleaned_dataset.csv")

# ENs quedarà la columna de puntuació amb nom 'rating_review' i la de descripció com 'review_full'
X = df['review_full']
y = df['rating_review']


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
train_df.to_csv("data/train_set.csv", index=False)
val_df.to_csv("data/val_set.csv", index=False)
test_df.to_csv("data/test_set.csv", index=False)

print("Els subconjunts han estat desats amb èxit.")
