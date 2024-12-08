import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar les dades des del fitxer CSV
<<<<<<< HEAD
#df = pd.read_csv("C:/Users/smart/personal/uni/enginyeria de dades/3r curs/1r semestre/aprenentatge computacional/proj/TripAdvisor_reviews.csv")
df = pd.read_csv("C:/Users/jpall/OneDrive/Documentos/TERCER CURS - 1r semestre/Aprenentatge Computacional/projecte/ACproject-g04/data")
=======
df = pd.read_csv("data/cleaned_dataset.csv")
>>>>>>> 2e885c55660d6f4f5271cc907c897b47c9ee9e64

# Suposem que la columna de puntuació és 'score' i la descripció és 'description'
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

# Guardar els subconjunts en fitxers CSV
train_df.to_csv("data/train_set.csv", index=False)
val_df.to_csv("data/val_set.csv", index=False)
test_df.to_csv("data/test_set.csv", index=False)

print("Els subconjunts han estat desats amb èxit.")
