import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample 

# Carregar datasets
train_data = pd.read_csv('./data/train_set.csv')
val_data = pd.read_csv('./data/val_set.csv')
test_data = pd.read_csv('./data/test_set.csv')

# Substituir NaN per cadenes buides
train_data['description'] = train_data['description'].fillna('')
val_data['description'] = val_data['description'].fillna('')
test_data['description'] = test_data['description'].fillna('')

# Convertir a tipus string (per si hi ha valors no textuals)
train_data['description'] = train_data['description'].astype(str)
val_data['description'] = val_data['description'].astype(str)
test_data['description'] = test_data['description'].astype(str)

# Submostreig per equilibrar les classes (ajustant la mida de les classes majoritàries a la menor classe)
min_class_size = min(train_data['score'].value_counts())  # Troba la mida de la classe menor

# Combina X i y per submostreig
train_data_balanced = pd.concat([train_data['description'], train_data['score']], axis=1)

classes = []
for score in train_data['score'].unique():
    class_data = train_data_balanced[train_data_balanced['score'] == score]
    class_downsampled = resample(class_data, 
                                 replace=False,  # Sense reemplaçament
                                 n_samples=min_class_size,  # Equilibra amb la classe més petita
                                 random_state=42)
    classes.append(class_downsampled)

# Combina les classes balancejades
balanced_train_data = pd.concat(classes)

# Mostra la distribució de les classes després del submostreig
print("Distribució de classes després del submostreig:")
print(balanced_train_data['score'].value_counts())

# Vectorització amb CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(balanced_train_data['description'])  # Usa el dataset balancejat
y_train = balanced_train_data['score']

X_val = vectorizer.transform(val_data['description'])
y_val = val_data['score']

X_test = vectorizer.transform(test_data['description'])
y_test = test_data['score']

# Verificar que X_train i y_train tenen el mateix nombre de mostres
print(f'Número de mostres en X_train: {X_train.shape[0]}')
print(f'Número de mostres en y_train: {y_train.shape[0]}')

# Entrenament del model
model = MultinomialNB()
model.fit(X_train, y_train)

# Validació
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Exactitud del model en validació: {val_accuracy:.4f}")

# Test
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Exactitud del model en test: {test_accuracy:.4f}")

# Informes finals
print("Informe de classificació (validació):")
print(classification_report(y_val, y_val_pred))

print("Informe de classificació (test):")
print(classification_report(y_test, y_test_pred))

print("Matriu de Confusió (Test):")
print(confusion_matrix(y_test, y_test_pred))

# Preguntar sobre la distribució de classes abans i després del submostreig
print("Distribució original de les classes (abans del submostreig):")
print(train_data['score'].value_counts())
