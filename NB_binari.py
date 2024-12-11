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

# Transformar les classes en dues classes: Negativa (1+2) i Positiva (4+5)
train_data['score'] = train_data['score'].apply(lambda x: 'Negativa' if x in [1, 2] else ('Positiva' if x in [4, 5] else 'Neutral'))
val_data['score'] = val_data['score'].apply(lambda x: 'Negativa' if x in [1, 2] else ('Positiva' if x in [4, 5] else 'Neutral'))
test_data['score'] = test_data['score'].apply(lambda x: 'Negativa' if x in [1, 2] else ('Positiva' if x in [4, 5] else 'Neutral'))

# Eliminar la classe "Neutral" (classe 3) per a una classificació binària
train_data = train_data[train_data['score'] != 'Neutral']
val_data = val_data[val_data['score'] != 'Neutral']
test_data = test_data[test_data['score'] != 'Neutral']

# Reajustar el submostreig o sobremostreig per les classes "Negativa" i "Positiva"
max_class_size = max(train_data['score'].value_counts())  # Mida de la classe majoritària

# Combina X i y per submostreig
train_data_balanced = pd.concat([train_data['description'], train_data['score']], axis=1)

classes = []
for score in train_data['score'].unique():
    class_data = train_data_balanced[train_data_balanced['score'] == score]
    class_upsampled = resample(class_data, 
                               replace=True,  # Amb reemplaçament
                               n_samples=max_class_size,  # Equilibra les classes
                               random_state=42)
    classes.append(class_upsampled)

# Combina les classes balancejades
balanced_train_data = pd.concat(classes)

# Mostra la distribució de les classes després del sobremostreig
print("Distribució de classes després del sobremostreig:")
print(balanced_train_data['score'].value_counts())

# Vectorització amb CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(balanced_train_data['description'])  # Usa el dataset balancejat
y_train = balanced_train_data['score']

X_val = vectorizer.transform(val_data['description'])
y_val = val_data['score']

X_test = vectorizer.transform(test_data['description'])
y_test = test_data['score']

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

# Preguntar sobre la distribució de classes abans i després del sobremostreig
print("Distribució original de les classes (abans del sobreajustament):")
print(train_data['score'].value_counts())
