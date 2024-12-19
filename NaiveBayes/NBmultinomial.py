import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Vectorització amb CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['description'])
y_train = train_data['score']

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


#PREGUNTA --> EQUILIBRI DE LES DADES
print(train_data['score'].value_counts())