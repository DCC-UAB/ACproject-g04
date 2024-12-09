import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
print("hello1")
train = pd.read_csv("data/train_set.csv", encoding='utf-8')
validate = pd.read_csv("data/val_set.csv", encoding='utf-8')
test = pd.read_csv("data/test_set.csv", encoding='utf-8')
#print(train.head())
#print(validate.head())
#print(test.head())
print("hello2")
# Buscar valors problemàtics
#print("Valores únicos en 'description':")
#print(train['description'].unique())



#vectorizer = CountVectorizer()
vectorizer = CountVectorizer(max_features=10000)

# Usar una muestra del dataset
#sample_train = train.sample(frac=0.5, random_state=42)  # 10% del dataset
#X_sample = vectorizer.fit_transform(sample_train['description'])
#y_sample = sample_train['score']


x_train = vectorizer.fit_transform(train['description'].fillna(''))
x_val = vectorizer.transform(validate['description'].fillna(''))
x_test = vectorizer.transform(test['description'].fillna(''))
print("hello3")
y_train = train['score']
y_val = validate['score']
y_test = test['score']
print("hello4")
model = RandomForestClassifier(n_jobs=2, random_state=42)

model.fit(x_train, y_train)
#model.fit(X_sample, y_sample)
print("hello5")
y_pred_validate = model.predict(x_val)
accuracy_validate = accuracy_score(y_val, y_pred_validate)
print(f'Precisión en validación: {accuracy_validate:.2f}')
print("hello6")

y_pred_test = model.predict(x_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f'Precisión en prueba: {accuracy_test:.2f}')


