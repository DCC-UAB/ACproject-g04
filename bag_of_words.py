import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier

print("llibreries importades ok")

#original
#train = pd.read_csv("data/train_set.csv", encoding='utf-8')
#validate = pd.read_csv("data/val_set.csv", encoding='utf-8')
#test = pd.read_csv("data/test_set.csv", encoding='utf-8')

#COMPARAR IDIOMES
#español 
#train = pd.read_csv("data/comparacio_idiomes/ressenyes_es_train.csv", encoding='utf-8')
#validate = pd.read_csv("data/comparacio_idiomes/ressenyes_es_val.csv", encoding='utf-8')
#test = pd.read_csv("data/comparacio_idiomes/ressenyes_es_test.csv", encoding='utf-8')

#inglés
train = pd.read_csv("data/comparacio_idiomes/ressenyes_en_train.csv", encoding='utf-8')
validate = pd.read_csv("data/comparacio_idiomes/ressenyes_en_val.csv", encoding='utf-8')
test = pd.read_csv("data/comparacio_idiomes/ressenyes_en_test.csv", encoding='utf-8')


#score binari
#train = train[train['score'] != 3]
#validate = validate[validate['score'] != 3]
#test = test[test['score'] != 3]

#print(train.head())
#print(validate.head())
#print(test.head())
# Buscar valors problemàtics
#print("Valores únicos en 'description':")
#print(train['description'].unique())
print("càrrega arxius ok")

# FREQÜÈNCIA BINÀRIA
#vectorizer = CountVectorizer()
#vectorizer = CountVectorizer(max_features=10000)
#vectorizer = CountVectorizer(max_features=100)
#vectorizer = CountVectorizer(min_df=0.005)  # Manté les paraules que apareixin almenys en el 5% dels docs

#FREQÜÈNCIA TF-IDF
vectorizer = TfidfVectorizer(min_df=0.005)

# Usar una muestra del dataset
#sample_train = train.sample(frac=0.5, random_state=42)  # 10% del dataset
#X_sample = vectorizer.fit_transform(sample_train['description'])
#y_sample = sample_train['score']
x_train = vectorizer.fit_transform(train['description'].fillna(''))
x_val = vectorizer.transform(validate['description'].fillna(''))
x_test = vectorizer.transform(test['description'].fillna(''))

num_features = x_train.shape[1]
print(f'Número de característiques (features): {num_features}')

print("x_train,val,test ok")


# Mostrar las primeras 20 palabras
print(vectorizer.get_feature_names_out()[:20])

# Convertir la matriu dispersa a una matriu densa
#x_dense = x_train[:100, :100].toarray()
# Crear un DataFrame de pandas amb les paraules com columnes
#df_preview = pd.DataFrame(x_dense, columns=vectorizer.get_feature_names_out()[:100])
# Guardar el DataFrame en un arxiu CSV
#df_preview.to_csv('bag_of_words_matrix.csv', index=False)
#print("Matriu guardada como 'bag_of_words_matrix.csv'")


y_train = train['score']
y_val = validate['score']
y_test = test['score']

#PER FER SCORE 0-1
#y_train = y_train.replace({1: 0, 2: 0, 4: 1, 5: 1})
#y_val = y_val.replace({1: 0, 2: 0, 4: 1, 5: 1})
#y_test = y_test.replace({1: 0, 2: 0, 4: 1, 5: 1})


print("y_train,val,test ok")

#apliquem SMOTE per sobre-mostrant(generar mostres sintètiques)
#smote = SMOTE(random_state=42)
#x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

#apliquem per fer un sota-mostrant(reduir motres majoritàries)
#rus = RandomUnderSampler(random_state=42)
#x_train_balanced, y_train_balanced = rus.fit_resample(x_train, y_train)

#MODEL RANDOM FOREST
model = RandomForestClassifier(n_jobs=-1,random_state=42)
#model = RandomForestClassifier(n_jobs=-1,random_state=42, class_weight='balanced')
model.fit(x_train, y_train)
#model.fit(X_sample, y_sample)
#model.fit(x_train_balanced, y_train_balanced)

#MODEL NAIVE BAYES
#model = MultinomialNB() 
#model.fit(x_train, y_train)

#MODEL XGBoost
#model =XGBClassifier(objective='multi:softmax', num_class=5, eval_metric='mlogloss', random_state=42)
# Reindexar classes per començar a partir de 0
#y_train = y_train - 1
#y_val = y_val - 1
#y_test = y_test - 1
#Reindexar per binari
#y_train = y_train - y_train.min()  
#y_val = y_val - y_val.min()
#y_test = y_test - y_test.min()
#model.fit(x_train, y_train)

print("model executat ok")


y_pred_validate = model.predict(x_val)
accuracy_validate = accuracy_score(y_val, y_pred_validate)
print(f'Exactitud del model en validació: {accuracy_validate:.2f}')

y_pred_test = model.predict(x_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f'Exactitud del model en test: {accuracy_test:.2f}')

print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test))

print("Matriu de confusió (Test)")
print(confusion_matrix(y_test, y_pred_test))


