import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

# Preparación de los datos
class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words("english"))
        self.punctuation = list(string.punctuation)

    def preprocess(self, text):
        """Preprocess the text: tokenize, remove punctuation, stopwords, and stem."""
        for c in self.punctuation:
            text = text.replace(c, " ")
        text = text.replace("\t", " ").replace("\n", " ")
        tokens = text.split()
        return [self.stemmer.stem(w.lower()) for w in tokens if w.lower() not in self.stopwords]


# Cargar el dataset CSV
df = pd.read_csv('/datasets/spam_ham_dataset.csv/spam_ham_dataset.csv')

# Verifica cómo está estructurado el dataset
print(df.head())

# Preprocesar los datos: Asumimos que el dataset tiene las columnas 'text' (contenido del correo) y 'label' (spam o ham)
X = df['text'].values
y = df['label'].map({'spam': 1, 'ham': 0}).values  # Convertir 'spam' a 1 y 'ham' a 0

# Preprocesar el texto
preprocessor = TextPreprocessor()
X = [' '.join(preprocessor.preprocess(text)) for text in X]

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Vectorización del texto
vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Creación de la regresión logística
clf = LogisticRegression(solver="saga", penalty="l1", max_iter=100, C=1.0, random_state=42)
clf.fit(X_train, y_train)

# Validación cruzada
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
print(f"Cross-validation scores: {scores}")
print(f"Average Cross-validation accuracy: {scores.mean():.3f}")

# Evaluación en el conjunto de prueba
y_pred = clf.predict(X_test)
print("\nPrediction:", y_pred)
print("\nTrue labels:", y_test)

# Cálculo de la precisión
print(f"Accuracy on test set: {accuracy_score(y_test, y_pred):.3f}")

# Predicciones con el modelo
new_emails = [
    "Congratulations! You've won a free lottery. Click here to claim your prize.",
    "Hello, I hope you're doing well. Let's catch up soon!"
]

# Preprocesar y vectorizar los nuevos correos electrónicos
new_emails_processed = [' '.join(preprocessor.preprocess(email)) for email in new_emails]
X_new = vectorizer.transform(new_emails_processed)
predictions = clf.predict(X_new)

# Visualización de los resultados de las predicciones
results = pd.DataFrame({
    'Email': new_emails,
    'Prediction': ['Spam' if pred == 1 else 'Ham' for pred in predictions]
})

# Mostrar los resultados en forma de tabla
print("\nPredictions for new emails:")
print(results)

# Visualización de la precisión (opcional)
accuracy = accuracy_score(y_test, y_pred)
plt.figure(figsize=(6, 4))
plt.bar(['Test Accuracy'], [accuracy], color='blue')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy on Test Set')
plt.show()
