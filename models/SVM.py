# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.impute import SimpleImputer

# Función para particionar el dataset
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return train_set, val_set, test_set

# Función para visualizar el límite de decisión de SVM
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]
    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

# Carga de datos
df = pd.read_csv("datasets/FinalDataset/Phishing.csv")

# Verificar valores faltantes o infinitos
print("Valores faltantes por columna:\n", df.isna().sum())
print("Valores infinitos por columna:\n", df.isin([np.inf, -np.inf]).sum())

# Visualización inicial
plt.figure(figsize=(12, 6))
plt.scatter(df["domainUrlRatio"][df['URL_Type_obf_Type'] == "phishing"], 
            df["domainlength"][df['URL_Type_obf_Type'] == "phishing"], c="r", marker=".")
plt.scatter(df["domainUrlRatio"][df['URL_Type_obf_Type'] == "benign"], 
            df["domainlength"][df['URL_Type_obf_Type'] == "benign"], c="g", marker="x")
plt.xlabel("domainUrlRatio", fontsize=13)
plt.ylabel("domainlength", fontsize=13)
plt.show()

# División de los datos
train_set, val_set, test_set = train_val_test_split(df)

X_train = train_set.drop("URL_Type_obf_Type", axis=1)
y_train = train_set["URL_Type_obf_Type"].copy()

X_val = val_set.drop("URL_Type_obf_Type", axis=1)
y_val = val_set["URL_Type_obf_Type"].copy()

X_test = test_set.drop("URL_Type_obf_Type", axis=1)
y_test = test_set["URL_Type_obf_Type"].copy()

# Preprocesamiento
X_train = X_train.drop("argPathRatio", axis=1)
X_val = X_val.drop("argPathRatio", axis=1)
X_test = X_test.drop("argPathRatio", axis=1)

imputer = SimpleImputer(strategy="median")
X_train_prep = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=y_train.index)
X_val_prep = pd.DataFrame(imputer.fit_transform(X_val), columns=X_val.columns, index=y_val.index)
X_test_prep = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns, index=y_test.index)

# Reducir las dimensiones para visualización
X_train_reduced = X_train_prep[["domainUrlRatio", "domainlength"]].copy()
X_val_reduced = X_val_prep[["domainUrlRatio", "domainlength"]].copy()

# Entrenamiento de SVM con kernel lineal
svm_clf = SVC(kernel="linear", C=50)
svm_clf.fit(X_train_reduced, y_train)

# Visualizar límite de decisión
plt.figure(figsize=(12, 6))
plt.plot(X_train_reduced.values[:, 0][y_train == "phishing"], 
         X_train_reduced.values[:, 1][y_train == "phishing"], "g^")
plt.plot(X_train_reduced.values[:, 0][y_train == "benign"], 
         X_train_reduced.values[:, 1][y_train == "benign"], "bs")
plot_svc_decision_boundary(svm_clf, 0, 1)
plt.xlabel("domainUrlRatio", fontsize=15)
plt.ylabel("domainlength", fontsize=15)
plt.title("$C = {}$".format(svm_clf.C), fontsize=12)
plt.show()

# Predicción y evaluación
y_pred = svm_clf.predict(X_val_reduced)
print("F1 Score (kernel lineal):", f1_score(y_pred, y_val, pos_label="phishing"))

# SVM con características polinómicas
poly_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=20, loss="hinge", random_state=42, max_iter=100000))
])
y_train_num = y_train.factorize()[0]
poly_svm_clf.fit(X_train_reduced, y_train_num)

# Visualización de predicciones polinómicas
fig, axes = plt.subplots(ncols=2, figsize=(15, 5), sharey=True)
plot_svc_decision_boundary(poly_svm_clf.named_steps['svm_clf'], 0, 1)
plt.show()

# Evaluación del modelo polinómico
y_prep = poly_svm_clf.predict(X_val_reduced)
print("F1 Score (polinómico):", f1_score(y_prep, y_val.factorize()[0]))
