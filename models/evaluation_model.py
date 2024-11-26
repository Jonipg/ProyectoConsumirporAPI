import os
import pandas as pd
import numpy as np
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, precision_score,
    recall_score, f1_score, RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt

# Configuración de directorios
RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Función para cargar el dataset NSL-KDD
def load_kdd_dataset(data_path, sample_fraction=0.1):
    """Carga y reduce el tamaño del conjunto de datos."""
    with open(data_path, 'r') as file:
        dataset = arff.load(file)
    attributes = [attr[0] for attr in dataset["attributes"]]
    df = pd.DataFrame(dataset["data"], columns=attributes)
    return df.sample(frac=sample_fraction, random_state=42)  # Reduce el tamaño

# Procesamiento y evaluación
def logistic_regression_pipeline(data_path):
    # Cargar el dataset reducido
    df = load_kdd_dataset(data_path)
    
    # División en conjuntos
    train_set, test_set = train_test_split(df, test_size=0.4, random_state=42, stratify=df["class"])
    val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=42, stratify=test_set["class"])

    # Separar características y etiquetas
    def split_features_labels(df, label_col="class"):
        X = df.drop(label_col, axis=1)
        y = df[label_col]
        return X, y

    X_train, y_train = split_features_labels(train_set)
    X_val, y_val = split_features_labels(val_set)
    X_test, y_test = split_features_labels(test_set)

    # Pipeline de preprocesamiento
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),  # Reemplazar sparse_output por sparse
        ("scaler", StandardScaler())
    ])


    # Preprocesar los datos
    X_train_prep = preprocessor.fit_transform(X_train)
    X_val_prep = preprocessor.transform(X_val)
    X_test_prep = preprocessor.transform(X_test)

    # Entrenar el modelo
    clf = LogisticRegression(max_iter=5000, random_state=42)
    clf.fit(X_train_prep, y_train)

    # Evaluación y métricas
    y_pred_val = clf.predict(X_val_prep)

    # Matriz de confusión
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    ConfusionMatrixDisplay.from_estimator(clf, X_val_prep, y_val, values_format="d")
    plt.savefig(cm_path)
    plt.close()

    # Graficar curvas ROC y Precision-Recall
    roc_path = os.path.join(RESULTS_DIR, "roc_curve.png")
    pr_path = os.path.join(RESULTS_DIR, "precision_recall_curve.png")
    RocCurveDisplay.from_estimator(clf, X_val_prep, y_val)
    plt.savefig(roc_path)
    plt.close()
    PrecisionRecallDisplay.from_estimator(clf, X_val_prep, y_val)
    plt.savefig(pr_path)
    plt.close()

    # Métricas
    precision = precision_score(y_val, y_pred_val, pos_label="anomaly", zero_division=1)
    recall = recall_score(y_val, y_pred_val, pos_label="anomaly", zero_division=1)
    f1 = f1_score(y_val, y_pred_val, pos_label="anomaly", zero_division=1)

    # Verificación de las métricas
    print(f"Precision: {precision}")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm_path,
        "roc_curve": roc_path,
        "precision_recall_curve": pr_path
    }


# Ruta del dataset
data_path = "datasets/KDD/KDDTrain+.arff"
results = logistic_regression_pipeline(data_path)
