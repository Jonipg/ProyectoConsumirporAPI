import os
import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

# Configuración de directorios
RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_kdd_dataset(data_path):
    """Carga el dataset NSL-KDD desde un archivo ARFF."""
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
    attributes = [attr[0] for attr in dataset["attributes"]]
    return pd.DataFrame(dataset["data"], columns=attributes)

def train_val_test_split(df, stratify_col=None, rstate=42, shuffle=True):
    """Divide el dataset en train, validation y test, con estratificación opcional."""
    strat = df[stratify_col] if stratify_col else None
    train, test = train_test_split(df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test[stratify_col] if stratify_col else None
    val, test = train_test_split(test, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return train, val, test

class DeleteNanRows(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.dropna()

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        scale_attrs = X_copy[self.attributes]
        robust_scaler = RobustScaler()
        X_scaled = robust_scaler.fit_transform(scale_attrs)
        X_scaled = pd.DataFrame(X_scaled, columns=self.attributes, index=X_copy.index)
        
        for attr in self.attributes:
            X_copy[attr] = X_scaled[attr]
        
        return X_copy

def create_pipeline(num_features, cat_features):
    """Crea un pipeline de procesamiento para datos numéricos y categóricos."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_features)
    ])
    return full_pipeline

def process_and_visualize_pipeline(data_path):
    """Procesa el dataset y genera visualizaciones."""
    df = load_kdd_dataset(data_path)

    # Validación de columnas antes de dividir
    if "class" not in df.columns or "protocol_type" not in df.columns:
        raise ValueError("El dataset debe contener las columnas 'class' y 'protocol_type'.")

    train_set, val_set, test_set = train_val_test_split(df, stratify_col="protocol_type")

    # Preparación de datos
    X_train = train_set.drop("class", axis=1)
    y_train = train_set["class"]

    # Validación de dimensiones
    if len(X_train) != len(y_train):
        raise ValueError(f"Las dimensiones no coinciden: {len(X_train)} muestras en X_train, {len(y_train)} en y_train")

    num_features = X_train.select_dtypes(exclude=["object"]).columns
    cat_features = X_train.select_dtypes(include=["object"]).columns

    print(f"Numéricas: {list(num_features)}, Categóricas: {list(cat_features)}")

    pipeline = create_pipeline(num_features, cat_features)

    try:
        X_train_prep = pd.DataFrame(pipeline.fit_transform(X_train), index=X_train.index)
    except Exception as e:
        raise ValueError(f"Error al procesar el pipeline: {e}")

    # Guardar estadísticas descriptivas
    original_stats = X_train[num_features].describe().to_html(classes="table table-striped table-bordered", border=0)
    transformed_stats = X_train_prep.describe().to_html(classes="table table-striped table-bordered", border=0)

    # Visualización de datos originales
    scatter_plot_path = os.path.join(RESULTS_DIR, "scatter_original_vs_transformed.png")
    if len(num_features) < 5:
        raise ValueError("No hay suficientes columnas numéricas para generar el gráfico.")
    
    try:
        pd.plotting.scatter_matrix(X_train[num_features[:5]], figsize=(10, 8), alpha=0.7, diagonal='kde')
        plt.suptitle("Scatter Matrix - Datos Originales", fontsize=14)
        plt.tight_layout()
        plt.savefig(scatter_plot_path)
        plt.close()
        print(f"Gráfico guardado en: {scatter_plot_path}")
    except Exception as e:
        print(f"Error al guardar el gráfico: {e}")

    return {
        "original_stats": original_stats,
        "transformed_stats": transformed_stats,
        "scatter_plot": scatter_plot_path
    }

def run():
    """Método para ejecutar el pipeline completo."""
    data_path = "datasets/KDD/KDDTrain+.arff"
    try:
        results = process_and_visualize_pipeline(data_path)
        return results
    except Exception as e:
        return {"error": str(e)}
    
