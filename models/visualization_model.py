import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix
import arff

# Configuración de directorios
RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Función para cargar el dataset
def load_kdd_dataset(data_path):
    """Carga y devuelve el dataset NSL-KDD como un DataFrame."""
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
    attributes = [attr[0] for attr in dataset['attributes']]
    return pd.DataFrame(dataset['data'], columns=attributes)

# Procesamiento del dataset
def process_and_visualize(data_path):
    """Carga, procesa el dataset y genera visualizaciones."""
    # Cargar el dataset
    df = load_kdd_dataset(data_path)

    # Generar estadísticas descriptivas
    stats = df.describe().to_html(classes="table table-striped", border=0)

    # Distribución de 'protocol_type'
    protocol_counts = df["protocol_type"].value_counts()
    protocol_plot_path = os.path.join(RESULTS_DIR, "protocol_distribution.png")
    protocol_counts.plot(kind="bar", color="skyblue")
    plt.title("Distribución de 'protocol_type'")
    plt.xlabel("Protocol Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(protocol_plot_path)
    plt.close()

    # Histogramas del dataset
    histograms_path = os.path.join(RESULTS_DIR, "histograms.png")
    df.hist(bins=50, figsize=(20, 15), color="teal", edgecolor="black")
    plt.suptitle("Histogramas de las columnas del dataset", fontsize=16)
    plt.tight_layout()
    plt.savefig(histograms_path)
    plt.close()

    # Codificar columnas categóricas
    labelencoder = LabelEncoder()
    df["class"] = labelencoder.fit_transform(df["class"])
    df["protocol_type"] = labelencoder.fit_transform(df["protocol_type"])
    
    # Codificar otras columnas categóricas si es necesario...
    if "service" in df.columns:
        df["service"] = labelencoder.fit_transform(df["service"])
    if "flag" in df.columns:
        df["flag"] = labelencoder.fit_transform(df["flag"])

    # Matriz de correlación
    corr_matrix = df.corr()
    correlation_plot_path = os.path.join(RESULTS_DIR, "correlation_matrix.png")
    plt.figure(figsize=(10, 10))
    plt.matshow(corr_matrix, fignum=1, cmap="coolwarm")
    plt.title("Matriz de correlación", pad=20, fontsize=14)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.colorbar()
    plt.savefig(correlation_plot_path)
    plt.close()

    # Scatter matrix
    attributes = ["same_srv_rate", "dst_host_srv_count", "class", "dst_host_same_srv_rate"]
    scatter_plot_path = os.path.join(RESULTS_DIR, "scatter_matrix.png")
    scatter_matrix(df[attributes], figsize=(12, 8), alpha=0.7, marker="o", hist_kwds={"bins": 15}, diagonal="kde")
    plt.suptitle("Scatter Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig(scatter_plot_path)
    plt.close()

    return {
        "stats": stats,
        "plots": {
            "protocol_distribution": protocol_plot_path,
            "histograms": histograms_path,
            "correlation_matrix": correlation_plot_path,
            "scatter_matrix": scatter_plot_path
        }
    }

# Ruta del dataset
data_path = "datasets/KDD/KDDTrain+.arff"
results = process_and_visualize(data_path)
