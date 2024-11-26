from flask import Flask, render_template, request
import os
import importlib
import pandas as pd
import joblib
from flask import render_template
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)

MODEL_PATH = os.path.join(os.getcwd(), "models")

def load_model(module_name):
    try:
        module_path = f"models.{module_name}"
        return importlib.import_module(module_path)
    except ModuleNotFoundError:
        return None

@app.route("/")
def index():
    return render_template("index.html")

def process_model(model):
    try:
        result = model.run()  
        return result.get("original_stats", ""), result.get("transformed_stas", ""), result.get("scatter_plot", ""), None
    except Exception as e:
        return None, None, None, f"Error durante el procesamiento: {str(e)}"

@app.route("/transformers", methods=["GET", "POST"])
def transformers():
    model = load_model("transformer_pipeline") 
    if model and hasattr(model, "run"):
        try:
            result = model.run()  
            original_stats = result.get("original_stats", "")
            transformed_stats = result.get("transformed_stats", "")
            scatter_plot = result.get("scatter_plot", "")
            error_message = None  
        except Exception as e:
            error_message = f"Error durante el procesamiento: {str(e)}"
            original_stats = transformed_stats = scatter_plot = None
    else:
        original_stats = transformed_stats = scatter_plot = None
        error_message = "Error: No se encontró el método 'run' en el modelo de transformadores."

    return render_template(
        "transformers.html", 
        original_stats=original_stats,
        transformed_stats=transformed_stats,
        scatter_plot=scatter_plot,
        error_message=error_message 
    )
@app.route("/evaluation", methods=["GET", "POST"])
def evaluation():
    model = load_model("evaluation_model")
    if model and hasattr(model, "evaluate"):
        result = model.evaluate()

        print("Resultado de evaluate:", result)

        precision = result.get("precision") if result.get("precision") else "No disponible"
        recall = result.get("recall") if result.get("recall") else "No disponible"
        f1 = result.get("f1") if result.get("f1") else "No disponible"
        confusion_matrix = result.get("confusion_matrix") if result.get("confusion_matrix") else "No disponible"
        roc_curve = result.get("roc_curve") if result.get("roc_curve") else "No disponible"
        precision_recall_curve = result.get("precision_recall_curve") if result.get("precision_recall_curve") else "No disponible"

        print("Métricas: Precision: {}, Recall: {}, F1: {}".format(precision, recall, f1))

    else:
        result = {}  
        precision = recall = f1 = "No disponible"
        confusion_matrix = roc_curve = precision_recall_curve = "No disponible"

    return render_template(
        "evaluation.html",
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=confusion_matrix,
        roc_curve=roc_curve,
        precision_recall_curve=precision_recall_curve
    )

@app.route('/preparation')
def preparation():
    train_size = 75583
    val_size = 25195
    test_size = 25195
    columns_after_preprocessing = 34
    sample_preprocessed_data = [
        {'duration': 0.10, 'protocol_type': 'tcp', 'service': 'ftp_data', 'flag': 'SF'},
        {'duration': 0.30, 'protocol_type': 'udp', 'service': 'other', 'flag': 'SF'},
        {'duration': 0.60, 'protocol_type': 'tcp', 'service': 'ftp_data', 'flag': 'SF'},
        {'duration': 0.60, 'protocol_type': 'udp', 'service': 'other', 'flag': 'SF'},
        {'duration': 0.90, 'protocol_type': 'tcp', 'service': 'ftp_data', 'flag': 'SF'},
        {'duration': 0.00, 'protocol_type': 'tcp', 'service': 'ftp_data', 'flag': 'SF'}
    ]
    data_columns = [
        'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]

    example_data = [
        {'duration': 0.10, 'src_bytes': 18.0, 'dst_bytes': 53508.0, 'wrong_fragment': 0.0, 'urgent': 0.0, 'hot': 0.0, 'num_failed_logins': 0.0, 'num_compromised': 0.0, 'root_shell': 0.0, 'su_attempted': 0.0, 'num_root': 0.0, 'num_file_creations': 0.0, 'num_shells': 0.0, 'num_access_files': 1.0, 'num_outbound_cmds': 5.0, 'count': 0.0, 'srv_count': 0.0, 'serror_rate': 1.0, 'srv_serror_rate': 0.0, 'rerror_rate': 0.4, 'srv_rerror_rate': 9.0, 'same_srv_rate': 255.0, 'diff_srv_rate': 1.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 0.11, 'dst_host_srv_count': 0.03, 'dst_host_same_srv_rate': 0.0, 'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 0.0, 'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0},
        {'duration': 0.20, 'src_bytes': 0.0, 'dst_bytes': 0.0, 'wrong_fragment': 0.0, 'urgent': 0.0, 'hot': 0.0, 'num_failed_logins': 0.0, 'num_compromised': 0.0, 'root_shell': 0.0, 'su_attempted': 0.0, 'num_root': 0.0, 'num_file_creations': 0.0, 'num_shells': 0.0, 'num_access_files': 0.0, 'num_outbound_cmds': 200.0, 'count': 4.0, 'srv_count': 1.0, 'serror_rate': 1.0, 'srv_serror_rate': 0.0, 'rerror_rate': 0.02, 'srv_rerror_rate': 0.05, 'same_srv_rate': 0.0, 'diff_srv_rate': 255.0, 'srv_diff_host_rate': 4.0, 'dst_host_count': 0.02, 'dst_host_srv_count': 0.05, 'dst_host_same_srv_rate': 0.0, 'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 1.0, 'dst_host_srv_diff_host_rate': 1.0, 'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0},
        {'duration': 0.30, 'src_bytes': 304.0, 'dst_bytes': 636.0, 'wrong_fragment': 0.0, 'urgent': 0.0, 'hot': 0.0, 'num_failed_logins': 0.0, 'num_compromised': 0.0, 'root_shell': 0.0, 'su_attempted': 0.0, 'num_root': 0.0, 'num_file_creations': 0.0, 'num_shells': 0.0, 'num_access_files': 0.0, 'num_outbound_cmds': 4.0, 'count': 0.0, 'srv_count': 0.0, 'serror_rate': 0.0, 'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0, 'dst_host_count': 0.02, 'dst_host_srv_count': 0.02, 'dst_host_same_srv_rate': 0.0, 'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 0.0, 'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0}
    ]

    df = pd.DataFrame(example_data, columns=data_columns)

    return render_template(
        "preparation.html", 
        train_size=train_size, 
        val_size=val_size, 
        test_size=test_size, 
        columns_after_preprocessing=columns_after_preprocessing, 
        sample_preprocessed_data=sample_preprocessed_data
    )
    
@app.route("/visualization", methods=["GET", "POST"])
def visualization():
    model = load_model("visualization_model")

    plots_dir = os.path.join("static", "results")
    os.makedirs(plots_dir, exist_ok=True)

    if model and hasattr(model, "visualize"):
        results = model.visualize()

        # Aquí obtienes las estadísticas descriptivas
        if hasattr(model, "get_statistics"):
            stats = model.get_statistics()  # Asegúrate de tener esta función en tu modelo
        else:
            stats = {"Error": "No se encontraron estadísticas."}

        # Genera los gráficos
        for plot_name, plot_data in results["plots"].items():
            plot_path = os.path.join(plots_dir, f"{plot_name}.png")
            plot_data.save(plot_path) 
            results["plots"][plot_name] = f"/static/results/{plot_name}.png"

        # Asegúrate de que `results` incluya las estadísticas
        results["stats"] = stats
    else:
        results = {
            "stats": {"Error": "No se encontraron estadísticas."},
            "plots": {
                "protocol_distribution": "/static/images/error.png",
                "histograms": "/static/images/error.png",
                "correlation_matrix": "/static/images/error.png",
                "scatter_matrix": "/static/images/error.png"
            }
        }

    # Enviar las estadísticas y los gráficos al frontend
    return render_template("visualization.html", stats=results["stats"], plots=results["plots"])

@app.route("/logistic_regression", methods=["GET", "POST"])
def logistic_regresion():
    return render_template("logistic_regression.html")


if __name__ == "__main__":
    app.run(debug=True)
    