import importlib

def load_model(module_name):
    try:
        # Carga dinámicamente el módulo
        module = importlib.import_module(f"models.{module_name}")
        print(f"Modelo cargado: {module_name}")
        
        # Verifica si el módulo tiene el método 'run'
        if hasattr(module, "run"):
            print(f"El modelo tiene el método 'run'.")
        else:
            print(f"El modelo no tiene el método 'run'.")
        return module
    except ModuleNotFoundError as e:
        print(f"Error al cargar el modelo: {e}")
        return None

if __name__ == "__main__":
    # Prueba con un modelo de ejemplo
    model = load_model("transformer_pipeline")  # Asegúrate de usar el nombre correcto del módulo
    if model:
        print("El modelo fue cargado exitosamente.")
    else:
        print("Hubo un problema al cargar el modelo.")
