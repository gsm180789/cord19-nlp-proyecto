import os
import joblib
import pickle

def guardar_entorno_completo(model, tokenizer, variables: dict, carpeta_drive: str):
    """
    Guarda modelo, tokenizer, variables y entorno en la carpeta especificada en Google Drive.
    Parámetros:
    - model: modelo HuggingFace
    - tokenizer: tokenizer HuggingFace
    - variables: diccionario de variables a guardar
    - carpeta_drive: ruta absoluta en Google Drive (ej: '/content/drive/MyDrive/MiProyecto')
    """

    os.makedirs(carpeta_drive, exist_ok=True)

    # Guardar modelo y tokenizer
    model.save_pretrained(f"{carpeta_drive}/modelo")
    tokenizer.save_pretrained(f"{carpeta_drive}/modelo")

    # **Change: Instead of pickling the Trainer object, pickle only essential data.**
    variables_to_save = {
        'training_args': variables['args'],
        'evaluation_results': variables['resultados_eval'],
        # Add other essential data as needed
    }

    with open(f"{carpeta_drive}/variables.pkl", 'wb') as f:
        pickle.dump(variables_to_save, f)  # Pickle the modified dictionary

    # Guardar lista de paquetes instalados
    os.system(f"pip freeze > {carpeta_drive}/requisitos.txt")

    print(f"Todo guardado exitosamente en: {carpeta_drive}")
