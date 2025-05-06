import json
import sys

def limpiar_widgets(path_entrada, path_salida):
    with open(path_entrada, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    if 'widgets' in notebook.get('metadata', {}):
        del notebook['metadata']['widgets']
        print("Metadatos 'widgets' eliminados.")
    else:
        print("No se encontró 'metadata.widgets'.")

    with open(path_salida, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)
        print(f"Notebook limpio guardado en {path_salida}")
