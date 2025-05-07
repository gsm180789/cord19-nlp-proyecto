import os
import zipfile
import time
from google.colab import drive

def montar_drive():
    """Monta Google Drive en Colab."""
    drive.mount('/content/drive')

def extraer_zip(archivo_zip, carpeta_destino):
    """Extrae un archivo ZIP en una carpeta dada."""
    os.makedirs(carpeta_destino, exist_ok=True)
    inicio = time.time()
    with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
        zip_ref.extractall(carpeta_destino)
    print(f"Descomprimido en {carpeta_destino} en {(time.time() - inicio)/60:.2f} minutos")

def calcular_tamano_carpeta(carpeta):
    """Calcula el tamaño total de una carpeta (en MB)."""
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(carpeta) for f in fn)
    print(f"Tamaño total: {total / (1024**2):.2f} MB")
