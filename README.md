# Aplicación de BioBERT y RAG para la Clasificación y Consulta de Literatura Científica en el Contexto de la Pandemia de COVID-19: Un Estudio Basado en el CORD-19

Este proyecto analiza el conjunto de datos CORD-19 utilizando técnicas de procesamiento de lenguaje natural (NLP). El enfoque principal se basa en:

- Clasificación binaria de documentos científicos utilizando el modelo BioBERT, fine-tuneado para identificar textos textos como "COVID" o "No COVID"

- RAG (Retrieval-Augmented Generation) para enriquecer respuestas generadas con recuperación de documentos basada en embeddings científicos.

## Requisitos

```bash
pip install -r requirements.txt
```

## Estructura

- `notebooks/`: Notebook principal con ejemplo de uso.
- `src/`: Código modular y comentado.
- `data/`: Instrucciones para obtener los datos CORD-19.
- `article/`: Artículo académico.

## Cómo ejecutar

1. Instala las dependencias.
2. Descarga el dataset CORD-19 (ver `/data/README.md`).
3. Abre el notebook y ejecútalo en Jupyter o Google Colab.

## Dataset

CORD-19: Conjunto de datos de artículos científicos sobre COVID-19, compilado por Allen Institute for AI.

---
Autor: Silvia Gamarra Morel
Repositorio: [GitHub](https://github.com/gsm180789/cord19-nlp-proyecto)
