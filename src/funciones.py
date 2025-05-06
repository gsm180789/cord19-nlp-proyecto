"""Funciones auxiliares para el análisis NLP del dataset CORD-19."""

def limpiar_texto(texto):
    """Limpia y normaliza un texto: minúsculas, sin puntuación, etc."""
    import re
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
    return texto.strip()
