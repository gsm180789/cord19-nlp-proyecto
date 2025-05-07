from transformers import AutoTokenizer, AutoModelForSequenceClassification

def cargar_modelo_biobert():
    """Carga el modelo y tokenizador BioBERT v1.1 para clasificación."""
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    return tokenizer, model

