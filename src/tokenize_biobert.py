def tokenize_multicolumn_biobert(dataset_splits, max_length=512):
    """
    Tokeniza los datasets de entrenamiento, validación y prueba usando las columnas
    'title', 'abstract' y 'body_text' concatenadas como entrada para BioBERT.

    Parámetros:
    - dataset_splits: DatasetDict con los splits 'train', 'validation' y 'test'.
    - max_length: número máximo de tokens (BioBERT: 512 por defecto).

    Return:
    - DatasetDict con los textos tokenizados.
    """
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    def tokenize_function(dataset):
        # Concatenar las columnas a ser utilizadas para el entrenamiento
        combined_text = [
            f"{title} {abstract} {body_text}"
            for title, abstract, body_text in zip(dataset['title'], dataset['abstract'], dataset['body_text'])
        ]  # Use zip to iterate over corresponding elements
        return tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=max_length
        )

    # Tokenizar todos los splits
    tokenized_datasets = dataset_splits.map(tokenize_function, batched=True)

    return tokenized_datasets
