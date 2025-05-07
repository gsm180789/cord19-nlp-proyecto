def split_dataset(dataset, train_size=0.8, val_size=0.1, test_size=0.1, seed=42):
    """
    Divide un dataset en subconjuntos de entrenamiento, validación y prueba.

    Parámetros:
    - dataset: objeto datasets.Dataset o DatasetDict (sin splits).
    - train_size: proporción del conjunto de entrenamiento (por defecto 0.8).
    - val_size: proporción del conjunto de validación (por defecto 0.1).
    - test_size: proporción del conjunto de prueba (por defecto 0.1).
    - seed: semilla para aleatorizar la partición.

    Return:
    - Un DatasetDict con keys 'train', 'validation' y 'test'.
    """
    from datasets import DatasetDict

    assert abs((train_size + val_size + test_size) - 1.0) < 1e-6, "Las proporciones deben sumar 1."

    # Dividir en train y temp
    train_test_split = dataset.train_test_split(test_size=(val_size + test_size), seed=seed)

    # Dividir el "temp" en validación y prueba
    temp = train_test_split['test'].train_test_split(test_size=test_size / (test_size + val_size), seed=seed)

    # Crear el DatasetDict final
    final_splits = DatasetDict({
        'train': train_test_split['train'],
        'validation': temp['train'],
        'test': temp['test']
    })

    return final_splits
