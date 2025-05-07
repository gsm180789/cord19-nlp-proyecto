# Se limpia la columna que contiene el texto y se guarda el texto limpio en una nueva columna
def preparar_data(dataset, combine_fields=True, fields_to_combine=('title', 'abstract'), new_field_name='text'):
    """
    Prepara el dataset para tareas de NLP.
    Argumentos que requiere la función:
        dataset: objeto Dataset de Hugging Face.
        combine_fields: bool, si True combina varias columnas en una nueva.
        fields_to_combine: tuple, nombres de columnas a combinar.
        new_field_name: str, nombre de la nueva columna combinada.

    Returns: dataset procesado listo para modelado.
    """

    if combine_fields:
        def combine_examples(example):
            combined = ' '.join(
                (example.get(field) or '') for field in fields_to_combine
            ).strip()
            example[new_field_name] = combined
            return example

        dataset = dataset.map(combine_examples)

    return dataset
