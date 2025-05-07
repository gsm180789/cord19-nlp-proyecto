covid_keywords = [
    'covid', 'coronavirus', 'covid-19', 'covid19',
    'sars-cov-2', 'pandemic', 'novel coronavirus', '2019-ncov'
]

def label_add(example):
    # Combina los campos relevantes en minúsculas
    full_text = f"{example['title']} {example['abstract']} {example['body_text']}".lower()

    # Devuelve la nueva columna 'label'
    return {
        'label': int(any(keyword in full_text for keyword in covid_keywords))
    }
