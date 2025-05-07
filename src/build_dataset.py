def build_dataset_v2(metadata_df, pmc_json_folder, pdf_json_folder, num_samples=100):
    """
    Construye un dataset Hugging Face usando archivos .json del corpus CORD-19.
    Da prioridad a los archivos en 'pmc_json', y si no existen, usa los de 'pdf_json'.

    Parámetros:
    - metadata_df: DataFrame con los metadatos.
    - pmc_json_folder: Ruta a la carpeta con archivos JSON PMC.
    - pdf_json_folder: Ruta a la carpeta con archivos JSON PDF.
    - num_samples: Número máximo de muestras a cargar (por defecto 100).

    Return:
    - Dataset de Hugging Face.
    """
    samples = []

    for idx, row in metadata_df.iterrows():
        sha = row['sha']

        if isinstance(sha, str):
            first_sha = sha.split(';')[0]

            # Intentar con pmc_json
            pmc_path = os.path.join(pmc_json_folder, f"{first_sha}.json")
            pdf_path = os.path.join(pdf_json_folder, f"{first_sha}.json")

            if os.path.exists(pmc_path):
                json_path = pmc_path
            elif os.path.exists(pdf_path):
                json_path = pdf_path
            else:
                continue  # Ninguna versión disponible

            title, abstract, body_text, bib_entries, ref_entries = load_article_text(json_path)

            if title or abstract or body_text or bib_entries or ref_entries:
                samples.append({
                    'title': title,
                    'abstract': abstract,
                    'body_text': body_text,
                    'bib_entries': bib_entries,
                    'ref_entries': ref_entries
                })

            # Mostrar progreso cada 1000 muestras cargadas
            if len(samples) % 1000 == 0:
                print(f"{len(samples)} muestras cargadas...")

        if len(samples) >= num_samples:
            break

    print(f"Total de muestras cargadas: {len(samples)}")
    dataset = Dataset.from_list(samples)
    return dataset
