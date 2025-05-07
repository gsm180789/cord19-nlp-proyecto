import pandas as pd
import pandas as pd
import json
import os
from datasets import Dataset

def load_metadata(metadata_path):
    """
       Carga el archivo metadata.csv como DataFrame.

       Parámetros:
       - metadata_path: Ruta al archivo metadata.csv.

       Returns:
       DataFrame con los metadatos.
    """
    return pd.read_csv(metadata_path)

def load_article_text(json_path):
    """
       Carga un archivo .json de un artículo y extrae el título, resumen y cuerpo.
       Además, extrae los artículos de referencia y artículos citados.

       Parámetro:
       - json_path: Ruta al archivo .json.

       Returns:
       - Tupla con el  título, resumen, cuerpo del artículo, artículos de referencia y artículos citados.
    """
    try:
        with open(json_path, 'r') as f:
            paper = json.load(f)

        # Título
        title = paper.get('metadata', {}).get('title', '')

        # Abstract (puede venir en forma de lista)
        abstract_parts = paper.get('abstract', [])
        abstract = ' '.join([part.get('text', '') for part in abstract_parts])

        # Body text (texto del artículo completo)
        body_parts = paper.get('body_text', [])
        body_text = ' '.join([part.get('text', '') for part in body_parts])

        # Bibliographic entries (referencias) - Articulos bibliografías
        bib_entries_parts = paper.get('bib_entries', {})
        bib_entries_list = []
        for entry_id, entry_data in bib_entries_parts.items():
            entry_title = entry_data.get('title', 'Título no disponible')
            authors_data = entry_data.get('authors', [])
            authors = ', '.join([f"{a.get('first', '')} {a.get('last', '')}".strip() for a in authors_data]) or 'Autor(es) no disponible'
            year = entry_data.get('year', 'Año no disponible')
            bib_entries_list.append(f"Título: {entry_title}, Autores: {authors}, Año: {year}")

        bib_entries = ' || '.join(bib_entries_list) or 'No hay referencias bibliográficas disponibles.'

        # Reference entries (artículos citados en cuerpo) - Articulos de referencia
        ref_entries_parts = paper.get('ref_entries', {})
        ref_entries_list = []
        for entry_id, entry_data in ref_entries_parts.items():
            entry_title = entry_data.get('text', 'Título no disponible')
            # Nota: ref_entries generalmente no tiene autores ni año, pero podrías agregarlo si existiera.
            ref_entries_list.append(f"Título: {entry_title}")

        ref_entries = ' || '.join(ref_entries_list) or 'No hay artículos citados disponibles.'

        return title, abstract, body_text, bib_entries, ref_entries

    except Exception as e:
        print(f"Error cargando {json_path}: {e}")
        return 'Título no disponible', 'Resumen no disponible', 'Cuerpo no disponible', 'No hay referencias bibliográficas disponibles.', 'No hay artículos citados disponibles.'
