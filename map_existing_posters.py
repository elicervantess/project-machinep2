"""
Script para mapear pósters existentes (nombrados por título) con movies_train.csv
"""
import os
import re
import pandas as pd
from tqdm import tqdm

INPUT_CSV = "data/movies_train.csv"
OUTPUT_CSV = "data/movies_train_with_posters.csv"
POSTER_DIR = "data/posters"


def normalize_filename(title):
    """
    Normaliza un título de película para comparar con nombres de archivo.
    Ejemplo: "Toy Story (1995)" -> "Toy_Story_1995"
    """
    # Extraer año si existe
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match:
        year = match.group(1)
        title_clean = title[:match.start()].strip()
    else:
        year = ""
        title_clean = title.strip()
    
    # Normalizar título: quitar caracteres especiales, espacios -> guiones bajos
    title_norm = re.sub(r"[^\w\s]", "", title_clean)
    title_norm = re.sub(r"\s+", "_", title_norm)
    
    if year:
        return f"{title_norm}_{year}"
    return title_norm


def find_poster_for_movie(title, poster_files_dict):
    """
    Busca un póster que coincida con el título dado.
    poster_files_dict: diccionario {nombre_normalizado: ruta_completa}
    """
    normalized = normalize_filename(title)
    
    # Búsqueda exacta
    if normalized in poster_files_dict:
        return poster_files_dict[normalized]
    
    # Búsqueda flexible (caso insensitivo)
    normalized_lower = normalized.lower()
    for key, path in poster_files_dict.items():
        if key.lower() == normalized_lower:
            return path
    
    return None


def main():
    print("[1] Cargando movies_train.csv...")
    df = pd.read_csv(INPUT_CSV)
    
    print("[2] Escaneando pósters existentes...")
    poster_files = os.listdir(POSTER_DIR)
    poster_dict = {}
    
    for fname in poster_files:
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Quitar extensión
            name_no_ext = os.path.splitext(fname)[0]
            full_path = os.path.join(POSTER_DIR, fname)
            poster_dict[name_no_ext] = full_path
    
    print(f"   Encontrados {len(poster_dict)} pósters")
    
    print("[3] Mapeando pósters con películas...")
    df["poster_path"] = ""
    
    matched = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        title = row["title"]
        poster_path = find_poster_for_movie(title, poster_dict)
        
        if poster_path and os.path.exists(poster_path):
            df.at[i, "poster_path"] = poster_path
            matched += 1
    
    print(f"   Películas con póster: {matched}/{len(df)}")
    
    # Filtrar solo las que tienen póster
    df_with_posters = df[df["poster_path"] != ""].reset_index(drop=True)
    
    print(f"[4] Guardando {len(df_with_posters)} películas en {OUTPUT_CSV}...")
    df_with_posters.to_csv(OUTPUT_CSV, index=False)
    
    print("✅ ¡Listo! Ahora puedes ejecutar: python train_pipeline.py")


if __name__ == "__main__":
    main()
