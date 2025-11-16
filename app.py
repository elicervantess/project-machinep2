import os
import re
import numpy as np
import pandas as pd
import random
import streamlit as st
from PIL import Image
import altair as alt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from src.feature_extractor import extract_features_for_image
from src.dimensionality import pca_transform
from src.config import (
    OUTPUT_DIR,
    APP_ARTIFACTS_DIR,
    TRAIN_META_FILTERED_CSV,
    TRAIN_META_FILTERED_CSV_FALLBACK,
    FEATURES_PCA,
    FEATURES_PCA_FALLBACK,
    PCA_MEAN_NPY,
    PCA_MEAN_NPY_FALLBACK,
    PCA_COMPONENTS_NPY,
    PCA_COMPONENTS_NPY_FALLBACK,
    COORDS_2D,
    COORDS_2D_FALLBACK,
)

# =========================
# Helper functions for clustering evaluation
def _euclidean_distances(A, B):
    """
    Calcula matriz de distancias euclidianas entre A (n_a, d) y B (n_b, d).
    Devuelve (n_a, n_b).
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    aa = np.sum(A * A, axis=1, keepdims=True)          # (n_a,1)
    bb = np.sum(B * B, axis=1, keepdims=True).T        # (1,n_b)
    d2 = aa + bb - 2.0 * (A @ B.T)
    d2[d2 < 0] = 0.0
    return np.sqrt(d2)

def approximate_silhouette_score(X, labels, sample_size=500, seed=42):
    """
    Cálculo aproximado del silhouette score PROMEDIO.
    - Toma una muestra aleatoria de puntos para no hacer O(N^2) completa.
    - X: (N,d)
    - labels: (N,)
    Retorna float (puede ser None si algún cluster tiene 1 solo punto en la muestra).
    """
    rnd = random.Random(seed)
    N = X.shape[0]
    idx_all = list(range(N))
    rnd.shuffle(idx_all)
    idx_sample = idx_all[:min(sample_size, N)]
    Xs = X[idx_sample]
    ls = labels[idx_sample]

    # precalculamos distancias entre los puntos sampleados
    D = _euclidean_distances(Xs, Xs)  # (m,m)

    silhouettes = []
    for i in range(len(idx_sample)):
        my_label = ls[i]

        # distancias a los de mi mismo cluster
        same_mask = (ls == my_label)
        same_mask[i] = False  # excluye self
        if np.sum(same_mask) == 0:
            # un único punto de ese cluster en la muestra => silhouette indefinido
            continue
        a_i = np.mean(D[i, same_mask])

        # distancias promedio hacia TODOS los otros clusters y tomar el mínimo
        b_i_candidates = []
        for other_label in set(ls):
            if other_label == my_label:
                continue
            other_mask = (ls == other_label)
            if np.sum(other_mask) == 0:
                continue
            b_i_candidates.append(np.mean(D[i, other_mask]))
        if not b_i_candidates:
            continue
        b_i = min(b_i_candidates)

        # silhouette individual
        denom = max(a_i, b_i)
        if denom > 0:
            s_i = (b_i - a_i) / denom
            silhouettes.append(s_i)

    if len(silhouettes) == 0:
        return None
    return float(np.mean(silhouettes))

def davies_bouldin_index(X, labels):
    """
    Calcula Davies-Bouldin Index de manera clásica.
    - X: (N,d)
    - labels: (N,)
    Retorna float.
    """
    unique_labels = sorted(set(labels))
    k = len(unique_labels)
    if k < 2:
        return None

    centroids = []
    S = []  # dispersión intra-cluster promedio
    for lab in unique_labels:
        cluster_pts = X[labels == lab]
        c = np.mean(cluster_pts, axis=0)
        centroids.append(c)
        # promedio de distancias al centroide
        dists = np.sqrt(np.sum((cluster_pts - c) ** 2, axis=1))
        S.append(np.mean(dists))

    centroids = np.vstack(centroids)  # (k,d)
    S = np.array(S)                   # (k,)

    # distancias entre centroides
    M = _euclidean_distances(centroids, centroids)  # (k,k)
    # Evitar división entre cero en la diagonal
    np.fill_diagonal(M, 1e-12)

    # R_ij = (S_i + S_j) / M_ij
    # Para cada i, tomar R_i = max_j R_ij
    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            R[i, j] = (S[i] + S[j]) / M[i, j]

    R_i = np.max(R, axis=1)
    DB = np.mean(R_i)
    return float(DB)
# Utilidades de datos
# =========================

def _ensure_artifact(path, friendly_name, fallback_path=None):
    """
    Verifica que exista el artefacto en `path` o, si no está,
    intenta devolver `fallback_path` (artefacto empaquetado en el repo).
    """
    if os.path.exists(path):
        return path
    if fallback_path and os.path.exists(fallback_path):
        return fallback_path
    raise FileNotFoundError(
        f"No se encontró {friendly_name}.\n"
        f"Buscado en: '{path}'"
        + (f" y '{fallback_path}'" if fallback_path else "")
        + ". Genera los artefactos ejecutando 'python train_pipeline.py'."
    )

@st.cache_resource
def load_artifacts():
    train_csv_path = _ensure_artifact(
        TRAIN_META_FILTERED_CSV, "train_meta_used.csv", TRAIN_META_FILTERED_CSV_FALLBACK
    )
    features_path = _ensure_artifact(
        FEATURES_PCA, "features_reduced_pca.npy", FEATURES_PCA_FALLBACK
    )
    coords_path = _ensure_artifact(
        COORDS_2D, "coords2d_pca.npy", COORDS_2D_FALLBACK
    )
    pca_mean_path = _ensure_artifact(
        PCA_MEAN_NPY, "pca_mean.npy", PCA_MEAN_NPY_FALLBACK
    )
    pca_components_path = _ensure_artifact(
        PCA_COMPONENTS_NPY, "pca_components.npy", PCA_COMPONENTS_NPY_FALLBACK
    )

    df_train = pd.read_csv(train_csv_path)
    X_train = np.load(features_path)  # (N, k)
    coords2d = np.load(coords_path)    # (N, 2)
    pca_mean = np.load(pca_mean_path) # (d_orig,)
    pca_components = np.load(pca_components_path) # (d_orig, k)

    # Reparar rutas de pósters: buscar en app_artifacts si no existen en data
    def fix_poster_path(path):
        if os.path.exists(path):
            return path
        # Intentar en app_artifacts/posters/
        filename = os.path.basename(path)
        fallback = os.path.join(APP_ARTIFACTS_DIR, "posters", filename)
        if os.path.exists(fallback):
            return fallback
        return path  # devolver original aunque no exista
    
    df_train["poster_path"] = df_train["poster_path"].apply(fix_poster_path)

    # sacamos el año del título para filtrado por año
    years = []
    for t in df_train["title"]:
        match = re.search(r"\((\d{4})\)\s*$", str(t))
        if match:
            years.append(int(match.group(1)))
        else:
            years.append(None)
    df_train["year"] = years

    # géneros: df_train["genres"] viene tipo "Comedy|Drama"
    # hacemos una lista rápida por película
    df_train["genre_list"] = df_train["genres"].fillna("").apply(lambda g: g.split("|"))

    return df_train, X_train, coords2d, pca_mean, pca_components

df_train, X_train, coords2d, pca_mean, pca_components = load_artifacts()

# =========================
# Filtros globales (sidebar)
# =========================
st.sidebar.header("Filtros por metadatos")
# años
valid_years = [y for y in df_train["year"] if y is not None]
if len(valid_years) == 0:
    min_year, max_year = 1900, 2025
else:
    min_year, max_year = int(np.nanmin(valid_years)), int(np.nanmax(valid_years))

year_range = st.sidebar.slider(
    "Rango de año",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1
)

# géneros
def build_all_genres(df):
    all_genres = set()
    for g_list in df["genre_list"]:
        for g in g_list:
            g = g.strip()
            if g:
                all_genres.add(g)
    return sorted(list(all_genres))

all_genres = build_all_genres(df_train)
sel_genres = st.sidebar.multiselect(
    "Filtrar por género",
    options=all_genres,
    default=[]
)

def apply_filters(df, coords2d, X_train, sel_genres, year_range):
    """
    Aplica filtros de género y año y devuelve:
    df_filtrado, coords2d_filtrado, X_filtrado, mask_indices (bool de longitud N)
    """
    # género
    if sel_genres:
        has_genre = []
        for g_list in df["genre_list"]:
            inter = set(g_list) & set(sel_genres)
            has_genre.append(len(inter) > 0)
        genre_mask = np.array(has_genre, dtype=bool)
    else:
        genre_mask = np.ones(len(df), dtype=bool)

    # año
    years_arr = df["year"].fillna(0).astype(int).values
    ymask = (years_arr >= year_range[0]) & (years_arr <= year_range[1])

    mask = genre_mask & ymask
    return (
        df[mask].reset_index(drop=True),
        coords2d[mask],
        X_train[mask],
        mask
    )

df_filt, coords2d_filt, X_filt, mask_idx = apply_filters(
    df_train, coords2d, X_train, sel_genres, year_range
)


def euclidean_neighbors(x_query, X_train, allowed_index=None, topn=10):
    """
    Retorna índices de las topn películas más cercanas en el embedding PCA.
    x_query shape: (1, k)
    Si `allowed_index` es un array de índices válidos, solo considera esos puntos como candidatos.
    """
    diff = X_train - x_query
    dists = np.sqrt(np.sum(diff * diff, axis=1))

    if allowed_index is not None:
        allowed_index = np.asarray(allowed_index)
        d_sub = dists[allowed_index]
        order_local = np.argsort(d_sub)[:topn]
        order = allowed_index[order_local]
        return order, dists[order]
    else:
        order = np.argsort(dists)[:topn]
        return order, dists[order]


def get_cluster_representatives(df, X_train, cluster_id, allowed_mask, topn=10):
    """
    Devuelve las topn películas más cercanas al centroide del cluster dado,
    restringiendo a los índices permitidos por `allowed_mask` (bool de longitud N).
    """
    base_mask = (df["cluster_kmeans"] == cluster_id).values
    mask = base_mask & allowed_mask
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return []

    centroid = X_train[idxs].mean(axis=0, keepdims=True)
    diff = X_train[idxs] - centroid
    dists = np.sqrt(np.sum(diff * diff, axis=1))
    order_local = np.argsort(dists)[:topn]
    return idxs[order_local]

# -------------------------
# Filtros globales y utilidades
# -------------------------
def majority_genre(series_genre_list):
    """
    Retorna lista con el 'género mayoritario' por registro.
    Si no hay géneros, devuelve 'Unknown'.
    (Usa el primer género como proxy cuando no hay one-hot.)
    """
    out = []
    for gl in series_genre_list:
        if isinstance(gl, list) and len(gl) > 0 and gl[0].strip():
            out.append(gl[0].strip())
        else:
            out.append("Unknown")
    return np.array(out)


def embed_uploaded_image(pil_img, pca_mean, pca_components):
    """
    - guarda temporalmente la imagen subida
    - extrae features visuales crudas
    - proyecta al espacio PCA entrenado (mismas componentes)
    """
    tmp_path = os.path.join(OUTPUT_DIR, "_tmp_query_image.jpg")
    pil_img.save(tmp_path)

    vec = extract_features_for_image(tmp_path)
    if vec is None:
        return None
    vec = vec.reshape(1, -1)
    x_proj = pca_transform(vec, pca_mean, pca_components)
    return x_proj


def embed_known_movie(movie_row, pca_mean, pca_components):
    """
    Toma una fila del df_train (que tiene poster_path),
    re-extrae features del póster, y proyecta a PCA.
    """
    poster_path = movie_row["poster_path"]
    if not os.path.exists(poster_path):
        return None
    vec = extract_features_for_image(poster_path)
    if vec is None:
        return None
    vec = vec.reshape(1, -1)
    x_proj = pca_transform(vec, pca_mean, pca_components)
    return x_proj


# =========================
# UI STREAMLIT
# =========================

st.set_page_config(
    page_title="Reco de Pelis por Póster 🎬",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Sistema de Recomendación Visual de Películas")
st.caption("Agrupamiento no supervisado + PCA manual + búsqueda por similitud de póster")

tabs = st.tabs([
    "🔍 Buscar por similitud visual",
    "📦 Clusters",
    "🗺️ Mapa 2D",
    "📈 Evaluación del modelo",
])

# -------------------------------------------------
# TAB 1: BUSCAR PELÍCULAS POR SIMILITUD VISUAL
# -------------------------------------------------

with tabs[0]:
    st.header("🔍 Buscar películas similares visualmente")

    col_left, col_right = st.columns([1,1])

    with col_left:
        st.subheader("Opción A: elegir una película existente")
        # dropdown con título
        # para que no sea gigante, podemos ordenar alfabéticamente
        df_sorted = df_filt.sort_values("title").reset_index(drop=True)

        selected_title = st.selectbox(
            "Elige una película como consulta",
            options=df_sorted["title"].tolist()
        )

        if st.button("Buscar similares (película seleccionada)"):
            row_q = df_sorted[df_sorted["title"] == selected_title].iloc[0]
            emb_q = embed_known_movie(row_q, pca_mean, pca_components)

            if emb_q is None:
                st.error("No se pudo procesar el póster de la película seleccionada.")
            else:
                allowed = np.where(mask_idx)[0]
                idxs, dists = euclidean_neighbors(emb_q, X_train, allowed_index=allowed, topn=10)
                st.subheader("Similares a: " + row_q["title"])

                # mostramos resultados
                for rank, (idx_res, dist_val) in enumerate(zip(idxs, dists), start=1):
                    rec_row = df_train.iloc[idx_res]
                    colA, colB = st.columns([1,4])
                    with colA:
                        if os.path.exists(rec_row["poster_path"]):
                            st.image(rec_row["poster_path"], use_container_width=True)
                    with colB:
                        st.markdown(f"**#{rank}: {rec_row['title']}**")
                        st.write(f"movieId: {rec_row['movieId']}")
                        st.write(f"Géneros: {rec_row['genres']}")
                        st.write(f"Cluster KMeans: {int(rec_row['cluster_kmeans'])}")
                        st.write(f"Distancia visual: {float(dist_val):.4f}")

    with col_right:
        st.subheader("Opción B: subir una imagen externa")
        uploaded = st.file_uploader("Sube un póster o fotograma (jpg/png)", type=["jpg","jpeg","png"])

        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Consulta subida", use_container_width=True)

            if st.button("Buscar similares (imagen subida)"):
                emb_q = embed_uploaded_image(pil_img, pca_mean, pca_components)
                if emb_q is None:
                    st.error("No pude extraer features de la imagen subida.")
                else:
                    allowed = np.where(mask_idx)[0]
                    idxs, dists = euclidean_neighbors(emb_q, X_train, allowed_index=allowed, topn=10)
                    st.subheader("Películas más parecidas visualmente:")

                    for rank, (idx_res, dist_val) in enumerate(zip(idxs, dists), start=1):
                        rec_row = df_train.iloc[idx_res]
                        colA, colB = st.columns([1,4])
                        with colA:
                            if os.path.exists(rec_row["poster_path"]):
                                st.image(rec_row["poster_path"], use_container_width=True)
                        with colB:
                            st.markdown(f"**#{rank}: {rec_row['title']}**")
                            st.write(f"movieId: {rec_row['movieId']}")
                            st.write(f"Géneros: {rec_row['genres']}")
                            st.write(f"Cluster KMeans: {int(rec_row['cluster_kmeans'])}")
                            st.write(f"Distancia visual: {float(dist_val):.4f}")


# -------------------------------------------------
# TAB 2: CLUSTERS
# -------------------------------------------------

with tabs[1]:
    st.header("📦 Películas representativas por cluster")

    # lista de clusters únicos
    clusters = sorted(df_filt["cluster_kmeans"].unique().tolist())
    cluster_sel = st.selectbox("Selecciona un cluster K-Means", options=clusters)

    st.write("Mostrando las películas más 'centrales' de ese cluster (visualmente más cercanas al centroide).")
    rep_idxs = get_cluster_representatives(df_train, X_train, cluster_sel, allowed_mask=mask_idx, topn=12)

    grid_cols = st.columns(4)
    for i, idx_res in enumerate(rep_idxs):
        rec_row = df_train.iloc[idx_res]
        with grid_cols[i % 4]:
            if os.path.exists(rec_row["poster_path"]):
                st.image(rec_row["poster_path"], use_container_width=True)
            st.markdown(f"**{rec_row['title']}**")
            st.caption(f"Géneros: {rec_row['genres']}")
            st.caption(f"movieId: {rec_row['movieId']}")


# -------------------------------------------------
# TAB 3: MAPA 2D
# -------------------------------------------------

with tabs[2]:
    st.header("🗺️ Distribución 2D de todas las películas")

    st.write("Cada punto es una película en el espacio visual reducido a 2D. El color es el cluster K-Means.")
    st.write("Los filtros de **género** y **año** se configuran en la barra lateral y aplican a todas las vistas.")
    plot_df = pd.DataFrame({
        "x": coords2d_filt[:,0],
        "y": coords2d_filt[:,1],
        "title": df_filt["title"],
        "cluster": df_filt["cluster_kmeans"].astype(int).astype(str),
        "year": df_filt["year"],
        "genres": df_filt["genres"]
    })
    chart = alt.Chart(plot_df).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X("x", title="PC2D-X"),
        y=alt.Y("y", title="PC2D-Y"),
        color=alt.Color("cluster", legend=alt.Legend(title="Cluster")),
        tooltip=["title", "year", "genres", "cluster"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# -------------------------------------------------
# TAB 4: EVALUACIÓN DEL MODELO
# -------------------------------------------------

with tabs[3]:
    st.header("📈 Evaluación del modelo")

    st.write("Estas métricas miden qué tan bien se agrupan visualmente las películas sin usar etiquetas externas.")

    # calculamos métricas bajo demanda (para no hacerlo si el usuario no mira esta pestaña)
    if st.button("Calcular métricas de calidad de clustering"):
        labels_vis = df_filt["cluster_kmeans"].to_numpy()
        X_vis = X_filt

        sil = approximate_silhouette_score(X_vis, labels_vis, sample_size=500, seed=42)
        dbi = davies_bouldin_index(X_vis, labels_vis)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Silhouette (aprox)")
            st.metric("Silhouette", "N/A" if sil is None else f"{sil:.3f}")

        with col2:
            st.subheader("Davies-Bouldin")
            st.metric("DB Index", "N/A" if dbi is None else f"{dbi:.3f}")

        # --- Métricas con etiquetas externas (género mayoritario) ---
        maj = majority_genre(df_filt["genre_list"])
        try:
            nmi = normalized_mutual_info_score(maj, labels_vis)
            ari = adjusted_rand_score(maj, labels_vis)
        except Exception:
            nmi, ari = None, None

        with col3:
            st.subheader("NMI (género)")
            st.metric("NMI", "N/A" if nmi is None else f"{nmi:.3f}")

        with col4:
            st.subheader("ARI (género)")
            st.metric("ARI", "N/A" if ari is None else f"{ari:.3f}")

        st.caption("Estas métricas se calculan sobre el subconjunto filtrado por la barra lateral.")

        # --- Coherencia de género por cluster (top-3) ---
        st.subheader("Coherencia de género por cluster (top-3)")
        GENRES = build_all_genres(df_train)
        rows = []
        for c in sorted(np.unique(labels_vis).tolist()):
            sub = df_filt[labels_vis == c]
            if len(sub) == 0:
                continue
            # contar proporciones
            counts = {}
            for g_list in sub["genre_list"]:
                for g in g_list:
                    g=g.strip()
                    if not g: continue
                    counts[g] = counts.get(g, 0) + 1
            if not counts:
                rows.append({"cluster": int(c), "top_genres": "None"})
                continue
            # top-3
            total = sum(counts.values())
            top3 = sorted(counts.items(), key=lambda t: t[1], reverse=True)[:3]
            desc = ", ".join([f"{g} {cnt/total:.2f}" for g,cnt in top3])
            rows.append({"cluster": int(c), "top_genres": desc})
        if rows:
            coh_df = pd.DataFrame(rows).sort_values("cluster")
            st.table(coh_df)
        else:
            st.info("No hay suficientes datos tras el filtrado para calcular coherencia.")

    st.subheader("Distribución de tamaños de cluster")
    cluster_counts = (
        df_filt["cluster_kmeans"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster_kmeans")
        .reset_index(name="count")
    )

    st.dataframe(cluster_counts)

    st.caption("Un cluster sano suele tener varias decenas o cientos de elementos. Si ves clusters con 1-2 elementos, el modelo los está tratando como outliers visuales.")
