# -*- coding: utf-8 -*-
"""
T1 — Línea base léxica: TF-IDF (parámetros por defecto) + similitud coseno
sobre el corpus mínimo d01–d10 y las 6 consultas q1–q6 del enunciado.
Calcula el ranking de los 10 documentos por consulta y las métricas
Hit@1, Hit@3 y MRR sobre las 6 consultas. Guarda todo en resultados.json.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# 1. Corpus: 10 documentos (d01–d10), texto exacto del enunciado
# ----------------------------------------------------------------------
doc_ids = ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10"]
doc_texts = [
    "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión.",
    "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización.",
    "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo.",
    "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento.",
    "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación.",
    "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado.",
    "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos.",
    "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales.",
    "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad.",
    "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
]

# ----------------------------------------------------------------------
# 2. Consultas (q1–q6) y juicios de relevancia (un documento relevante cada una)
# ----------------------------------------------------------------------
query_ids = ["q1", "q2", "q3", "q4", "q5", "q6"]
query_texts = [
    "¿Qué función de ranking léxica pondera la frecuencia de términos?",
    "¿Cómo se añaden fragmentos recuperados al prompt?",
    "¿Qué técnica entrena matrices de bajo rango?",
    "¿Cómo se compara la orientación de dos vectores de texto?",
    "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
    "¿Qué arquitectura combina autoatención con capas feed-forward?",
]
relevant = {"q1": "d04", "q2": "d03", "q3": "d08", "q4": "d05", "q5": "d09", "q6": "d02"}

# ----------------------------------------------------------------------
# 3. Vectorización TF-IDF (parámetros por defecto) y similitud coseno
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parámetros por defecto, sin ajuste sobre consultas
X_docs = vectorizer.fit_transform(doc_texts)      # ajuste solo con el corpus
X_queries = vectorizer.transform(query_texts)     # mismas columnas/vocabulario

S = cosine_similarity(X_queries, X_docs)          # matriz (6 consultas x 10 docs)

# ----------------------------------------------------------------------
# 4. Ranking de los 10 documentos por consulta y métricas por consulta
# ----------------------------------------------------------------------
per_query = {}
hits1, hits3, reciprocal_ranks = [], [], []

for i, qid in enumerate(query_ids):
    # Orden descendente por similitud; desempate determinista por índice de documento
    order = np.argsort(-S[i], kind="stable")
    ranked_ids = [doc_ids[j] for j in order]

    rel = relevant[qid]
    rank_pos = ranked_ids.index(rel) + 1  # posición 1-based del relevante

    h1 = 1.0 if rank_pos == 1 else 0.0
    h3 = 1.0 if rank_pos <= 3 else 0.0
    rr = 1.0 / rank_pos

    hits1.append(h1)
    hits3.append(h3)
    reciprocal_ranks.append(rr)

    per_query[qid] = {
        "consulta": query_texts[i],
        "relevante": rel,
        "rank_relevante": float(rank_pos),
        "hit_at_1": float(h1),
        "hit_at_3": float(h3),
        "reciprocal_rank": float(rr),
        "ranking_documentos": ranked_ids,
        "similitud_coseno": {doc_ids[j]: float(S[i, j]) for j in range(len(doc_ids))},
    }

# ----------------------------------------------------------------------
# 5. Métricas agregadas sobre las 6 consultas
# ----------------------------------------------------------------------
hit_at_1 = float(np.mean(hits1))
hit_at_3 = float(np.mean(hits3))
mrr = float(np.mean(reciprocal_ranks))

# ----------------------------------------------------------------------
# 6. Guardar TODAS las cifras en resultados.json (floats de Python, sin redondear)
# ----------------------------------------------------------------------
resultados = {
    "tarea": "T1_linea_base_lexica_tfidf_coseno",
    "metricas_globales": {
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "mrr": mrr,
        "num_documentos": float(len(doc_ids)),
        "num_consultas": float(len(query_ids)),
        "tamano_vocabulario_tfidf": float(len(vectorizer.vocabulary_)),
    },
    "por_consulta": per_query,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7. Imprimir cifras principales con 4 decimales
# ----------------------------------------------------------------------
print("=== T1: Línea base léxica (TF-IDF por defecto + similitud coseno) ===")
print(f"Hit@1: {hit_at_1:.4f}")
print(f"Hit@3: {hit_at_3:.4f}")
print(f"MRR:   {mrr:.4f}")
print("\nRanking por consulta (top-3) y posición del documento relevante:")
for qid in query_ids:
    pq = per_query[qid]
    top3 = ", ".join(pq["ranking_documentos"][:3])
    print(
        f"{qid}: top-3 = [{top3}] | relevante {pq['relevante']} en rank "
        f"{pq['rank_relevante']:.0f} | RR = {pq['reciprocal_rank']:.4f}"
    )
