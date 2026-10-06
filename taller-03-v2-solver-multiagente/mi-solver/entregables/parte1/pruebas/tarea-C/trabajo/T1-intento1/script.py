# -*- coding: utf-8 -*-
"""
Subtarea T1 — Línea base léxica sobre el corpus mínimo (d01–d10).

Método:
  - TfidfVectorizer de scikit-learn con parámetros por defecto (minúsculas,
    token_pattern por defecto, sin strip_accents, norm l2).
  - Similitud coseno entre las 6 consultas (q1–q6) y los 10 documentos.
  - Por consulta: ordenar los 10 documentos por similitud descendente.
  - Evaluación con los juicios dados (q1→d04, q2→d03, q3→d08, q4→d05,
    q5→d09, q6→d02): Hit@1, Hit@3 y MRR.

Salidas:
  - resultados.json (un objeto con TODAS las cifras calculadas, sin redondear).
  - Impresión de las cifras principales con 4 decimales.
  - Sin figura (la subtarea indica "Figura: no").
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# 1. Datos del enunciado (corpus, consultas y juicios de relevancia)
# ----------------------------------------------------------------------------
doc_ids = ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10"]

corpus = [
    "El mecanismo de atención pondera cada token según su similitud con la consulta; "
    "la atención escalada divide por la raíz de la dimensión.",
    "Los transformadores apilan capas de autoatención y redes feed-forward, con "
    "conexiones residuales y normalización.",
    "La recuperación aumentada con generación busca fragmentos relevantes y los añade "
    "al prompt del modelo.",
    "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la "
    "longitud del documento.",
    "Los embeddings densos representan textos como vectores; la similitud coseno "
    "compara su orientación.",
    "Un agente con herramientas decide en cada paso qué función llamar y observa el "
    "resultado.",
    "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal "
    "entre intentos.",
    "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos "
    "originales.",
    "La temperatura reescala los logits antes del softmax; valores bajos concentran "
    "la probabilidad.",
    "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar "
    "memoria.",
]

query_ids = ["q1", "q2", "q3", "q4", "q5", "q6"]

queries = [
    "¿Qué función de ranking léxica pondera la frecuencia de términos?",
    "¿Cómo se añaden fragmentos recuperados al prompt?",
    "¿Qué técnica entrena matrices de bajo rango?",
    "¿Cómo se compara la orientación de dos vectores de texto?",
    "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
    "¿Qué arquitectura combina autoatención con capas feed-forward?",
]

relevant = {
    "q1": "d04",
    "q2": "d03",
    "q3": "d08",
    "q4": "d05",
    "q5": "d09",
    "q6": "d02",
}

# ----------------------------------------------------------------------------
# 2. Vectorización TF-IDF (parámetros por defecto) y similitud coseno
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # defaults: lowercase=True, token_pattern=(?u)\b\w\w+\b
X_docs = vectorizer.fit_transform(corpus)    # matriz (10 x V)
X_queries = vectorizer.transform(queries)    # matriz (6 x V), mismo vocabulario

S = cosine_similarity(X_queries, X_docs)     # matriz (6 x 10)

# ----------------------------------------------------------------------------
# 3. Ranking por consulta y evaluación (Hit@1, Hit@3, MRR)
# ----------------------------------------------------------------------------
por_consulta = {}
hit1_list, hit3_list, rr_list = [], [], []

for i, qid in enumerate(query_ids):
    sims = S[i]
    # Orden descendente por similitud; desempates deterministas por índice de documento
    order = np.argsort(-sims, kind="stable")
    ranked_ids = [doc_ids[j] for j in order]
    ranked_sims = [float(sims[j]) for j in order]

    rel = relevant[qid]
    rank = ranked_ids.index(rel) + 1  # posición (1-based) del documento relevante

    hit1 = 1.0 if rank <= 1 else 0.0
    hit3 = 1.0 if rank <= 3 else 0.0
    rr = 1.0 / float(rank)

    hit1_list.append(hit1)
    hit3_list.append(hit3)
    rr_list.append(rr)

    por_consulta[qid] = {
        "consulta": queries[i],
        "documento_relevante": rel,
        "ranking_documentos": ranked_ids,
        "similitudes_ordenadas": ranked_sims,
        "rank_del_relevante": int(rank),
        "similitud_con_relevante": float(sims[doc_ids.index(rel)]),
        "hit_at_1": float(hit1),
        "hit_at_3": float(hit3),
        "reciprocal_rank": float(rr),
    }

hit_at_1 = float(np.mean(hit1_list))
hit_at_3 = float(np.mean(hit3_list))
mrr = float(np.mean(rr_list))

# ----------------------------------------------------------------------------
# 4. Matriz completa de similitud coseno (consulta x documento)
# ----------------------------------------------------------------------------
matriz_similitud = {
    qid: {did: float(S[i, j]) for j, did in enumerate(doc_ids)}
    for i, qid in enumerate(query_ids)
}

# ----------------------------------------------------------------------------
# 5. Guardar TODAS las cifras en resultados.json (un objeto, floats sin redondear)
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T1_linea_base_lexica_tfidf",
    "descripcion": "TF-IDF (TfidfVectorizer por defecto) + similitud coseno; "
                   "ranking de 10 documentos por consulta; Hit@1, Hit@3 y MRR",
    "num_documentos": int(len(doc_ids)),
    "num_consultas": int(len(query_ids)),
    "tamano_vocabulario": int(len(vectorizer.vocabulary_)),
    "hit_at_1": hit_at_1,
    "hit_at_3": hit_at_3,
    "mrr": mrr,
    "metricas_globales": {
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "mrr": mrr,
    },
    "por_consulta": por_consulta,
    "matriz_similitud_coseno": matriz_similitud,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 6. Impresión de las cifras principales (4 decimales)
# ----------------------------------------------------------------------------
print("=== T1: Línea base léxica (TF-IDF por defecto + coseno) ===")
print(f"Tamaño del vocabulario: {len(vectorizer.vocabulary_)}")
print("-" * 64)
for qid in query_ids:
    pq = por_consulta[qid]
    top3 = ", ".join(pq["ranking_documentos"][:3])
    print(
        f"{qid}: relevante={pq['documento_relevante']}  "
        f"rank={pq['rank_del_relevante']}  top-3=[{top3}]  "
        f"RR={pq['reciprocal_rank']:.4f}"
    )
print("-" * 64)
print(f"Hit@1 = {hit_at_1:.4f}")
print(f"Hit@3 = {hit_at_3:.4f}")
print(f"MRR   = {mrr:.4f}")
print("Resultados guardados en resultados.json")
