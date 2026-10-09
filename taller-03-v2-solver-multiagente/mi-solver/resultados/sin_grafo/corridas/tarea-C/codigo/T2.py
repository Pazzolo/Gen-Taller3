# -*- coding: utf-8 -*-
"""
T2 - Parte 2: Semantica latente (LSA) sobre la linea base TF-IDF de la Parte 1.

Script autocontenido: reconstruye la Parte 1 (TF-IDF con parametros por
defecto + similitud coseno, evaluada con Hit@1, Hit@3 y MRR sobre las 6
consultas con los mismos juicios de relevancia) y despues:
  1) proyecta la matriz TF-IDF de los 10 documentos a 4 dimensiones con
     TruncatedSVD(n_components=4, random_state=0) (LSA);
  2) proyecta las consultas al mismo espacio latente (fold-in con transform)
     y repite exactamente la misma evaluacion;
  3) imprime la tabla conjunta con las tres metricas de los dos metodos y
     guarda todas las cifras calculadas en "resultados.json".

No se genera ninguna figura (la subtarea indica: Figura: no).
"""

import json

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1) Datos de la tarea: los mismos 10 documentos, las mismas 6 consultas y los
#    mismos juicios de relevancia de la Parte 1 (se re-declaran aqui para que
#    el script sea independiente).
# ---------------------------------------------------------------------------
documentos = [
    "La recuperación de información clásica ordena los documentos con el modelo vectorial.",
    "TF-IDF pondera los términos según su frecuencia en el documento y su rareza en la colección.",
    "El índice invertido mapea cada término a la lista de documentos que lo contienen.",
    "La similitud coseno compara el ángulo entre los vectores de la consulta y del documento.",
    "La segmentación de palabras y la eliminación de palabras vacías preparan el texto antes de indexar.",
    "La descomposición en valores singulares comprime la matriz término-documento en un espacio latente.",
    "El análisis semántico latente agrupa términos que aparecen en contextos parecidos, como sinónimos.",
    "Los embeddings densos representan consultas y documentos en vectores de baja dimensión.",
    "La búsqueda vectorial con embeddings recupera resultados aunque la consulta no comparta términos con el documento.",
    "Los sistemas RAG combinan un recuperador de documentos con un modelo de lenguaje que genera la respuesta.",
]

consultas = [
    "recuperación de documentos con el modelo vectorial",
    "ponderación de términos según su frecuencia",
    "índice invertido con listas de documentos por término",
    "similitud coseno entre vectores de consulta y documento",
    "palabras con significado parecido",
    "buscador que responde con un modelo generativo",
]

# Juicios de relevancia: para cada consulta, indices (0-based) de los
# documentos considerados relevantes.
juicios = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [6],
    5: [9],
}


# ---------------------------------------------------------------------------
# 2) Evaluacion comun: ranking por similitud coseno + Hit@1, Hit@3 y MRR
#    (Hit@k = proporcion de consultas con algun relevante en el top-k;
#     MRR = media del recíproco del rango del primer documento relevante).
# ---------------------------------------------------------------------------
def evaluar(sims, juicios):
    """Rankea los 10 documentos para cada consulta y calcula las metricas.

    sims: matriz (n_consultas x n_documentos) de similitudes coseno.
    Devuelve: rankings, rango (1-based) del primer relevante por consulta
    (0 si no aparece), listas por consulta de Hit@1/Hit@3/RR y metricas
    agregadas (promediadas sobre las consultas).
    """
    rankings = [np.argsort(-sims[q], kind="stable") for q in range(sims.shape[0])]
    ranks_primer, hit1, hit3, rr = [], [], [], []
    for q, orden in enumerate(rankings):
        relevantes = set(juicios[q])
        r = None
        for pos, doc in enumerate(orden, start=1):
            if int(doc) in relevantes:
                r = pos
                break
        ranks_primer.append(int(r) if r is not None else 0)
        hit1.append(1.0 if r == 1 else 0.0)
        hit3.append(1.0 if (r is not None and r <= 3) else 0.0)
        rr.append(1.0 / r if r is not None else 0.0)
    metricas = {
        "hit_at_1": float(np.mean(hit1)),
        "hit_at_3": float(np.mean(hit3)),
        "mrr": float(np.mean(rr)),
    }
    return rankings, ranks_primer, hit1, hit3, rr, metricas


# ---------------------------------------------------------------------------
# 3) Parte 1 (linea base): TF-IDF con parametros por defecto + coseno
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parametros por defecto, como en la Parte 1
X_docs = vectorizer.fit_transform(documentos)   # matriz TF-IDF (10 x V)
X_queries = vectorizer.transform(consultas)     # mismas columnas/vocabulario

sims_tfidf = cosine_similarity(X_queries, X_docs)
rank_tfidf, pr_tfidf, h1_t, h3_t, rr_t, m_tfidf = evaluar(sims_tfidf, juicios)

# ---------------------------------------------------------------------------
# 4) Parte 2: LSA - proyeccion de la matriz TF-IDF a 4 dimensiones
# ---------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)     # documentos en el espacio latente
X_queries_lsa = svd.transform(X_queries)   # consultas proyectadas (fold-in)

sims_lsa = cosine_similarity(X_queries_lsa, X_docs_lsa)
rank_lsa, pr_lsa, h1_l, h3_l, rr_l, m_lsa = evaluar(sims_lsa, juicios)

# ---------------------------------------------------------------------------
# 5) Tabla conjunta y cifras principales (4 decimales)
# ---------------------------------------------------------------------------
var_exp = [float(v) for v in svd.explained_variance_ratio_]
var_total = float(np.sum(svd.explained_variance_ratio_))

print("T2 - Parte 2: LSA (TruncatedSVD, 4 componentes) frente a la linea base TF-IDF")
print(f"Documentos: {X_docs.shape[0]} | Consultas: {X_queries.shape[0]} | "
      f"Vocabulario TF-IDF: {len(vectorizer.vocabulary_)} terminos")
print(f"Varianza explicada por las 4 componentes LSA: {var_total:.4f}")
print()
print("Tabla conjunta (metricas promediadas sobre las 6 consultas):")
print(f"{'Metodo':<20}{'Hit@1':>10}{'Hit@3':>10}{'MRR':>10}")
print(f"{'TF-IDF + coseno':<20}{m_tfidf['hit_at_1']:>10.4f}"
      f"{m_tfidf['hit_at_3']:>10.4f}{m_tfidf['mrr']:>10.4f}")
print(f"{'LSA-4D + coseno':<20}{m_lsa['hit_at_1']:>10.4f}"
      f"{m_lsa['hit_at_3']:>10.4f}{m_lsa['mrr']:>10.4f}")
print()
print("Rango (1-based) del primer documento relevante por consulta:")
print(f"  TF-IDF : {pr_tfidf}")
print(f"  LSA-4D : {pr_lsa}")

# ---------------------------------------------------------------------------
# 6) Guardado de todas las cifras calculadas en resultados.json
# ---------------------------------------------------------------------------
resultados = {
    "n_documentos": int(X_docs.shape[0]),
    "n_consultas": int(X_queries.shape[0]),
    "vocabulario_tamano": int(len(vectorizer.vocabulary_)),
    "lsa_n_componentes": 4,
    "lsa_random_state": 0,
    "lsa_varianza_explicada_por_componente": var_exp,
    "lsa_varianza_explicada_total": var_total,
    "tfidf_hit_at_1": m_tfidf["hit_at_1"],
    "tfidf_hit_at_3": m_tfidf["hit_at_3"],
    "tfidf_mrr": m_tfidf["mrr"],
    "lsa_hit_at_1": m_lsa["hit_at_1"],
    "lsa_hit_at_3": m_lsa["hit_at_3"],
    "lsa_mrr": m_lsa["mrr"],
    "tfidf_rank_primer_relevante_por_consulta": pr_tfidf,
    "lsa_rank_primer_relevante_por_consulta": pr_lsa,
    "tfidf_hit1_por_consulta": h1_t,
    "tfidf_hit3_por_consulta": h3_t,
    "tfidf_rr_por_consulta": rr_t,
    "lsa_hit1_por_consulta": h1_l,
    "lsa_hit3_por_consulta": h3_l,
    "lsa_rr_por_consulta": rr_l,
    "tfidf_ranking_por_consulta": [[int(d) for d in r] for r in rank_tfidf],
    "lsa_ranking_por_consulta": [[int(d) for d in r] for r in rank_lsa],
    "tfidf_similitudes_consulta_documento": [[float(v) for v in fila] for fila in sims_tfidf],
    "lsa_similitudes_consulta_documento": [[float(v) for v in fila] for fila in sims_lsa],
    "tabla_conjunta": {
        "TF-IDF + coseno": {
            "Hit@1": m_tfidf["hit_at_1"],
            "Hit@3": m_tfidf["hit_at_3"],
            "MRR": m_tfidf["mrr"],
        },
        "LSA-4D + coseno": {
            "Hit@1": m_lsa["hit_at_1"],
            "Hit@3": m_lsa["hit_at_3"],
            "MRR": m_lsa["mrr"],
        },
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

print()
print("Cifras guardadas en resultados.json")
