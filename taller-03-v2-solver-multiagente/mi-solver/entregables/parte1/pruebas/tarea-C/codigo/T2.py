# -*- coding: utf-8 -*-
"""
T2 - Semantica latente: LSA (TruncatedSVD, 4 componentes) frente a la linea
base lexica TF-IDF de T1 sobre el corpus minimo de la Tarea C.

Pasos:
  1) Recalcula de forma independiente la linea base de T1
     (TfidfVectorizer por defecto + similitud coseno).
  2) Proyecta la matriz TF-IDF a 4 dimensiones con
     TruncatedSVD(n_components=4, random_state=0) (LSA). El SVD se ajusta
     SOLO con los documentos; las consultas solo se proyectan.
  3) Repite la misma evaluacion (Hit@1, Hit@3, MRR) sobre las mismas 6
     consultas y juicios de relevancia.
  4) Construye la tabla conjunta con las tres metricas de los dos metodos.

Salidas:
  - resultados.json con todas las cifras (float de Python, sin redondear)
  - impresion de las cifras principales con 4 decimales
  - sin figura (la subtarea indica "Figura: no")
"""

import json

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. Datos del enunciado
# ---------------------------------------------------------------------------
corpus_ids = ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10"]

corpus_texts = [
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

query_ids = ["q1", "q2", "q3", "q4", "q5", "q6"]

query_texts = [
    "¿Qué función de ranking léxica pondera la frecuencia de términos?",
    "¿Cómo se añaden fragmentos recuperados al prompt?",
    "¿Qué técnica entrena matrices de bajo rango?",
    "¿Cómo se compara la orientación de dos vectores de texto?",
    "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?",
    "¿Qué arquitectura combina autoatención con capas feed-forward?",
]

relevante = {
    "q1": "d04",
    "q2": "d03",
    "q3": "d08",
    "q4": "d05",
    "q5": "d09",
    "q6": "d02",
}

# Referencia reportada por T1 (solo para verificacion; todo se recalcula aqui)
referencia_t1 = {
    "hit_at_1": 0.8333333333333334,
    "hit_at_3": 0.8333333333333334,
    "mrr": 0.8611111111111112,
}


# ---------------------------------------------------------------------------
# 2. Evaluacion: ranking de los 10 documentos por consulta + metricas
# ---------------------------------------------------------------------------
def evaluar(similitudes):
    """Evalua una matriz de similitud (6 consultas x 10 documentos).

    Ordena los diez documentos por similitud descendente (empates rotos por
    orden de id: d01 antes que d02, ...) y calcula Hit@1, Hit@3 y MRR.
    """
    por_consulta = {}
    hits1, hits3, reciprocos = [], [], []
    for i, qid in enumerate(query_ids):
        fila = np.asarray(similitudes[i], dtype=float)
        orden = np.argsort(-fila, kind="stable")
        ranking = [corpus_ids[j] for j in orden]
        sims_ordenadas = [float(fila[j]) for j in orden]
        rel = relevante[qid]
        rank = ranking.index(rel) + 1
        h1 = 1.0 if rank <= 1 else 0.0
        h3 = 1.0 if rank <= 3 else 0.0
        rr = 1.0 / float(rank)
        hits1.append(h1)
        hits3.append(h3)
        reciprocos.append(rr)
        por_consulta[qid] = {
            "consulta": query_texts[i],
            "documento_relevante": rel,
            "ranking_documentos": ranking,
            "similitudes_ordenadas": sims_ordenadas,
            "rank_del_relevante": int(rank),
            "similitud_con_relevante": float(fila[corpus_ids.index(rel)]),
            "hit_at_1": h1,
            "hit_at_3": h3,
            "reciprocal_rank": rr,
        }
    metricas = {
        "hit_at_1": float(np.mean(hits1)),
        "hit_at_3": float(np.mean(hits3)),
        "mrr": float(np.mean(reciprocos)),
    }
    return metricas, por_consulta


# ---------------------------------------------------------------------------
# 3. Linea base lexica (recalculo independiente de T1): TF-IDF + coseno
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # parametros por defecto, como en T1
X_docs_tfidf = vectorizer.fit_transform(corpus_texts)  # ajuste SOLO con el corpus
X_queries_tfidf = vectorizer.transform(query_texts)    # consultas transformadas

tamano_vocabulario = int(len(vectorizer.vocabulary_))

sim_tfidf = cosine_similarity(X_queries_tfidf, X_docs_tfidf)
metricas_tfidf, por_consulta_tfidf = evaluar(sim_tfidf)

# ---------------------------------------------------------------------------
# 4. Semantica latente (LSA): TruncatedSVD a 4 dimensiones
# ---------------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs_tfidf)    # se ajusta SOLO con los documentos
X_queries_lsa = svd.transform(X_queries_tfidf)  # las consultas solo se proyectan

sim_lsa = cosine_similarity(X_queries_lsa, X_docs_lsa)
metricas_lsa, por_consulta_lsa = evaluar(sim_lsa)

varianza_explicada = [float(v) for v in svd.explained_variance_ratio_]
varianza_total = float(sum(varianza_explicada))

# ---------------------------------------------------------------------------
# 5. Tabla conjunta con las tres metricas de los dos metodos
# ---------------------------------------------------------------------------
tabla_conjunta = [
    {
        "metodo": "TF-IDF (linea base lexica, T1)",
        "hit_at_1": metricas_tfidf["hit_at_1"],
        "hit_at_3": metricas_tfidf["hit_at_3"],
        "mrr": metricas_tfidf["mrr"],
    },
    {
        "metodo": "LSA (TruncatedSVD n_components=4, random_state=0)",
        "hit_at_1": metricas_lsa["hit_at_1"],
        "hit_at_3": metricas_lsa["hit_at_3"],
        "mrr": metricas_lsa["mrr"],
    },
]

filas_md = ["| Metodo | Hit@1 | Hit@3 | MRR |", "|---|---|---|---|"]
for fila in tabla_conjunta:
    filas_md.append(
        "| {} | {:.4f} | {:.4f} | {:.4f} |".format(
            fila["metodo"], fila["hit_at_1"], fila["hit_at_3"], fila["mrr"]
        )
    )
tabla_md = "\n".join(filas_md)

diferencias = {
    "hit_at_1": metricas_lsa["hit_at_1"] - metricas_tfidf["hit_at_1"],
    "hit_at_3": metricas_lsa["hit_at_3"] - metricas_tfidf["hit_at_3"],
    "mrr": metricas_lsa["mrr"] - metricas_tfidf["mrr"],
}

verificacion_t1 = {
    "hit_at_1_coincide": bool(abs(metricas_tfidf["hit_at_1"] - referencia_t1["hit_at_1"]) < 1e-12),
    "hit_at_3_coincide": bool(abs(metricas_tfidf["hit_at_3"] - referencia_t1["hit_at_3"]) < 1e-12),
    "mrr_coincide": bool(abs(metricas_tfidf["mrr"] - referencia_t1["mrr"]) < 1e-12),
}

# ---------------------------------------------------------------------------
# 6. Guardar todas las cifras en resultados.json
# ---------------------------------------------------------------------------
resultados = {
    "subtarea": "T2_semantica_latente_lsa",
    "descripcion": (
        "LSA: TruncatedSVD(n_components=4, random_state=0) sobre la matriz "
        "TF-IDF; misma evaluacion (Hit@1, Hit@3, MRR) sobre las mismas 6 "
        "consultas y juicios; tabla conjunta de los dos metodos."
    ),
    "num_documentos": len(corpus_ids),
    "num_consultas": len(query_ids),
    "tamano_vocabulario": tamano_vocabulario,
    "n_componentes_lsa": 4,
    "random_state_svd": 0,
    "tfidf_linea_base": {
        "hit_at_1": metricas_tfidf["hit_at_1"],
        "hit_at_3": metricas_tfidf["hit_at_3"],
        "mrr": metricas_tfidf["mrr"],
        "por_consulta": por_consulta_tfidf,
    },
    "lsa": {
        "hit_at_1": metricas_lsa["hit_at_1"],
        "hit_at_3": metricas_lsa["hit_at_3"],
        "mrr": metricas_lsa["mrr"],
        "varianza_explicada_por_componente": varianza_explicada,
        "varianza_explicada_total": varianza_total,
        "por_consulta": por_consulta_lsa,
    },
    "tabla_conjunta": tabla_conjunta,
    "tabla_conjunta_markdown": tabla_md,
    "diferencias_lsa_menos_tfidf": diferencias,
    "verificacion_contra_t1": verificacion_t1,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 7. Impresion de las cifras principales con 4 decimales
# ---------------------------------------------------------------------------
print("T2 - Semantica latente (LSA, 4 componentes) vs linea base TF-IDF")
print("")
print("Tabla conjunta (6 consultas, 10 documentos):")
print(tabla_md)
print("")
print("Diferencias (LSA - TF-IDF):")
print("  Hit@1: {:+.4f}".format(diferencias["hit_at_1"]))
print("  Hit@3: {:+.4f}".format(diferencias["hit_at_3"]))
print("  MRR  : {:+.4f}".format(diferencias["mrr"]))
print("")
print("Varianza explicada por LSA (4 componentes): {:.4f}".format(varianza_total))
print("")
print("Rank del documento relevante por consulta (TF-IDF -> LSA):")
for qid in query_ids:
    print(
        "  {}: {:d} -> {:d}".format(
            qid,
            por_consulta_tfidf[qid]["rank_del_relevante"],
            por_consulta_lsa[qid]["rank_del_relevante"],
        )
    )
print("")
print("Verificacion del recalculo de la linea base T1:")
print(
    "  Hit@1={:.4f} (ref {:.4f}, coincide: {}), ".format(
        metricas_tfidf["hit_at_1"],
        referencia_t1["hit_at_1"],
        verificacion_t1["hit_at_1_coincide"],
    )
    + "Hit@3={:.4f} (ref {:.4f}, coincide: {}), ".format(
        metricas_tfidf["hit_at_3"],
        referencia_t1["hit_at_3"],
        verificacion_t1["hit_at_3_coincide"],
    )
    + "MRR={:.4f} (ref {:.4f}, coincide: {})".format(
        metricas_tfidf["mrr"],
        referencia_t1["mrr"],
        verificacion_t1["mrr_coincide"],
    )
)
print("")
print("Resultados guardados en resultados.json")
