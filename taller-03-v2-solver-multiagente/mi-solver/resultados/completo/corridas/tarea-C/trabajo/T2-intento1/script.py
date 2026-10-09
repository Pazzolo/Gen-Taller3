# -*- coding: utf-8 -*-
"""
Subtarea T2 — Semántica latente (LSA) sobre la matriz TF-IDF del corpus mínimo.

Reutiliza la misma matriz TF-IDF de la Parte 1 (TfidfVectorizer con parámetros
por defecto, ajustado solo sobre los diez documentos) y la proyecta a 4
dimensiones con TruncatedSVD(n_components=4, random_state=0). Repite la
evaluación (Hit@1, Hit@3, MRR) sobre las seis consultas con sus juicios de
relevancia y construye UNA sola tabla con las tres métricas de los dos métodos
(TF-IDF y LSA). La línea base TF-IDF se recalcula aquí para que el script sea
independiente. No genera figura. Guarda todas las cifras en resultados.json.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# 1) Datos del enunciado: corpus, consultas y juicios de relevancia
# ----------------------------------------------------------------------
corpus_ids = ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09", "d10"]

corpus_textos = [
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

consultas = [
    {"id": "q1", "texto": "¿Qué función de ranking léxica pondera la frecuencia de términos?", "relevante": "d04"},
    {"id": "q2", "texto": "¿Cómo se añaden fragmentos recuperados al prompt?", "relevante": "d03"},
    {"id": "q3", "texto": "¿Qué técnica entrena matrices de bajo rango?", "relevante": "d08"},
    {"id": "q4", "texto": "¿Cómo se compara la orientación de dos vectores de texto?", "relevante": "d05"},
    {"id": "q5", "texto": "¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "relevante": "d09"},
    {"id": "q6", "texto": "¿Qué arquitectura combina autoatención con capas feed-forward?", "relevante": "d02"},
]

# ----------------------------------------------------------------------
# 2) Matriz TF-IDF (parámetros por defecto, ajustada SOLO con el corpus)
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()
X_docs = vectorizer.fit_transform(corpus_textos)
X_consultas = vectorizer.transform([c["texto"] for c in consultas])
tam_vocab = len(vectorizer.vocabulary_)

# ----------------------------------------------------------------------
# 3) Línea base TF-IDF + coseno (recalculada: script independiente)
# ----------------------------------------------------------------------
sim_tfidf = cosine_similarity(X_consultas, X_docs)

# ----------------------------------------------------------------------
# 4) LSA: TruncatedSVD a 4 dimensiones sobre la MISMA matriz TF-IDF
# ----------------------------------------------------------------------
svd = TruncatedSVD(n_components=4, random_state=0)
X_docs_lsa = svd.fit_transform(X_docs)          # documentos en el espacio latente
X_consultas_lsa = svd.transform(X_consultas)    # consultas proyectadas al mismo espacio
sim_lsa = cosine_similarity(X_consultas_lsa, X_docs_lsa)

varianza_explicada = svd.explained_variance_ratio_
varianza_acumulada = float(np.sum(varianza_explicada))

# ----------------------------------------------------------------------
# 5) Evaluación: Hit@1, Hit@3 y MRR (desempate estable por orden del corpus)
# ----------------------------------------------------------------------
def evaluar(matriz_sim, consultas, corpus_ids):
    por_consulta = {}
    hits1, hits3, rranks = [], [], []
    for i, c in enumerate(consultas):
        sims = matriz_sim[i]
        orden = np.argsort(-sims, kind="stable")  # descendente; empates -> orden del corpus
        ranking = [corpus_ids[j] for j in orden]
        rank = ranking.index(c["relevante"]) + 1
        h1 = 1.0 if rank == 1 else 0.0
        h3 = 1.0 if rank <= 3 else 0.0
        rr = 1.0 / rank
        hits1.append(h1)
        hits3.append(h3)
        rranks.append(rr)
        por_consulta[c["id"]] = {
            "consulta": c["texto"],
            "relevante": c["relevante"],
            "rank_relevante": float(rank),
            "hit_at_1": float(h1),
            "hit_at_3": float(h3),
            "reciprocal_rank": float(rr),
            "ranking_documentos": ranking,
            "similitudes": {corpus_ids[j]: float(sims[j]) for j in range(len(corpus_ids))},
        }
    metricas = {
        "hit_at_1": float(np.mean(hits1)),
        "hit_at_3": float(np.mean(hits3)),
        "mrr": float(np.mean(rranks)),
    }
    return metricas, por_consulta

metricas_tfidf, por_consulta_tfidf = evaluar(sim_tfidf, consultas, corpus_ids)
metricas_lsa, por_consulta_lsa = evaluar(sim_lsa, consultas, corpus_ids)

# ----------------------------------------------------------------------
# 6) Tabla única con las tres métricas de los dos métodos
# ----------------------------------------------------------------------
tabla = [
    {
        "metodo": "TF-IDF (coseno)",
        "hit_at_1": metricas_tfidf["hit_at_1"],
        "hit_at_3": metricas_tfidf["hit_at_3"],
        "mrr": metricas_tfidf["mrr"],
    },
    {
        "metodo": "LSA-4 (TruncatedSVD, coseno)",
        "hit_at_1": metricas_lsa["hit_at_1"],
        "hit_at_3": metricas_lsa["hit_at_3"],
        "mrr": metricas_lsa["mrr"],
    },
]

diferencias = {
    "hit_at_1": float(metricas_lsa["hit_at_1"] - metricas_tfidf["hit_at_1"]),
    "hit_at_3": float(metricas_lsa["hit_at_3"] - metricas_tfidf["hit_at_3"]),
    "mrr": float(metricas_lsa["mrr"] - metricas_tfidf["mrr"]),
}

# ----------------------------------------------------------------------
# 7) Guardar TODAS las cifras calculadas en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "tarea": "T2_semantica_latente_lsa_truncatedsvd4",
    "configuracion": {
        "n_componentes_lsa": 4.0,
        "random_state_svd": 0.0,
        "num_documentos": float(len(corpus_ids)),
        "num_consultas": float(len(consultas)),
        "tamano_vocabulario_tfidf": float(tam_vocab),
    },
    "tabla_metricas": tabla,
    "metricas_tfidf": metricas_tfidf,
    "metricas_lsa": metricas_lsa,
    "diferencia_lsa_menos_tfidf": diferencias,
    "varianza_explicada_por_componente_lsa": [float(v) for v in varianza_explicada],
    "varianza_explicada_acumulada_lsa": varianza_acumulada,
    "vectores_latentes_documentos_lsa": {
        corpus_ids[i]: [float(x) for x in X_docs_lsa[i]] for i in range(len(corpus_ids))
    },
    "vectores_latentes_consultas_lsa": {
        consultas[i]["id"]: [float(x) for x in X_consultas_lsa[i]] for i in range(len(consultas))
    },
    "por_consulta_tfidf": por_consulta_tfidf,
    "por_consulta_lsa": por_consulta_lsa,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 8) Impresión de las cifras principales (4 decimales)
# ----------------------------------------------------------------------
print("T2 — TF-IDF vs LSA (TruncatedSVD, n_components=4, random_state=0)")
print(f"Documentos: {len(corpus_ids)} | Consultas: {len(consultas)} | Vocabulario TF-IDF: {tam_vocab}")
print()
print("Tabla única de métricas (6 consultas):")
encabezado = f"{'Método':<32}{'Hit@1':>10}{'Hit@3':>10}{'MRR':>10}"
print(encabezado)
print("-" * len(encabezado))
for fila in tabla:
    print(f"{fila['metodo']:<32}{fila['hit_at_1']:>10.4f}{fila['hit_at_3']:>10.4f}{fila['mrr']:>10.4f}")
print()
print("Varianza explicada por componente (LSA): "
      + ", ".join(f"{v:.4f}" for v in varianza_explicada))
print(f"Varianza explicada acumulada (4 componentes): {varianza_acumulada:.4f}")
print()
print("Por consulta — LSA:")
for c in consultas:
    pc = por_consulta_lsa[c["id"]]
    print(f"  {c['id']}: relevante={pc['relevante']} rank={pc['rank_relevante']:.0f} "
          f"Hit@1={pc['hit_at_1']:.4f} Hit@3={pc['hit_at_3']:.4f} RR={pc['reciprocal_rank']:.4f}")
print()
print("Por consulta — TF-IDF (línea base):")
for c in consultas:
    pc = por_consulta_tfidf[c["id"]]
    print(f"  {c['id']}: relevante={pc['relevante']} rank={pc['rank_relevante']:.0f} "
          f"Hit@1={pc['hit_at_1']:.4f} Hit@3={pc['hit_at_3']:.4f} RR={pc['reciprocal_rank']:.4f}")
print()
print(f"Diferencias (LSA - TF-IDF): Hit@1={diferencias['hit_at_1']:.4f} "
      f"Hit@3={diferencias['hit_at_3']:.4f} MRR={diferencias['mrr']:.4f}")
print("Resultados guardados en resultados.json")
