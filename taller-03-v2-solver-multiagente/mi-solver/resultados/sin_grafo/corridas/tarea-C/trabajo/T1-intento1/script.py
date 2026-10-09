# -*- coding: utf-8 -*-
"""
T1 - Parte 1: Línea base léxica de recuperación de información.

Corpus: 10 documentos (d01-d10) y 6 consultas (q1-q6) del enunciado.
Juicios de relevancia (un documento relevante por consulta):
    q1 -> d04, q2 -> d03, q3 -> d08, q4 -> d05, q5 -> d09, q6 -> d02

Pipeline:
  1) TfidfVectorizer (parámetros por defecto), ajustado SOLO con los documentos
     (las consultas solo se transforman: nunca se ajusta nada con las consultas).
  2) Similitud coseno entre cada consulta y los 10 documentos.
  3) Ranking de los 10 documentos por similitud decreciente para cada consulta.
  4) Métricas sobre las 6 consultas: Hit@1, Hit@3 y MRR.

Salidas:
  - resultados.json : rankings, similitudes y métricas (todas las cifras calculadas).
  - stdout          : cifras principales con 4 decimales.
Figura: esta subtarea no pide figura.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# 1. Datos del enunciado: corpus (d01-d10) y consultas (q1-q6)
# ----------------------------------------------------------------------------
DOCS = {
    "d01": ("El aprendizaje supervisado entrena modelos con ejemplos etiquetados. "
            "Algoritmos como la regresión logística y los árboles de decisión ajustan "
            "parámetros para minimizar el error de predicción sobre los datos."),
    "d02": ("Las redes neuronales profundas se entrenan mediante retropropagación del "
            "error. Cada capa aplica una transformación lineal seguida de una función "
            "de activación no lineal, y el descenso de gradiente actualiza los pesos."),
    "d03": ("El procesamiento del lenguaje natural estudia cómo las máquinas comprenden "
            "y generan texto. Los modelos de lenguaje modernos se preentrenan sobre "
            "grandes corpus textuales y luego se ajustan a tareas específicas."),
    "d04": ("Los sistemas de recuperación de información ordenan documentos según su "
            "relevancia para una consulta. El modelo de espacio vectorial representa "
            "los textos como vectores y los compara con la similitud del coseno."),
    "d05": ("La visión por computadora permite a las máquinas interpretar imágenes. "
            "Las redes convolucionales aplican filtros que detectan bordes, texturas "
            "y formas para clasificar objetos en fotografías."),
    "d06": ("El aprendizaje por refuerzo entrena un agente que interactúa con un "
            "entorno. El agente aprende una política de acciones que maximiza la "
            "recompensa acumulada mediante prueba y error."),
    "d07": ("El aprendizaje no supervisado agrupa datos sin etiquetas. Los algoritmos "
            "de clustering como k-means particionan las observaciones en grupos según "
            "su distancia en el espacio de características."),
    "d08": ("El mecanismo de atención calcula la relevancia entre tokens mediante "
            "productos punto entre consultas, claves y valores. La función softmax "
            "normaliza los scores y cada token se representa como suma ponderada."),
    "d09": ("La generación aumentada por recuperación combina un buscador de documentos "
            "con un modelo generativo. El sistema recupera pasajes relevantes y el "
            "modelo los usa como contexto para redactar respuestas fundamentadas."),
    "d10": ("Los vectores de palabras representan términos en un espacio semántico "
            "continuo. Métodos como word2vec aprenden embeddings donde las palabras "
            "similares quedan cercanas tras reducir la dimensionalidad."),
}

QUERIES = {
    "q1": "búsqueda y ordenamiento de documentos por relevancia para una consulta",
    "q2": "modelos que comprenden y generan lenguaje natural",
    "q3": "atención con consultas, claves y valores normalizados con softmax",
    "q4": "clasificación de objetos en imágenes con filtros convolucionales",
    "q5": "sistemas que recuperan pasajes y generan respuestas con contexto",
    "q6": "entrenamiento de redes neuronales con retropropagación y descenso de gradiente",
}

RELEVANT = {"q1": "d04", "q2": "d03", "q3": "d08", "q4": "d05", "q5": "d09", "q6": "d02"}

doc_ids = sorted(DOCS.keys())      # d01 .. d10
query_ids = sorted(QUERIES.keys()) # q1 .. q6
n_docs = len(doc_ids)

# ----------------------------------------------------------------------------
# 2. Vectorización TF-IDF (parámetros por defecto)
#    fit solo con documentos; las consultas se transforman (no se ajustan).
# ----------------------------------------------------------------------------
vectorizer = TfidfVectorizer()
X_docs = vectorizer.fit_transform([DOCS[d] for d in doc_ids])   # (10 x V)
X_queries = vectorizer.transform([QUERIES[q] for q in query_ids])  # (6 x V)

# ----------------------------------------------------------------------------
# 3. Similitud coseno y ranking por consulta
# ----------------------------------------------------------------------------
S = cosine_similarity(X_queries, X_docs)  # matriz (6 x 10)

rankings = {}       # q -> lista de los 10 doc_ids ordenados por similitud decreciente
similitudes = {}    # q -> {doc_id: score de coseno}
for i, q in enumerate(query_ids):
    order = np.argsort(-S[i], kind="stable")  # decreciente; empates por orden d01..d10
    rankings[q] = [doc_ids[j] for j in order]
    similitudes[q] = {doc_ids[j]: float(S[i, j]) for j in range(n_docs)}

# ----------------------------------------------------------------------------
# 4. Métricas: Hit@1, Hit@3 y MRR (un documento relevante por consulta)
# ----------------------------------------------------------------------------
per_query = {}
for q in query_ids:
    rel = RELEVANT[q]
    rank = rankings[q].index(rel) + 1  # posición 1-based del relevante
    per_query[q] = {
        "documento_relevante": rel,
        "rank_relevante": int(rank),
        "hit@1": float(1.0 if rank == 1 else 0.0),
        "hit@3": float(1.0 if rank <= 3 else 0.0),
        "reciprocal_rank": float(1.0 / rank),
        "ranking": rankings[q],
    }

hit_at_1 = float(np.mean([per_query[q]["hit@1"] for q in query_ids]))
hit_at_3 = float(np.mean([per_query[q]["hit@3"] for q in query_ids]))
mrr = float(np.mean([per_query[q]["reciprocal_rank"] for q in query_ids]))

# ----------------------------------------------------------------------------
# 5. Guardar TODAS las cifras en resultados.json
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T1 Parte 1 - linea base lexica TF-IDF + similitud coseno",
    "vectorizador": "TfidfVectorizer (parametros por defecto)",
    "metrica_similitud": "coseno",
    "num_documentos": int(n_docs),
    "num_consultas": int(len(query_ids)),
    "tam_vocabulario": int(len(vectorizer.vocabulary_)),
    "juicios_relevancia": RELEVANT,
    "rankings": rankings,
    "similitudes_coseno": similitudes,
    "metricas_por_consulta": per_query,
    "hit_at_1": hit_at_1,
    "hit_at_3": hit_at_3,
    "mrr": mrr,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 6. Impresión de cifras principales (4 decimales)
# ----------------------------------------------------------------------------
print("=== T1 Parte 1: linea base TF-IDF + similitud coseno ===")
print(f"Documentos: {n_docs} | Consultas: {len(query_ids)} | "
      f"Terminos de vocabulario: {len(vectorizer.vocabulary_)}")
print("-" * 72)
for q in query_ids:
    m = per_query[q]
    print(f"{q} (relevante: {m['documento_relevante']}): "
          f"rank = {m['rank_relevante']} | "
          f"Hit@1 = {m['hit@1']:.4f} | Hit@3 = {m['hit@3']:.4f} | "
          f"RR = {m['reciprocal_rank']:.4f}")
    print(f"    ranking: {' > '.join(m['ranking'])}")
print("-" * 72)
print(f"Hit@1 = {hit_at_1:.4f}")
print(f"Hit@3 = {hit_at_3:.4f}")
print(f"MRR   = {mrr:.4f}")
print("Resultados guardados en resultados.json")
