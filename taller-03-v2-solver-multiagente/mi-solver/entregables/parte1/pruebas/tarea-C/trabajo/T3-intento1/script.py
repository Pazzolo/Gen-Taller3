# -*- coding: utf-8 -*-
"""
T3 — Análisis por consulta (Tarea C, Parte 3).

Script autocontenido que:
  1) Reconstruye de forma independiente los rankings de T1 y T2:
       - T1: TF-IDF (TfidfVectorizer por defecto) + similitud coseno.
       - T2: LSA con TruncatedSVD(n_components=4, random_state=0) sobre la
         matriz TF-IDF; consultas proyectadas al espacio latente + coseno.
  2) Identifica en qué consultas falla cada método (relevante fuera del top 3).
  3) Extrae, con el vocabulario TF-IDF, los términos que comparte cada consulta
     con su documento relevante (pesos TF-IDF de consulta y documento, e IDF)
     para explicar los fallos.
  4) Genera la tabla de la Parte 3 y guarda TODAS las cifras en resultados.json.
No se usa el conjunto de consulta para ajustar nada: k=4 y random_state=0 son
los fijados por T2; el vectorizador es el de T1 sin modificación.
"""

import json
import math

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# 1) Datos del enunciado (corpus, consultas y juicios de relevancia)
# ----------------------------------------------------------------------
DOCS = {
    "d01": "El mecanismo de atención pondera cada token según su similitud con la consulta; la atención escalada divide por la raíz de la dimensión.",
    "d02": "Los transformadores apilan capas de autoatención y redes feed-forward, con conexiones residuales y normalización.",
    "d03": "La recuperación aumentada con generación busca fragmentos relevantes y los añade al prompt del modelo.",
    "d04": "BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento.",
    "d05": "Los embeddings densos representan textos como vectores; la similitud coseno compara su orientación.",
    "d06": "Un agente con herramientas decide en cada paso qué función llamar y observa el resultado.",
    "d07": "ReAct intercala razonamiento y acciones; Reflexion añade una autocrítica verbal entre intentos.",
    "d08": "El ajuste fino con LoRA entrena matrices de bajo rango y congela los pesos originales.",
    "d09": "La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad.",
    "d10": "La cuantización reduce la precisión de los pesos a 8 o 4 bits para ahorrar memoria.",
}

QUERIES = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

doc_ids = sorted(DOCS)
query_ids = sorted(QUERIES)
doc_texts = [DOCS[d] for d in doc_ids]
TOP_K = 3  # definición de fallo: relevante en puesto > 3

assert len(doc_ids) == 10 and len(query_ids) == 6

# ----------------------------------------------------------------------
# 2) T1: línea base léxica TF-IDF + coseno (recalculada íntegra)
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()  # configuración por defecto, idéntica a T1
X_docs = vectorizer.fit_transform(doc_texts)
X_q = vectorizer.transform([QUERIES[q][0] for q in query_ids])
S_tfidf = cosine_similarity(X_q, X_docs)

# ----------------------------------------------------------------------
# 3) T2: LSA (TruncatedSVD k=4, random_state=0) sobre la matriz TF-IDF
# ----------------------------------------------------------------------
N_COMPONENTES = 4
RANDOM_STATE_SVD = 0
svd = TruncatedSVD(n_components=N_COMPONENTES, random_state=RANDOM_STATE_SVD)
X_docs_lsa = svd.fit_transform(X_docs)
X_q_lsa = svd.transform(X_q)
S_lsa = cosine_similarity(X_q_lsa, X_docs_lsa)

# ----------------------------------------------------------------------
# 4) Rankings y métricas por método (Hit@1, Hit@3, MRR)
# ----------------------------------------------------------------------
def evaluar_metodo(S, nombre_metodo):
    por_consulta = {}
    for i, qid in enumerate(query_ids):
        orden = np.argsort(-S[i], kind="stable")
        ranking = [doc_ids[j] for j in orden]
        rel = QUERIES[qid][1]
        rank = int(ranking.index(rel) + 1)
        por_consulta[qid] = {
            "ranking_documentos": ranking,
            "similitudes_ordenadas": [float(x) for x in S[i, orden]],
            "rank_del_relevante": rank,
            "similitud_con_relevante": float(S[i, doc_ids.index(rel)]),
            "hit_at_1": 1.0 if rank == 1 else 0.0,
            "hit_at_3": 1.0 if rank <= TOP_K else 0.0,
            "reciprocal_rank": 1.0 / rank,
            "top_3": ranking[:TOP_K],
        }
    n = len(query_ids)
    globales = {
        "hit_at_1": float(sum(p["hit_at_1"] for p in por_consulta.values()) / n),
        "hit_at_3": float(sum(p["hit_at_3"] for p in por_consulta.values()) / n),
        "mrr": float(sum(p["reciprocal_rank"] for p in por_consulta.values()) / n),
    }
    return {"metodo": nombre_metodo, "por_consulta": por_consulta, "globales": globales}


ev_tfidf = evaluar_metodo(S_tfidf, "TF-IDF")
ev_lsa = evaluar_metodo(S_lsa, "LSA")

# ----------------------------------------------------------------------
# 5) Análisis por consulta: fallos y términos compartidos (vocabulario TF-IDF)
# ----------------------------------------------------------------------
analyzer = vectorizer.build_analyzer()
vocab = vectorizer.vocabulary_
idf_vec = vectorizer.idf_
umbral_idf = float(np.median(idf_vec))  # IDF por debajo de la mediana -> término genérico

Xq = X_q.toarray()
Xd = X_docs.toarray()


def terminos_en_vocabulario(texto):
    """Tokens del texto que pertenecen al vocabulario TF-IDF (orden alfabético)."""
    return sorted({t for t in analyzer(texto) if t in vocab})


def veredicto(nombre, rank):
    if rank > TOP_K:
        return f"{nombre}: FALLO (relevante en el puesto {rank}, fuera del top 3)"
    return f"{nombre}: acierto (relevante en el puesto {rank})"


analisis = {}
tabla_parte3 = []

for i, qid in enumerate(query_ids):
    texto_q, rel = QUERIES[qid]
    r = doc_ids.index(rel)
    rank_t = ev_tfidf["por_consulta"][qid]["rank_del_relevante"]
    rank_l = ev_lsa["por_consulta"][qid]["rank_del_relevante"]

    q_terminos = terminos_en_vocabulario(texto_q)
    d_terminos = set(terminos_en_vocabulario(DOCS[rel]))

    # Términos compartidos consulta <-> documento relevante, con pesos TF-IDF e IDF
    compartidos = []
    for t in q_terminos:
        if t in d_terminos:
            j = vocab[t]
            compartidos.append({
                "termino": t,
                "peso_consulta": float(Xq[i, j]),
                "peso_documento": float(Xd[r, j]),
                "idf": float(idf_vec[j]),
            })
    compartidos.sort(key=lambda d: (-(d["peso_consulta"] + d["peso_documento"]), d["termino"]))

    genericos = [c["termino"] for c in compartidos if c["idf"] < umbral_idf]
    contenido = [c["termino"] for c in compartidos if c["idf"] >= umbral_idf]

    # Términos de la consulta (en vocabulario) que NO están en el relevante
    ausentes = []
    for t in q_terminos:
        if t not in d_terminos:
            j = vocab[t]
            ausentes.append({"termino": t, "peso_consulta": float(Xq[i, j]), "idf": float(idf_vec[j])})
    ausentes.sort(key=lambda d: -d["peso_consulta"])
    ausentes_contenido = [a for a in ausentes if a["idf"] >= umbral_idf] or ausentes

    # --- explicación automática del resultado de la consulta ---
    if not compartidos:
        solape = (f"La consulta no comparte ningún término del vocabulario TF-IDF con {rel} "
                  f"(solapamiento léxico nulo).")
    else:
        partes = []
        if contenido:
            partes.append("solapamiento de contenido: " + ", ".join(f"'{t}'" for t in contenido))
        if genericos:
            partes.append("términos genéricos (idf < {:.4f}): ".format(umbral_idf)
                          + ", ".join(f"'{t}'" for t in genericos))
        solape = (f"La consulta comparte {len(compartidos)} término(s) con {rel}: "
                  + "; ".join(partes) + ".")
    extra = ""
    if rank_t > TOP_K or rank_l > TOP_K:
        top_aus = ", ".join(f"'{a['termino']}'" for a in ausentes_contenido[:5])
        extra = (f" Términos distintivos de la consulta ausentes en {rel}: {top_aus}. "
                 f"Puesto del relevante: {rank_t} (TF-IDF) y {rank_l} (LSA).")
    explicacion = solape + extra + f" {veredicto('TF-IDF', rank_t)}; {veredicto('LSA', rank_l)}."

    analisis[qid] = {
        "consulta": texto_q,
        "documento_relevante": rel,
        "rank_tfidf": rank_t,
        "rank_lsa": rank_l,
        "fallo_tfidf": bool(rank_t > TOP_K),
        "fallo_lsa": bool(rank_l > TOP_K),
        "similitud_tfidf_con_relevante": ev_tfidf["por_consulta"][qid]["similitud_con_relevante"],
        "similitud_lsa_con_relevante": ev_lsa["por_consulta"][qid]["similitud_con_relevante"],
        "top3_tfidf": ev_tfidf["por_consulta"][qid]["top_3"],
        "top3_lsa": ev_lsa["por_consulta"][qid]["top_3"],
        "terminos_consulta_en_vocabulario": q_terminos,
        "num_terminos_compartidos": len(compartidos),
        "terminos_compartidos": compartidos,
        "terminos_compartidos_de_contenido": contenido,
        "terminos_compartidos_genericos": genericos,
        "terminos_consulta_ausentes_en_relevante": ausentes,
        "explicacion": explicacion,
    }

    fmt_terms = [f"{c['termino']} (q={c['peso_consulta']:.4f}, d={c['peso_documento']:.4f})"
                 for c in compartidos]
    tabla_parte3.append({
        "consulta_id": qid,
        "consulta": texto_q,
        "relevante": rel,
        "rank_tfidf": rank_t,
        "rank_lsa": rank_l,
        "fallo_tfidf": bool(rank_t > TOP_K),
        "fallo_lsa": bool(rank_l > TOP_K),
        "num_terminos_compartidos": len(compartidos),
        "terminos_compartidos_fmt": fmt_terms,
        "terminos_compartidos": compartidos,
        "explicacion": explicacion,
    })

# ----------------------------------------------------------------------
# 6) Agregados de fallos y verificación de coherencia con T1
# ----------------------------------------------------------------------
fallos_tfidf = [qid for qid in query_ids if analisis[qid]["fallo_tfidf"]]
fallos_lsa = [qid for qid in query_ids if analisis[qid]["fallo_lsa"]]
fallos_ambos = [qid for qid in fallos_tfidf if qid in fallos_lsa]

t1_ref = {"hit_at_1": 0.8333333333333334, "hit_at_3": 0.8333333333333334, "mrr": 0.8611111111111112}
coincide_t1 = {k: bool(math.isclose(ev_tfidf["globales"][k], v, rel_tol=1e-9, abs_tol=1e-12))
               for k, v in t1_ref.items()}

# ----------------------------------------------------------------------
# 7) Guardar TODAS las cifras en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "subtarea": "T3_analisis_por_consulta",
    "descripcion": ("Análisis por consulta a partir de los rankings de T1 (TF-IDF + coseno) y T2 "
                    "(LSA, TruncatedSVD k=4, random_state=0), recalculados de forma independiente: "
                    "consultas donde cada método falla (relevante fuera del top 3) y términos que "
                    "comparte cada consulta con su documento relevante según el vocabulario TF-IDF "
                    "(pesos TF-IDF e IDF) para explicar los fallos. Incluye la tabla de la Parte 3."),
    "parametros": {
        "num_documentos": len(doc_ids),
        "num_consultas": len(query_ids),
        "tamano_vocabulario": int(len(vocab)),
        "n_componentes_lsa": N_COMPONENTES,
        "random_state_svd": RANDOM_STATE_SVD,
        "definicion_fallo": "documento relevante en puesto > 3 (fuera del top 3)",
        "umbral_idf_generico": umbral_idf,
        "criterio_umbral_generico": "mediana del IDF del vocabulario TF-IDF",
    },
    "lsa_varianza_explicada_por_componente": [float(x) for x in svd.explained_variance_ratio_],
    "lsa_varianza_explicada_total": float(sum(svd.explained_variance_ratio_)),
    "metricas_globales": {"tfidf": ev_tfidf["globales"], "lsa": ev_lsa["globales"]},
    "fallos_por_metodo": {
        "tfidf": {"consultas": fallos_tfidf, "num_fallos": len(fallos_tfidf)},
        "lsa": {"consultas": fallos_lsa, "num_fallos": len(fallos_lsa)},
        "ambos_metodos": {"consultas": fallos_ambos, "num_fallos": len(fallos_ambos)},
    },
    "tabla_parte3": tabla_parte3,
    "por_consulta": analisis,
    "rankings_recalculados": {
        "tfidf": {qid: ev_tfidf["por_consulta"][qid] for qid in query_ids},
        "lsa": {qid: ev_lsa["por_consulta"][qid] for qid in query_ids},
    },
    "coherencia_con_T1": {
        "metricas_tfidf_referencia_T1": t1_ref,
        "coinciden": coincide_t1,
    },
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 8) Salida por consola (cifras principales con 4 decimales)
# ----------------------------------------------------------------------
print("=" * 110)
print("T3 — Análisis por consulta (Parte 3): fallos por método y términos compartidos consulta-relevante")
print("=" * 110)
print(f"Vocabulario TF-IDF: {len(vocab)} términos | umbral IDF genérico (mediana): {umbral_idf:.4f}")
g_t, g_l = ev_tfidf["globales"], ev_lsa["globales"]
print(f"TF-IDF    : Hit@1={g_t['hit_at_1']:.4f}  Hit@3={g_t['hit_at_3']:.4f}  MRR={g_t['mrr']:.4f}  "
      f"fallos (fuera de top 3): {fallos_tfidf}")
print(f"LSA (k=4) : Hit@1={g_l['hit_at_1']:.4f}  Hit@3={g_l['hit_at_3']:.4f}  MRR={g_l['mrr']:.4f}  "
      f"fallos (fuera de top 3): {fallos_lsa}")
print(f"Varianza explicada por LSA (k=4): "
      f"{', '.join(f'{v:.4f}' for v in svd.explained_variance_ratio_)} "
      f"(total {float(sum(svd.explained_variance_ratio_)):.4f})")
print(f"Fallan en ambos métodos: {fallos_ambos}")
print("-" * 110)
print("Tabla de la Parte 3 (rank del documento relevante; fallo = rank > 3):")
print(f"{'q':<4} {'rel':<5} {'rkTF':>4} {'rkLSA':>5} {'falloTF':>8} {'falloLSA':>9} "
      f"{'nCompart':>9}  términos compartidos (peso consulta / peso documento)")
print("-" * 110)
for fila in tabla_parte3:
    terms = fila["terminos_compartidos_fmt"]
    mostrados = "; ".join(terms[:4]) + ("; ..." if len(terms) > 4 else "")
    print(f"{fila['consulta_id']:<4} {fila['relevante']:<5} {fila['rank_tfidf']:>4} {fila['rank_lsa']:>5} "
          f"{('SÍ' if fila['fallo_tfidf'] else 'no'):>8} {('SÍ' if fila['fallo_lsa'] else 'no'):>9} "
          f"{fila['num_terminos_compartidos']:>9}  {mostrados}")
print("-" * 110)
print("Explicaciones por consulta:")
for qid in query_ids:
    print(f"[{qid}] {analisis[qid]['explicacion']}")
print("-" * 110)
print("Coherencia con T1 (TF-IDF recalculado vs. T1): "
      + ", ".join(f"{k}={'ok' if v else 'NO'}" for k, v in coincide_t1.items()))
print("Resultados guardados en resultados.json")
