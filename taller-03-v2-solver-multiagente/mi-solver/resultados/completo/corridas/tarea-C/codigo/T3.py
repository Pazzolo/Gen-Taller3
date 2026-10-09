# -*- coding: utf-8 -*-
"""
T3 — Análisis por consulta (Parte 3 de la Tarea C).

Por consulta y por método (TF-IDF + coseno y LSA-4 + coseno):
  * identifica los fallos: el documento relevante queda fuera del top 3;
  * calcula los términos que comparten la consulta y su documento relevante
    (intersección de tokens según la vectorización TF-IDF ajustada SOLO con
    el corpus), para explicar cada fallo;
  * genera la tabla del análisis por consulta.

El script es autocontenido: recrea la línea base léxica (T1) y el LSA-4 (T2)
con la misma configuración (TfidfVectorizer por defecto; TruncatedSVD con
n_components=4 y random_state=0, ajustado solo con los documentos; las
consultas solo se transforman, nunca se usan para ajustar nada).
"""

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------
# 1. Datos del enunciado (corpus mínimo: 10 documentos, 6 consultas)
# ----------------------------------------------------------------------
DOCUMENTOS = {
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

CONSULTAS = {
    "q1": ("¿Qué función de ranking léxica pondera la frecuencia de términos?", "d04"),
    "q2": ("¿Cómo se añaden fragmentos recuperados al prompt?", "d03"),
    "q3": ("¿Qué técnica entrena matrices de bajo rango?", "d08"),
    "q4": ("¿Cómo se compara la orientación de dos vectores de texto?", "d05"),
    "q5": ("¿Qué hace que el modelo sea más determinista al elegir la siguiente palabra?", "d09"),
    "q6": ("¿Qué arquitectura combina autoatención con capas feed-forward?", "d02"),
}

K_TOP = 3            # umbral de éxito: relevante dentro del top 3
N_COMPONENTES_LSA = 4
RANDOM_STATE_SVD = 0

DOC_IDS = sorted(DOCUMENTOS.keys())
Q_IDS = sorted(CONSULTAS.keys())
doc_texts = [DOCUMENTOS[d] for d in DOC_IDS]
query_texts = [CONSULTAS[q][0] for q in Q_IDS]
relevante_de = {q: CONSULTAS[q][1] for q in Q_IDS}

# ----------------------------------------------------------------------
# 2. Vectorización y métodos (idénticos a T1 y T2)
# ----------------------------------------------------------------------
vectorizer = TfidfVectorizer()
X_docs = vectorizer.fit_transform(doc_texts)      # ajuste SOLO con el corpus
X_queries = vectorizer.transform(query_texts)     # las consultas solo se transforman
vocab = vectorizer.vocabulary_
inv_vocab = {idx: term for term, idx in vocab.items()}
analyzer = vectorizer.build_analyzer()
vocab_set = set(vocab.keys())

# Método 1: TF-IDF + coseno
S_tfidf = cosine_similarity(X_queries, X_docs)    # forma (6, 10)

# Método 2: LSA-4 (TruncatedSVD sobre el TF-IDF de los documentos) + coseno
svd = TruncatedSVD(n_components=N_COMPONENTES_LSA, random_state=RANDOM_STATE_SVD)
D_lsa = svd.fit_transform(X_docs)                 # ajuste SOLO con los documentos
Q_lsa = svd.transform(X_queries)
S_lsa = cosine_similarity(Q_lsa, D_lsa)

var_por_componente = [float(v) for v in svd.explained_variance_ratio_]
var_acum = float(np.sum(var_por_componente))

METODOS = [
    ("TF-IDF (coseno)", S_tfidf, "tfidf"),
    ("LSA-4 (TruncatedSVD, coseno)", S_lsa, "lsa"),
]

# ----------------------------------------------------------------------
# 3. Funciones auxiliares
# ----------------------------------------------------------------------
def ranking_de_fila(sims_row):
    """Ranking de documentos (empates resueltos por orden d01..d10) y similitudes."""
    orden = np.argsort(-np.asarray(sims_row), kind="stable")
    ranking = [DOC_IDS[j] for j in orden]
    sims = {DOC_IDS[j]: float(sims_row[j]) for j in range(len(DOC_IDS))}
    return ranking, sims


def metricas_de_ranks(ranks):
    r = np.asarray(ranks, dtype=float)
    return {
        "hit_at_1": float(np.mean(r <= 1.0)),
        "hit_at_3": float(np.mean(r <= 3.0)),
        "mrr": float(np.mean(1.0 / r)),
    }


def construir_explicacion(metodo, qid, rel, rank_rel, topk, comunes, sim_rel):
    cabecera = (
        f"FALLO de {metodo} en {qid}: el documento relevante {rel} queda en el puesto "
        f"{float(rank_rel):.0f}, fuera del top {K_TOP} (similitud coseno con la consulta = "
        f"{sim_rel:.4f}; top {K_TOP} devuelto: {', '.join(topk)})."
    )
    if len(comunes) == 0:
        cuerpo = (
            "Consulta y documento relevante no comparten NINGÚN término del vocabulario de la "
            "vectorización: sin solapamiento léxico, la similitud coseno no puede favorecer al "
            "relevante frente a documentos que sí comparten términos con la consulta."
        )
    else:
        cuerpo = (
            f"Consulta y relevante comparten {len(comunes)} término(s) del vocabulario "
            f"[{', '.join(comunes)}], pero ese solapamiento no basta para situar al relevante "
            f"en el top {K_TOP} con este método."
        )
    if metodo.startswith("LSA"):
        cola = (
            f"En el espacio latente de {N_COMPONENTES_LSA} componentes (varianza explicada "
            f"acumulada = {var_acum:.4f}), la consulta proyecta más cerca de otros documentos, "
            f"lo que desplaza al relevante fuera del top {K_TOP}."
        )
        return f"{cabecera} {cuerpo} {cola}"
    return f"{cabecera} {cuerpo}"


# ----------------------------------------------------------------------
# 4. Análisis por consulta: fallos por método y términos comunes
# ----------------------------------------------------------------------
tabla = []
fallos_por_metodo = {nombre: [] for nombre, _, _ in METODOS}
ranks_por_metodo = {nombre: [] for nombre, _, _ in METODOS}

for qi, qid in enumerate(Q_IDS):
    texto_q = query_texts[qi]
    rel = relevante_de[qid]

    # Tokens (según la vectorización) de la consulta y del documento relevante
    tokens_q = sorted(set(analyzer(texto_q)) & vocab_set)
    tokens_rel = sorted(set(analyzer(DOCUMENTOS[rel])) & vocab_set)
    comunes = sorted(set(tokens_q) & set(tokens_rel))

    # Peso TF-IDF de cada término común dentro del vector de la consulta
    fila = X_queries[qi].toarray().ravel()
    pesos_q = {inv_vocab[j]: float(fila[j]) for j in np.nonzero(fila)[0]}
    pesos_comunes = {t: pesos_q[t] for t in comunes}

    entrada = {
        "consulta_id": qid,
        "consulta": texto_q,
        "relevante": rel,
        "tokens_consulta_en_vocabulario": tokens_q,
        "num_tokens_consulta_en_vocabulario": float(len(tokens_q)),
        "tokens_relevante_en_vocabulario": tokens_rel,
        "num_tokens_relevante_en_vocabulario": float(len(tokens_rel)),
        "terminos_comunes_consulta_relevante": comunes,
        "num_terminos_comunes": float(len(comunes)),
        "peso_tfidf_en_consulta_de_terminos_comunes": pesos_comunes,
    }

    for nombre, S, clave in METODOS:
        ranking, sims = ranking_de_fila(S[qi])
        rank_rel = ranking.index(rel) + 1
        topk = ranking[:K_TOP]
        fallo = bool(rank_rel > K_TOP)
        ranks_por_metodo[nombre].append(rank_rel)
        entrada[clave] = {
            "metodo": nombre,
            "ranking_documentos": ranking,
            "top3": topk,
            "rank_relevante": float(rank_rel),
            "fallo_top3": fallo,
            "similitud_relevante": sims[rel],
            "similitudes_coseno": sims,
        }
        if fallo:
            fallos_por_metodo[nombre].append({
                "consulta_id": qid,
                "consulta": texto_q,
                "relevante": rel,
                "rank_relevante": float(rank_rel),
                "top3_devuelto": topk,
                "similitud_relevante": sims[rel],
                "terminos_comunes_consulta_relevante": comunes,
                "num_terminos_comunes": float(len(comunes)),
                "tokens_consulta_en_vocabulario": tokens_q,
                "explicacion": construir_explicacion(
                    nombre, qid, rel, rank_rel, topk, comunes, sims[rel]
                ),
            })
    tabla.append(entrada)

# ----------------------------------------------------------------------
# 5. Resumen de fallos y métricas de comprobación
# ----------------------------------------------------------------------
met_tfidf = metricas_de_ranks(ranks_por_metodo["TF-IDF (coseno)"])
met_lsa = metricas_de_ranks(ranks_por_metodo["LSA-4 (TruncatedSVD, coseno)"])
fallos_tfidf_ids = [e["consulta_id"] for e in fallos_por_metodo["TF-IDF (coseno)"]]
fallos_lsa_ids = [e["consulta_id"] for e in fallos_por_metodo["LSA-4 (TruncatedSVD, coseno)"]]

resumen_fallos = {
    "k_umbral_top3": float(K_TOP),
    "num_consultas": float(len(Q_IDS)),
    "num_fallos_tfidf": float(len(fallos_tfidf_ids)),
    "num_fallos_lsa": float(len(fallos_lsa_ids)),
    "consultas_con_fallo_tfidf": fallos_tfidf_ids,
    "consultas_con_fallo_lsa": fallos_lsa_ids,
    "consultas_con_fallo_en_ambos_metodos": sorted(set(fallos_tfidf_ids) & set(fallos_lsa_ids)),
    "consultas_sin_fallo_en_ningun_metodo": sorted(
        set(Q_IDS) - set(fallos_tfidf_ids) - set(fallos_lsa_ids)
    ),
    "tasa_exito_top3_tfidf": met_tfidf["hit_at_3"],
    "tasa_exito_top3_lsa": met_lsa["hit_at_3"],
    "metricas_tfidf": met_tfidf,
    "metricas_lsa": met_lsa,
    "num_terminos_comunes_por_consulta": {
        e["consulta_id"]: e["num_terminos_comunes"] for e in tabla
    },
}

# ----------------------------------------------------------------------
# 6. Guardar resultados.json
# ----------------------------------------------------------------------
resultados = {
    "tarea": "T3_analisis_por_consulta_fallos_y_terminos_comunes",
    "configuracion": {
        "k_umbral_top3": float(K_TOP),
        "n_componentes_lsa": float(N_COMPONENTES_LSA),
        "random_state_svd": float(RANDOM_STATE_SVD),
        "num_documentos": float(len(DOC_IDS)),
        "num_consultas": float(len(Q_IDS)),
        "tamano_vocabulario_tfidf": float(len(vocab)),
        "varianza_explicada_acumulada_lsa": var_acum,
    },
    "varianza_explicada_por_componente_lsa": var_por_componente,
    "tabla_analisis_por_consulta": tabla,
    "fallos_por_metodo": fallos_por_metodo,
    "resumen_fallos": resumen_fallos,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# 7. Impresión de las cifras principales (4 decimales)
# ----------------------------------------------------------------------
print("=" * 78)
print("T3 · Análisis por consulta: fallos (relevante fuera del top 3) y términos")
print("compartidos entre cada consulta y su documento relevante")
print("=" * 78)
print(
    f"Vocabulario TF-IDF: {len(vocab)} términos | LSA-{N_COMPONENTES_LSA}: "
    f"varianza explicada acumulada = {var_acum:.4f}"
)
print(f"Umbral de éxito: relevante dentro del top {K_TOP}")
print()
print("--- Tabla de análisis por consulta ---")
for e in tabla:
    comunes_txt = (
        ", ".join(e["terminos_comunes_consulta_relevante"])
        if e["terminos_comunes_consulta_relevante"]
        else "(ninguno)"
    )
    print(
        f"{e['consulta_id']} | relevante: {e['relevante']} | "
        f"términos comunes ({int(e['num_terminos_comunes'])}): {comunes_txt}"
    )
    for clave in ("tfidf", "lsa"):
        m = e[clave]
        print(
            f"    {m['metodo']:<30s} rank relevante = {m['rank_relevante']:.4f} | "
            f"fallo top3 = {'SÍ' if m['fallo_top3'] else 'no'} | "
            f"sim = {m['similitud_relevante']:.4f} | top3 = {', '.join(m['top3'])}"
        )
print()
for nombre in fallos_por_metodo:
    fallos = fallos_por_metodo[nombre]
    ids_txt = ", ".join(e["consulta_id"] for e in fallos) if fallos else "ninguno"
    print(f"Fallos con {nombre}: {len(fallos)}/{len(Q_IDS)} consultas -> {ids_txt}")
    for e in fallos:
        print(f"  * {e['explicacion']}")
print()
print(
    f"Resumen: tasa de éxito top-3 TF-IDF = {met_tfidf['hit_at_3']:.4f} | "
    f"tasa de éxito top-3 LSA-4 = {met_lsa['hit_at_3']:.4f}"
)
print(
    f"         MRR TF-IDF = {met_tfidf['mrr']:.4f} | MRR LSA-4 = {met_lsa['mrr']:.4f}"
)
print("Resultados guardados en resultados.json")
