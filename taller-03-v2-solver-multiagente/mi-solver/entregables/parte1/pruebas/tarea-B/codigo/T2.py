# -*- coding: utf-8 -*-
"""
SUBTAREA T2 — Muestreo de núcleo (top-p) con p = 0.9 sobre la distribución de T = 1.

Pasos:
1) Recalcula de forma independiente la distribución softmax con temperatura T = 1
   (Parte 1) a partir de los logits del enunciado.
2) Ordena los tokens por probabilidad descendente.
3) Conserva el conjunto mínimo de tokens cuya probabilidad acumulada alcanza p = 0.9
   (el token que hace cruzar el umbral se incluye).
4) Renormaliza las probabilidades de los tokens supervivientes.

Guarda todas las cifras en 'resultados.json' e imprime las principales con 4 decimales.
No se requiere figura.
"""

import json
import numpy as np

# ---------------------------------------------------------------------
# Datos del enunciado (Parte 1)
# ---------------------------------------------------------------------
TOKENS = ["t0", "t1", "t2", "t3", "t4", "t5"]
LOGITS = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
T = 1.0   # temperatura de la distribución de la Parte 1 sobre la que se aplica top-p
P = 0.9   # umbral del núcleo (nucleus sampling)

# ---------------------------------------------------------------------
# Softmax con temperatura, numéricamente estable (recálculo de la Parte 1)
# ---------------------------------------------------------------------
def softmax_con_temperatura(logits, temperatura):
    z = np.asarray(logits, dtype=float) / float(temperatura)
    z = z - np.max(z)                     # estabilidad numérica: restar el máximo
    e = np.exp(z)
    return e / np.sum(e)

p_T1 = softmax_con_temperatura(LOGITS, T)

# ---------------------------------------------------------------------
# Top-p (nucleus sampling) con p = 0.9
# ---------------------------------------------------------------------
# 1) Ordenar los tokens por probabilidad descendente
orden = np.argsort(-p_T1, kind="stable")
probs_ord = p_T1[orden]
tokens_ord = [TOKENS[i] for i in orden]

# 2) Probabilidad acumulada en ese orden
acum = np.cumsum(probs_ord)

# 3) Conjunto mínimo cuya probabilidad acumulada alcanza p
#    (se incluye el token con el que la acumulada cruza el umbral)
k = 0
acc = 0.0
for prob in probs_ord:
    k += 1
    acc += float(prob)
    if acc >= P:
        break

idx_nucleo = orden[:k]
tokens_nucleo = [TOKENS[i] for i in idx_nucleo]
tokens_fuera = [TOKENS[i] for i in orden[k:]]
suma_nucleo = acc  # probabilidad acumulada del núcleo antes de renormalizar

# 4) Renormalizar
p_renorm_nucleo = p_T1[idx_nucleo] / suma_nucleo

# Distribución renormalizada en el orden original t0..t5 (0.0 para descartados)
dist_renorm = {tok: 0.0 for tok in TOKENS}
for i, tok in zip(idx_nucleo, tokens_nucleo):
    dist_renorm[tok] = float(p_T1[i] / suma_nucleo)

# Cifras auxiliares
probs_T1 = {tok: float(v) for tok, v in zip(TOKENS, p_T1)}
acum_dict = {tok: float(a) for tok, a in zip(tokens_ord, acum)}
probs_ord_list = [float(v) for v in probs_ord]

def entropia_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

H_T1 = entropia_bits(p_T1)
H_renorm = entropia_bits(p_renorm_nucleo)
suma_renorm = float(sum(dist_renorm.values()))

# ---------------------------------------------------------------------
# Guardar TODAS las cifras en resultados.json (fusionando con lo previo)
# ---------------------------------------------------------------------
resultados_T2 = {
    "subtarea": "T2: top-p (nucleus) con p=0.9 sobre la distribucion de T=1",
    "tokens": list(TOKENS),
    "logits": [float(z) for z in LOGITS],
    "temperatura": float(T),
    "p": float(P),
    "distribucion_T1": probs_T1,
    "tokens_ordenados_desc": tokens_ord,
    "probs_ordenadas_desc": probs_ord_list,
    "prob_acumulada_orden_desc": acum_dict,
    "num_tokens_nucleo": int(k),
    "tokens_sobrevivientes": tokens_nucleo,
    "tokens_descartados": tokens_fuera,
    "suma_prob_nucleo_antes_renorm": float(suma_nucleo),
    "distribucion_renormalizada": dist_renorm,
    "suma_distribucion_renormalizada": suma_renorm,
    "entropia_bits_T1_original": H_T1,
    "entropia_bits_tras_top_p": H_renorm,
}

try:
    with open("resultados.json", "r") as f:
        previo = json.load(f)
    if not isinstance(previo, dict):
        previo = {}
except (FileNotFoundError, json.JSONDecodeError):
    previo = {}

previo["T2_top_p"] = resultados_T2

with open("resultados.json", "w") as f:
    json.dump(previo, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------
print("Top-p (nucleus sampling) con p = 0.9 sobre la distribucion de T = 1")
print("------------------------------------------------------------------")
print(f"{'token':>6} {'p(T=1)':>10} {'acumulada':>10}   nucleo")
print("------------------------------------------------------------------")
en_nucleo = set(tokens_nucleo)
for tok in tokens_ord:
    marca = "si" if tok in en_nucleo else "no"
    print(f"{tok:>6} {probs_T1[tok]:>10.4f} {acum_dict[tok]:>10.4f}   {marca}")
print("------------------------------------------------------------------")
print(f"Tokens sobrevivientes (conjunto minimo con acumulada >= {P:.1f}): {tokens_nucleo}")
print(f"Probabilidad acumulada del nucleo antes de renormalizar: {suma_nucleo:.4f}")
print(f"Tokens descartados: {tokens_fuera}")
print()
print("Distribucion renormalizada:")
for tok in TOKENS:
    print(f"  {tok}: {dist_renorm[tok]:.4f}")
print(f"Suma: {suma_renorm:.4f}")
print()
print("Cifras principales:")
print(f"  Nucleo (p={P:.1f}): {tokens_nucleo} "
      f"(k={k} tokens, acumulada original = {suma_nucleo:.4f})")
print("  Distribucion renormalizada: "
      + ", ".join(f"{tok}={dist_renorm[tok]:.4f}" for tok in tokens_nucleo))
print(f"  Entropia T=1 original: {H_T1:.4f} bits | Entropia tras top-p: {H_renorm:.4f} bits")
print()
print("Resultados guardados en 'resultados.json'.")
