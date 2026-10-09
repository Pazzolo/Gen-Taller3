# -*- coding: utf-8 -*-
"""
SUBTAREA T3 — Comprobación empírica del muestreo de núcleo (top-p)
===================================================================
1) Recalcula (de forma autónoma) la distribución softmax con T=1 (Parte 1)
   y el núcleo top-p con p=0.9 renormalizado (Parte 2).
2) Toma 10 000 muestras de esa distribución renormalizada con
   numpy.random.default_rng(0).
3) Compara frecuencias observadas vs probabilidades teóricas en un
   gráfico de barras (PNG) y calcula la divergencia KL (en bits) de la
   distribución empírica respecto de la teórica.
Guarda todas las cifras en resultados.json.
"""

import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. Datos del enunciado (Partes 1 y 2), recomputados para autonomía
# ----------------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
temperatura = 1.0
p_nucleo = 0.9
num_muestras = 10000
semilla = 0

# Softmax con temperatura
z = logits / temperatura
z = z - np.max(z)
exp_z = np.exp(z)
probs_T = exp_z / np.sum(exp_z)

# Top-p: conjunto mínimo de tokens (orden descendente) con acumulada >= p
orden = np.argsort(-probs_T, kind="stable")
probs_ord = probs_T[orden]
acum = np.cumsum(probs_ord)
k = int(np.searchsorted(acum, p_nucleo - 1e-12, side="left")) + 1
idx_nucleo = np.sort(orden[:k])
tokens_nucleo = [tokens[i] for i in idx_nucleo]
tokens_fuera = [tokens[i] for i in orden[k:]]
masa_nucleo = float(np.sum(probs_T[idx_nucleo]))

# Distribución renormalizada sobre los 6 tokens (los eliminados quedan con 0)
probs_teo = np.zeros(len(tokens), dtype=float)
probs_teo[idx_nucleo] = probs_T[idx_nucleo] / masa_nucleo

# Entropía teórica de la distribución renormalizada (bits)
entropia_teo = 0.0
for pt in probs_teo:
    if pt > 0.0:
        entropia_teo -= pt * math.log2(pt)

# ----------------------------------------------------------------------------
# 2. Muestreo: 10 000 muestras con numpy.random.default_rng(0)
# ----------------------------------------------------------------------------
rng = np.random.default_rng(semilla)
muestras = rng.choice(len(tokens), size=num_muestras, p=probs_teo)
conteos = np.bincount(muestras, minlength=len(tokens)).astype(int)
frec_obs = conteos / float(num_muestras)

# ----------------------------------------------------------------------------
# 3. Divergencia KL (bits) de la empírica respecto de la teórica:
#    D_KL(P_emp || P_teo) = sum_x P_emp(x) * log2(P_emp(x) / P_teo(x))
#    Convención: los términos con P_emp(x) = 0 aportan 0.
#    Si P_teo(x) = 0 con P_emp(x) > 0, la KL sería infinita (no ocurre aquí,
#    porque solo se muestrea del núcleo).
# ----------------------------------------------------------------------------
kl_bits = 0.0
kl_infinita = False
for pe, pt in zip(frec_obs, probs_teo):
    if pe > 0.0:
        if pt <= 0.0:
            kl_infinita = True
            break
        kl_bits += pe * math.log2(pe / pt)
if kl_infinita:
    kl_bits = float("inf")

# Entropía empírica (bits), para comparar con la teórica
entropia_emp = 0.0
for pe in frec_obs:
    if pe > 0.0:
        entropia_emp -= pe * math.log2(pe)

# Errores por token
errores_abs = np.abs(frec_obs - probs_teo)
error_max = float(np.max(errores_abs))
error_medio = float(np.mean(errores_abs))

# ----------------------------------------------------------------------------
# 4. Figura: barras agrupadas teórica vs observada
# ----------------------------------------------------------------------------
nombre_figura = "t3_frecuencias_vs_teoricas.png"
x = np.arange(len(tokens))
ancho = 0.38
fig, ax = plt.subplots(figsize=(9.5, 5.8))
b1 = ax.bar(x - ancho / 2.0, probs_teo, ancho,
            label="Teórica (top-p renormalizada)",
            color="#4C72B0", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + ancho / 2.0, frec_obs, ancho,
            label=f"Observada ({num_muestras} muestras, rng({semilla}))",
            color="#DD8452", edgecolor="black", linewidth=0.5)
for barras in (b1, b2):
    for b in barras:
        h = b.get_height()
        ax.annotate(f"{h:.4f}",
                    (b.get_x() + b.get_width() / 2.0, h),
                    ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / frecuencia")
ax.set_ylim(0.0, max(float(np.max(probs_teo)), float(np.max(frec_obs))) * 1.20)
ax.set_title(f"T3: Frecuencias observadas vs probabilidades teóricas\n"
             f"(softmax T={temperatura}, top-p p={p_nucleo}, "
             f"{num_muestras} muestras, semilla {semilla})")
ax.legend(loc="upper right")
ax.text(0.985, 0.90,
        f"D_KL(emp||teo) = {kl_bits:.4f} bits",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))
fig.tight_layout()
fig.savefig(nombre_figura, dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 5. Guardar todas las cifras en resultados.json
# ----------------------------------------------------------------------------
resultados = {
    "subtarea": "T3_comprobacion_empirica_top_p",
    "num_muestras": int(num_muestras),
    "semilla": int(semilla),
    "temperatura": float(temperatura),
    "p_nucleo": float(p_nucleo),
    "tokens": tokens,
    "tokens_supervivientes": tokens_nucleo,
    "tokens_eliminados": tokens_fuera,
    "masa_nucleo_antes_renormalizar": masa_nucleo,
    "distribucion_teorica_renormalizada": {t: float(p) for t, p in zip(tokens, probs_teo)},
    "suma_distribucion_teorica": float(np.sum(probs_teo)),
    "entropia_teorica_bits": float(entropia_teo),
    "conteos_observados": {t: int(c) for t, c in zip(tokens, conteos)},
    "frecuencias_observadas": {t: float(f) for t, f in zip(tokens, frec_obs)},
    "suma_frecuencias_observadas": float(np.sum(frec_obs)),
    "entropia_empirica_bits": float(entropia_emp),
    "divergencia_KL_bits": float(kl_bits),
    "divergencia_KL_infinita": bool(kl_infinita),
    "error_absoluto_por_token": {t: float(e) for t, e in zip(tokens, errores_abs)},
    "error_absoluto_maximo": error_max,
    "error_absoluto_medio": error_medio,
    "figura": nombre_figura,
    "figuras": [nombre_figura],
}
with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 6. Impresión de las cifras principales (4 decimales)
# ----------------------------------------------------------------------------
print("SUBTAREA T3 — Comprobación empírica del muestreo top-p")
print("=" * 66)
print(f"Muestras: {num_muestras} | semilla: {semilla} | T = {temperatura} | p = {p_nucleo}")
print(f"Tokens del núcleo: {tokens_nucleo} | eliminados: {tokens_fuera}")
print("-" * 66)
print("token   teórica   observada   conteo   |error|")
for t, pt, pe, c in zip(tokens, probs_teo, frec_obs, conteos):
    print(f"{t:>5}    {pt:.4f}    {pe:.4f}    {c:6d}   {abs(pe - pt):.4f}")
print("-" * 66)
print(f"Divergencia KL (empírica || teórica): {kl_bits:.4f} bits")
print(f"Entropía empírica:  {entropia_emp:.4f} bits")
print(f"Entropía teórica:   {entropia_teo:.4f} bits")
print(f"Error absoluto máximo: {error_max:.4f}")
print(f"Error absoluto medio:  {error_medio:.4f}")
print("-" * 66)
print(f"Figura guardada: {nombre_figura}")
print("Resultados guardados en resultados.json")
