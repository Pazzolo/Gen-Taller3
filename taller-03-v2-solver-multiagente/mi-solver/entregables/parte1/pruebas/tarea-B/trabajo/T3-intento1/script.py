# -*- coding: utf-8 -*-
"""
SUBTAREA T3 — Comprobación empírica del muestreo top-p (Tarea B, Parte 3)

Pasos:
  1) Recalcula la distribución softmax con T = 1 (Parte 1) a partir de los logits.
  2) Recalcula el muestreo de núcleo top-p con p = 0.9 y la renormalización (Parte 2).
  3) Toma 10000 muestras de la distribución renormalizada con numpy.random.default_rng(0).
  4) Compara frecuencias observadas vs probabilidades teóricas en un gráfico de barras (PNG).
  5) Calcula la divergencia KL (en bits) de la distribución empírica respecto de la teórica.
  6) Guarda todas las cifras en 'resultados.json' e imprime las principales con 4 decimales.
"""

import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Datos del enunciado
# ----------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)

# ----------------------------------------------------------------------
# 2. Parte 1 (recalculada): softmax con temperatura, numéricamente estable
# ----------------------------------------------------------------------
def softmax_con_temperatura(z, T):
    z = np.asarray(z, dtype=float)
    escalados = z / float(T)
    escalados = escalados - np.max(escalados)  # truco de estabilidad numérica
    expz = np.exp(escalados)
    return expz / np.sum(expz)

T = 1.0
p_T1 = softmax_con_temperatura(logits, T)

# ----------------------------------------------------------------------
# 3. Parte 2 (recalculada): top-p (nucleus) con p = 0.9 y renormalización
# ----------------------------------------------------------------------
p_objetivo = 0.9
orden_desc = np.argsort(-p_T1, kind="stable")  # índices por probabilidad descendente

acum = 0.0
indices_nucleo = []
for i in orden_desc:
    indices_nucleo.append(int(i))
    acum += float(p_T1[i])
    if acum >= p_objetivo:  # conjunto mínimo cuya acumulada alcanza 0.9
        break

suma_nucleo = float(np.sum(p_T1[indices_nucleo]))

q_teorica = np.zeros(len(tokens), dtype=float)
q_teorica[indices_nucleo] = p_T1[indices_nucleo] / suma_nucleo  # renormalización

# ----------------------------------------------------------------------
# 4. Parte 3: 10000 muestras de la distribución renormalizada con default_rng(0)
# ----------------------------------------------------------------------
N = 10000
semilla = 0
rng = np.random.default_rng(semilla)

q_muestreo = q_teorica / np.sum(q_teorica)  # garantiza suma exactamente 1 para rng.choice
muestras = rng.choice(len(tokens), size=N, p=q_muestreo)

conteos = np.bincount(muestras, minlength=len(tokens)).astype(int)
frecuencias_obs = conteos / float(N)

# ----------------------------------------------------------------------
# 5. Divergencia KL (en bits) de la empírica respecto de la teórica
#    KL(q || p) = sum_i q_i * log2(q_i / p_i),  con la convención 0*log(0/.) = 0
# ----------------------------------------------------------------------
def kl_bits(q, p):
    kl = 0.0
    for qi, pi in zip(q, p):
        if qi > 0.0:
            if pi <= 0.0:
                return float("inf")
            kl += qi * math.log2(qi / pi)
    return float(kl)

def entropia_bits(p):
    h = 0.0
    for pi in p:
        if pi > 0.0:
            h -= pi * math.log2(pi)
    return float(h)

kl_emp_teo = kl_bits(frecuencias_obs, q_teorica)
H_teorica = entropia_bits(q_teorica)
H_empirica = entropia_bits(frecuencias_obs)

diferencias = frecuencias_obs - q_teorica
error_abs_max = float(np.max(np.abs(diferencias)))
tokens_sin_muestras = [tokens[i] for i in range(len(tokens)) if conteos[i] == 0]

# ----------------------------------------------------------------------
# 6. Figura: barras de frecuencias observadas vs probabilidades teóricas
# ----------------------------------------------------------------------
nombre_figura = "t3_frecuencias_vs_teoricas.png"
x = np.arange(len(tokens))
ancho = 0.38

fig, ax = plt.subplots(figsize=(8.5, 5.2))
b_obs = ax.bar(x - ancho / 2.0, frecuencias_obs, ancho,
               label="Frecuencia observada (10000 muestras)",
               color="#4C72B0", edgecolor="black", linewidth=0.6)
b_teo = ax.bar(x + ancho / 2.0, q_teorica, ancho,
               label="Probabilidad teórica (top-p p=0.9 renormalizada)",
               color="#DD8452", edgecolor="black", linewidth=0.6)

for barras in (b_obs, b_teo):
    for b in barras:
        h = b.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(b.get_x() + b.get_width() / 2.0, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad / frecuencia")
ax.set_ylim(0, max(np.max(frecuencias_obs), np.max(q_teorica)) * 1.18)
ax.set_title("Parte 3 — Frecuencias observadas vs probabilidades teóricas\n"
             f"top-p (p=0.9) sobre T=1, N={N}, semilla={semilla} · "
             f"$D_{{KL}}$(empírica||teórica) = {kl_emp_teo:.4f} bits")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3, linestyle="--")

fig.tight_layout()
fig.savefig(nombre_figura, dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# 7. Guardar TODAS las cifras en resultados.json (objeto, floats sin redondear)
# ----------------------------------------------------------------------
cifras_t3 = {
    "subtarea": ("T3: comprobacion empirica — 10000 muestras de la distribucion renormalizada "
                 "top-p (p=0.9 sobre T=1) con numpy.random.default_rng(0); comparacion de "
                 "frecuencias observadas vs probabilidades teoricas y divergencia KL en bits"),
    "tokens": tokens,
    "logits": [float(z) for z in logits],
    "temperatura": float(T),
    "p_top_p": float(p_objetivo),
    "distribucion_T1_original": {tokens[i]: float(p_T1[i]) for i in range(len(tokens))},
    "tokens_nucleo": [tokens[i] for i in indices_nucleo],
    "num_tokens_nucleo": int(len(indices_nucleo)),
    "suma_prob_nucleo_antes_renorm": float(suma_nucleo),
    "distribucion_renormalizada_teorica": {tokens[i]: float(q_teorica[i]) for i in range(len(tokens))},
    "n_muestras": int(N),
    "semilla": int(semilla),
    "generador": "numpy.random.default_rng(0)",
    "conteos_observados": {tokens[i]: int(conteos[i]) for i in range(len(tokens))},
    "frecuencias_observadas": {tokens[i]: float(frecuencias_obs[i]) for i in range(len(tokens))},
    "diferencia_obs_menos_teorica": {tokens[i]: float(diferencias[i]) for i in range(len(tokens))},
    "error_absoluto_maximo": float(error_abs_max),
    "suma_frecuencias_observadas": float(np.sum(frecuencias_obs)),
    "tokens_sin_muestras": tokens_sin_muestras,
    "kl_empirica_respecto_teorica_bits": float(kl_emp_teo),
    "entropia_bits_distribucion_teorica": float(H_teorica),
    "entropia_bits_distribucion_empirica": float(H_empirica),
    "figura": nombre_figura,
}

try:
    with open("resultados.json", "r", encoding="utf-8") as f:
        previo = json.load(f)
    if not isinstance(previo, dict):
        previo = {}
except (FileNotFoundError, OSError, ValueError):
    previo = {}

previo["T3_comprobacion_empirica"] = cifras_t3

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(previo, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# 8. Impresión de las cifras principales (4 decimales)
# ----------------------------------------------------------------------
print("Parte 3 — Comprobacion empirica del muestreo top-p")
print("--------------------------------------------------")
print(f"Nucleo top-p (p=0.9): {[tokens[i] for i in indices_nucleo]} "
      f"(suma original = {suma_nucleo:.4f})")
print(f"Muestreo: N = {N} muestras, generador numpy.random.default_rng({semilla})")
print("--------------------------------------------------")
print("token   conteo   frec_obs   p_teorica    diff")
for i in range(len(tokens)):
    print(f"  {tokens[i]}  {conteos[i]:8d}   {frecuencias_obs[i]:.4f}     "
          f"{q_teorica[i]:.4f}   {diferencias[i]:+.4f}")
print("--------------------------------------------------")
print(f"D_KL(empirica || teorica) = {kl_emp_teo:.4f} bits")
print(f"Entropia teorica = {H_teorica:.4f} bits | Entropia empirica = {H_empirica:.4f} bits")
print(f"Error absoluto maximo |frec_obs - p_teorica| = {error_abs_max:.4f}")
print(f"Tokens sin muestras: {tokens_sin_muestras if tokens_sin_muestras else 'ninguno'}")
print(f"Figura guardada en: {nombre_figura}")
print("Resultados guardados en 'resultados.json'.")
