import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Datos del enunciado (Tarea B)
# ----------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)

T = 1.0           # temperatura de la Parte 1 usada en la Parte 2
p_nucleo = 0.9    # umbral top-p de la Parte 2
n_muestras = 10000
semilla = 0

# ----------------------------------------------------------------------
# Paso 1: softmax con temperatura (numéricamente estable) -> distribución T=1
# ----------------------------------------------------------------------
z = logits / T
z = z - np.max(z)                 # estabilidad numérica
exp_z = np.exp(z)
probs_T1 = exp_z / np.sum(exp_z)

# ----------------------------------------------------------------------
# Paso 2: muestreo de núcleo (top-p, p=0.9) y renormalización (Parte 2)
# ----------------------------------------------------------------------
orden = np.argsort(-probs_T1, kind="stable")     # orden descendente
probs_ordenadas = probs_T1[orden]
acumulada = np.cumsum(probs_ordenadas)

# conjunto mínimo cuya probabilidad acumulada alcanza p
k = int(np.searchsorted(acumulada, p_nucleo, side="left")) + 1
idx_nucleo = orden[:k]
masa_nucleo = float(probs_ordenadas[:k].sum())

p_renorm_nucleo = probs_ordenadas[:k] / masa_nucleo

# distribución renormalizada sobre el vocabulario completo (ceros fuera del núcleo)
p_teorica = np.zeros(len(tokens), dtype=float)
p_teorica[idx_nucleo] = p_renorm_nucleo
p_teorica = p_teorica / p_teorica.sum()   # normalización exacta para rng.choice

# entropía teórica de la distribución renormalizada (bits)
mask_teor = p_teorica > 0
entropia_teorica = float(-np.sum(p_teorica[mask_teor] * np.log2(p_teorica[mask_teor])))

# ----------------------------------------------------------------------
# Paso 3: 10000 muestras de la distribución renormalizada con default_rng(0)
# ----------------------------------------------------------------------
rng = np.random.default_rng(semilla)
muestras = rng.choice(len(tokens), size=n_muestras, p=p_teorica)

conteos = np.bincount(muestras, minlength=len(tokens))
frecuencias = conteos / n_muestras

# ----------------------------------------------------------------------
# Paso 4: divergencia KL (bits) de la empírica respecto de la teórica
#   KL(P_emp || P_teor) = sum_i p_emp_i * log2(p_emp_i / p_teor_i)
#   (términos con p_emp_i = 0 contribuyen 0 por convención)
# ----------------------------------------------------------------------
mask_emp = frecuencias > 0
kl_bits = float(
    np.sum(frecuencias[mask_emp] * np.log2(frecuencias[mask_emp] / p_teorica[mask_emp]))
)

entropia_empirica = float(-np.sum(frecuencias[mask_emp] * np.log2(frecuencias[mask_emp])))

errores_abs = np.abs(frecuencias - p_teorica)
error_max = float(errores_abs.max())
error_medio = float(errores_abs.mean())

# ----------------------------------------------------------------------
# Figura: barras agrupadas, frecuencias observadas vs probabilidades teóricas
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(tokens))
ancho = 0.38

b1 = ax.bar(x - ancho / 2, frecuencias, ancho,
            label="Frecuencia observada (empírica)",
            color="#4C72B0", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + ancho / 2, p_teorica, ancho,
            label="Probabilidad teórica (top-p renormalizada)",
            color="#DD8452", edgecolor="black", linewidth=0.5)

for barras in (b1, b2):
    for b in barras:
        h = b.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(tokens)
ax.set_xlabel("Token")
ax.set_ylabel("Probabilidad")
ax.set_ylim(0, max(frecuencias.max(), p_teorica.max()) * 1.18)
ax.set_title(f"Parte 3 - Comprobación empírica del muestreo top-p\n"
             f"n = {n_muestras}, semilla = {semilla}, p = {p_nucleo}   |   "
             f"KL(empírica || teórica) = {kl_bits:.4f} bits")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("t3_frecuencias_vs_teoricas.png", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------
# Resultados -> resultados.json
# ----------------------------------------------------------------------
resultados = {
    "n_muestras": int(n_muestras),
    "semilla": int(semilla),
    "tokens": tokens,
    "logits": [float(v) for v in logits],
    "temperatura": float(T),
    "p_nucleo": float(p_nucleo),
    "tokens_sobrevivientes": [tokens[i] for i in idx_nucleo],
    "tamano_nucleo": int(k),
    "masa_nucleo_antes_renormalizar": masa_nucleo,
    "distribucion_renormalizada_teorica": {t: float(p) for t, p in zip(tokens, p_teorica)},
    "conteos_observados": {t: int(c) for t, c in zip(tokens, conteos)},
    "frecuencias_observadas": {t: float(f) for t, f in zip(tokens, frecuencias)},
    "error_absoluto_por_token": {t: float(e) for t, e in zip(tokens, errores_abs)},
    "error_absoluto_maximo": error_max,
    "error_absoluto_medio": error_medio,
    "suma_frecuencias_observadas": float(frecuencias.sum()),
    "entropia_teorica_bits": entropia_teorica,
    "entropia_empirica_bits": entropia_empirica,
    "kl_empirica_respecto_teorica_bits": kl_bits,
    "figura": "t3_frecuencias_vs_teoricas.png",
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# Salida por consola (4 decimales)
# ----------------------------------------------------------------------
print("Parte 3 - Comprobación empírica del muestreo top-p")
print(f"n = {n_muestras} muestras, numpy.random.default_rng({semilla}), p = {p_nucleo}")
print(f"Tokens sobrevivientes (núcleo): {[tokens[i] for i in idx_nucleo]}")
print()
print("Token   Teórica   Observada   Conteo   |Error abs.|")
print("----------------------------------------------------")
for i, t in enumerate(tokens):
    print(f"{t:5s}   {p_teorica[i]:.4f}    {frecuencias[i]:.4f}    {conteos[i]:5d}     {errores_abs[i]:.4f}")
print("----------------------------------------------------")
print(f"Suma frecuencias observadas: {frecuencias.sum():.4f}")
print()
print("Cifras principales:")
print(f"Divergencia KL(empírica || teórica) = {kl_bits:.4f} bits")
print(f"Entropía teórica (renormalizada):  {entropia_teorica:.4f} bits")
print(f"Entropía empírica:                 {entropia_empirica:.4f} bits")
print(f"Error absoluto máximo por token:   {error_max:.4f}")
print(f"Error absoluto medio:              {error_medio:.4f}")
print("Figura guardada: t3_frecuencias_vs_teoricas.png")
print("Resultados guardados en resultados.json")
