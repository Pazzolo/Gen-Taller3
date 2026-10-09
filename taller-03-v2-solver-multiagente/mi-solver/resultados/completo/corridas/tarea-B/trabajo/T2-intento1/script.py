import json
import numpy as np

# ============================================================
# SUBTAREA T2: Muestreo de núcleo (top-p) con p = 0.9
# sobre la distribución de T = 1 de la Parte 1 (recalculada)
# ============================================================

# ---------- Datos del enunciado ----------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)

# ---------- Softmax con temperatura (numéricamente estable) ----------
def softmax_con_temperatura(z, T):
    z = np.asarray(z, dtype=float)
    escalados = z / float(T)
    escalados = escalados - escalados.max()  # estabilidad numérica
    ex = np.exp(escalados)
    return ex / ex.sum()

T = 1.0
p = softmax_con_temperatura(logits, T)

def entropia_bits(probs):
    probs = np.asarray(probs, dtype=float)
    mask = probs > 0
    return float(-(probs[mask] * np.log2(probs[mask])).sum())

H_original = entropia_bits(p)

# ---------- Top-p (nucleus) con p = 0.9 ----------
p_nucleo = 0.9

# Ordenar tokens por probabilidad descendente (estable: desempates por índice)
orden = np.argsort(-p, kind="stable")
tokens_ordenados = [tokens[i] for i in orden]
probs_ordenadas = p[orden]
logits_ordenados = logits[orden]
acumulada = np.cumsum(probs_ordenadas)

# Conjunto mínimo cuya probabilidad acumulada alcanza p:
# se añaden tokens en orden descendente hasta que la suma acumulada >= p
# (el token que cruza el umbral se incluye). Tolerancia por punto flotante.
alcanza = acumulada >= (p_nucleo - 1e-12)
if alcanza.any():
    k = int(np.argmax(alcanza)) + 1  # tamaño del núcleo
else:
    k = len(tokens)  # salvaguarda (no debería ocurrir: la masa total es 1)

tokens_sobrevivientes = tokens_ordenados[:k]
probs_nucleo = probs_ordenadas[:k]
masa_nucleo = float(acumulada[k - 1])
masa_antes_del_ultimo = float(acumulada[k - 2]) if k >= 2 else 0.0
probs_renormalizadas = probs_nucleo / masa_nucleo

tokens_excluidos = tokens_ordenados[k:]
masa_excluida = float(probs_ordenadas[k:].sum())

# Distribución renormalizada completa (0.0 para los tokens excluidos)
dist_renorm_completa = {t: 0.0 for t in tokens}
for t, pr in zip(tokens_sobrevivientes, probs_renormalizadas):
    dist_renorm_completa[t] = float(pr)

H_renormalizada = entropia_bits(probs_renormalizadas)

# Verificaciones del criterio de conjunto mínimo
verif_alcanza = bool(masa_nucleo >= p_nucleo - 1e-12)
verif_minima = bool(masa_antes_del_ultimo < p_nucleo - 1e-12) if k >= 2 else True

# ---------- Guardar todas las cifras en resultados.json ----------
resultados = {
    "tokens": tokens,
    "logits": [float(z) for z in logits],
    "temperatura": float(T),
    "p_nucleo": float(p_nucleo),
    "distribucion_T1": {t: float(v) for t, v in zip(tokens, p)},
    "entropia_original_bits": H_original,
    "orden_descendente": tokens_ordenados,
    "logits_ordenados": [float(v) for v in logits_ordenados],
    "probs_ordenadas": [float(v) for v in probs_ordenadas],
    "prob_acumulada_ordenada": [float(v) for v in acumulada],
    "tamano_nucleo": int(k),
    "tokens_sobrevivientes": tokens_sobrevivientes,
    "tokens_excluidos": tokens_excluidos,
    "masa_nucleo_antes_renormalizar": masa_nucleo,
    "masa_acumulada_antes_del_ultimo_token": masa_antes_del_ultimo,
    "masa_excluida": masa_excluida,
    "distribucion_renormalizada": {
        t: float(v) for t, v in zip(tokens_sobrevivientes, probs_renormalizadas)
    },
    "distribucion_renormalizada_completa": dist_renorm_completa,
    "suma_renormalizada": float(probs_renormalizadas.sum()),
    "entropia_renormalizada_bits": H_renormalizada,
    "verificacion_alcanza_p": verif_alcanza,
    "verificacion_conjunto_minimo": verif_minima,
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# ---------- Impresión de las cifras principales (4 decimales) ----------
print("Muestreo de núcleo (top-p) sobre la distribución de T=1")
print(f"Logits: {[float(z) for z in logits]}")
print(f"Temperatura T = {T:.1f}   |   p = {p_nucleo:.1f}")
print()

print("Tokens ordenados por probabilidad descendente:")
print("Token   Prob.      Acumulada   En núcleo")
print("-" * 44)
for t, pr, ac in zip(tokens_ordenados, probs_ordenadas, acumulada):
    dentro = "sí" if t in tokens_sobrevivientes else "no"
    print(f"{t}      {pr:.4f}    {ac:.4f}      {dentro}")
print("-" * 44)

print()
print(f"Conjunto mínimo con masa acumulada >= {p_nucleo:.1f}: "
      f"{tokens_sobrevivientes} (masa = {masa_nucleo:.4f})")
print(f"Tokens excluidos: {tokens_excluidos} "
      f"(masa excluida = {masa_excluida:.4f})")
print()

print("Distribución renormalizada dentro del núcleo:")
print("Token   Original   Renormalizada")
print("-" * 34)
for t, pr, rn in zip(tokens_sobrevivientes, probs_nucleo, probs_renormalizadas):
    print(f"{t}      {pr:.4f}     {rn:.4f}")
print("-" * 34)
print(f"Suma renormalizada: {float(probs_renormalizadas.sum()):.4f}")
print()

print("Cifras principales:")
print(f"Tokens sobrevivientes (top-p, p={p_nucleo:.1f}): {tokens_sobrevivientes}")
print(f"Tamaño del núcleo: {k}")
print(f"Masa del núcleo antes de renormalizar: {masa_nucleo:.4f}")
print("Distribución renormalizada: "
      + ", ".join(f"{t}={v:.4f}" for t, v in zip(tokens_sobrevivientes, probs_renormalizadas)))
print(f"Suma: {float(probs_renormalizadas.sum()):.4f}")
print(f"Entropía original (T=1): {H_original:.4f} bits")
print(f"Entropía renormalizada:  {H_renormalizada:.4f} bits")
