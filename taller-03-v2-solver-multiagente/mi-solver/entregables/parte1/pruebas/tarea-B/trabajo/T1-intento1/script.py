# -*- coding: utf-8 -*-
"""
SUBTAREA T1 — Softmax con temperatura y entropía en bits
Tarea B — Temperatura, top-p y entropía de la distribución de salida (Parte 1)

Define el vector de logits de los seis tokens t0..t5, implementa la softmax
con temperatura de forma numéricamente estable (restando el máximo), calcula
la distribución para T = 0.5, 1 y 2, y la entropía en bits de cada una.
Presenta una tabla con cuatro decimales y guarda todas las cifras en
"resultados.json". No se requiere figura.
"""

import json
import numpy as np

# ----------------------------------------------------------------------
# 1. Datos del enunciado
# ----------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)

# ----------------------------------------------------------------------
# 2. Softmax con temperatura, numéricamente estable
#    p_i = exp(z_i / T) / sum_j exp(z_j / T)
#    Estabilidad: se resta el máximo de los logits escalados antes de exp,
#    lo que evita desbordamientos sin cambiar el resultado (factor común).
# ----------------------------------------------------------------------
def softmax_con_temperatura(z, T):
    z = np.asarray(z, dtype=float)
    if T <= 0:
        raise ValueError("La temperatura T debe ser positiva.")
    escalados = z / T
    escalados = escalados - np.max(escalados)  # resta del máximo (estabilidad)
    exps = np.exp(escalados)
    return exps / np.sum(exps)

# ----------------------------------------------------------------------
# 3. Entropía en bits: H = -sum_i p_i * log2(p_i)
# ----------------------------------------------------------------------
def entropia_bits(p):
    p = np.asarray(p, dtype=float)
    mascara = p > 0  # los términos con p_i = 0 aportan 0 a la entropía
    return float(-np.sum(p[mascara] * np.log2(p[mascara])))

# ----------------------------------------------------------------------
# 4. Cálculo para T = 0.5, 1 y 2
# ----------------------------------------------------------------------
temperaturas = [0.5, 1.0, 2.0]
distribuciones = {}
entropias = {}
sumas = {}

for T in temperaturas:
    p = softmax_con_temperatura(logits, T)
    distribuciones[T] = p
    entropias[T] = entropia_bits(p)
    sumas[T] = float(np.sum(p))  # verificación: debe ser 1.0

# ----------------------------------------------------------------------
# 5. Tabla con cuatro decimales (distribuciones y entropías)
# ----------------------------------------------------------------------
print("Tabla: distribuciones softmax con temperatura y entropia en bits")
print("-" * 70)
encabezado = f"{'T':<6}" + "".join(f"{tok:>9}" for tok in tokens) + f"{'H (bits)':>12}"
print(encabezado)
print("-" * 70)
for T in temperaturas:
    fila = f"{T:<6}" + "".join(f"{p:>9.4f}" for p in distribuciones[T])
    fila += f"{entropias[T]:>12.4f}"
    print(fila)
print("-" * 70)

# ----------------------------------------------------------------------
# 6. Guardar TODAS las cifras calculadas en resultados.json
# ----------------------------------------------------------------------
resultados = {
    "tokens": tokens,
    "logits": [float(z) for z in logits],
    "temperaturas": [float(T) for T in temperaturas],
    "distribuciones_softmax": {
        f"T={T}": {tok: float(p) for tok, p in zip(tokens, distribuciones[T])}
        for T in temperaturas
    },
    "suma_distribuciones": {f"T={T}": float(sumas[T]) for T in temperaturas},
    "entropia_bits": {f"T={T}": float(entropias[T]) for T in temperaturas},
}

with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# ----------------------------------------------------------------------
# 7. Cifras principales con 4 decimales
# ----------------------------------------------------------------------
print("\nCifras principales:")
for T in temperaturas:
    probs_str = ", ".join(f"{tok}={p:.4f}" for tok, p in zip(tokens, distribuciones[T]))
    print(f"T={T}: p = [{probs_str}]  |  H = {entropias[T]:.4f} bits  |  suma = {sumas[T]:.4f}")

print("\nResultados guardados en 'resultados.json'.")
