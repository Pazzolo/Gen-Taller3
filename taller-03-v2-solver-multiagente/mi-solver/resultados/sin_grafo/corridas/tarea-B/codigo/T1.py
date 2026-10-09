# -*- coding: utf-8 -*-
"""
SUBTAREA T1: Softmax con temperatura (numéricamente estable) en NumPy.

Logits del enunciado para los tokens t0–t5: [2.0, 1.0, 0.5, 0.2, -1.0, -3.0]
Se calcula la distribución p_i = exp(z_i/T) / Σ_j exp(z_j/T) para T = 0.5, 1 y 2,
la entropía en bits H = -Σ p log2 p de cada una, y se presenta todo en una tabla
con cuatro decimales. Todas las cifras se guardan en "resultados.json".
No se requiere figura.
"""

import json
import numpy as np

# ----------------------------------------------------------------------
# Datos del enunciado
# ----------------------------------------------------------------------
tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)

temperaturas = [0.5, 1.0, 2.0]


# ----------------------------------------------------------------------
# Funciones
# ----------------------------------------------------------------------
def softmax_con_temperatura(z, T):
    """
    Softmax con temperatura, numéricamente estable.

    p_i = exp(z_i / T) / Σ_j exp(z_j / T)

    Estabilidad: se resta el máximo de los logits escalados antes de la
    exponencial, de modo que el mayor exponente es 0 y no hay desbordamiento.
    """
    z = np.asarray(z, dtype=float)
    T = float(T)
    if T <= 0:
        raise ValueError("La temperatura T debe ser positiva.")
    z_escalado = z / T
    z_escalado = z_escalado - np.max(z_escalado)  # resta del máximo (estabilidad)
    e = np.exp(z_escalado)
    return e / np.sum(e)


def entropia_bits(p):
    """Entropía de Shannon en bits: H = -Σ p log2 p (términos con p=0 aportan 0)."""
    p = np.asarray(p, dtype=float)
    p_nz = p[p > 0]
    return float(-np.sum(p_nz * np.log2(p_nz)))


# ----------------------------------------------------------------------
# Cálculo de distribuciones y entropías
# ----------------------------------------------------------------------
distribuciones = {}
entropias = {}

for T in temperaturas:
    p = softmax_con_temperatura(logits, T)
    H = entropia_bits(p)
    clave = f"T={T}"
    distribuciones[clave] = {tok: float(pi) for tok, pi in zip(tokens, p)}
    entropias[clave] = float(H)

# ----------------------------------------------------------------------
# Guardar TODAS las cifras calculadas en resultados.json (floats de Python)
# ----------------------------------------------------------------------
resultados = {
    "tokens": tokens,
    "logits": [float(z) for z in logits],
    "temperaturas": [float(T) for T in temperaturas],
    "distribuciones": distribuciones,
    "entropias_bits": entropias,
}
# Sumas de verificación (deben ser 1.0)
for T in temperaturas:
    clave = f"T={T}"
    resultados[f"suma_probabilidades_{clave}"] = float(
        sum(distribuciones[clave].values())
    )

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# ----------------------------------------------------------------------
# Tabla con cuatro decimales
# ----------------------------------------------------------------------
anchos = 12
linea_sep = "-" * (8 + anchos * len(temperaturas))

print("Softmax con temperatura (distribuciones y entropía, 4 decimales)")
print(linea_sep)
encabezado = "Token".ljust(8) + "".join(f"T={T}".rjust(anchos) for T in temperaturas)
print(encabezado)
print(linea_sep)
for tok in tokens:
    fila = tok.ljust(8)
    for T in temperaturas:
        fila += f"{distribuciones[f'T={T}'][tok]:.4f}".rjust(anchos)
    print(fila)
print(linea_sep)
fila_H = "H (bits)".ljust(8)
for T in temperaturas:
    fila_H += f"{entropias[f'T={T}']:.4f}".rjust(anchos)
print(fila_H)
print(linea_sep)

# ----------------------------------------------------------------------
# Cifras principales con 4 decimales
# ----------------------------------------------------------------------
print("\nCifras principales:")
print(f"Logits: {[f'{z:.4f}' for z in logits]}")
for T in temperaturas:
    clave = f"T={T}"
    probs = distribuciones[clave]
    print(
        f"T={T}: p = [{', '.join(f'{probs[tok]:.4f}' for tok in tokens)}], "
        f"H = {entropias[clave]:.4f} bits"
    )

print("\nResultados guardados en resultados.json")
