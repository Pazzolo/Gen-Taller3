import json
import numpy as np

# ----------------------------------------------------------------------
# SUBTAREA T1: Softmax con temperatura (numéricamente estable) y entropía
# Datos del enunciado: vocabulario de seis tokens t0–t5
# ----------------------------------------------------------------------

tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
temperaturas = [0.5, 1.0, 2.0]


def softmax_con_temperatura(z, T):
    """Softmax con temperatura p_i = exp(z_i/T) / sum_j exp(z_j/T).

    Numéricamente estable: se resta el máximo de los logits escalados
    antes de aplicar exp, para evitar desbordamiento.
    """
    z = np.asarray(z, dtype=float) / float(T)
    z = z - np.max(z)  # estabilidad numérica
    e = np.exp(z)
    return e / np.sum(e)


def entropia_bits(p):
    """Entropía de Shannon en bits: H = -sum p_i * log2(p_i)."""
    p = np.asarray(p, dtype=float)
    mascara = p > 0  # evita log2(0); 0*log2(0) se define como 0
    return float(-np.sum(p[mascara] * np.log2(p[mascara])))


# ----------------------------------------------------------------------
# Cálculo de distribuciones y entropías para cada temperatura
# ----------------------------------------------------------------------
resultados = {
    "tokens": tokens,
    "logits": [float(x) for x in logits],
    "temperaturas": [float(T) for T in temperaturas],
    "distribuciones": {},
    "entropias_bits": {},
    "suma_distribucion": {},
    "prob_max": {},
}

for T in temperaturas:
    p = softmax_con_temperatura(logits, T)
    H = entropia_bits(p)
    clave = f"T={T}"
    resultados["distribuciones"][clave] = {
        tok: float(pi) for tok, pi in zip(tokens, p)
    }
    resultados["entropias_bits"][clave] = float(H)
    resultados["suma_distribucion"][clave] = float(np.sum(p))
    resultados["prob_max"][clave] = float(np.max(p))

# ----------------------------------------------------------------------
# Guardar todas las cifras (sin redondear) en resultados.json
# ----------------------------------------------------------------------
with open("resultados.json", "w") as f:
    json.dump(resultados, f, indent=2)

# ----------------------------------------------------------------------
# Tabla con cuatro decimales: distribuciones y entropías
# ----------------------------------------------------------------------
col_w = 10
encabezado = "Token ".ljust(7) + "".join(f"T={T}".rjust(col_w) for T in temperaturas)
separador = "-" * len(encabezado)

print("Softmax con temperatura sobre logits [2.0, 1.0, 0.5, 0.2, -1.0, -3.0]")
print()
print(encabezado)
print(separador)
for i, tok in enumerate(tokens):
    fila = f"{tok}".ljust(7)
    for T in temperaturas:
        fila += f"{resultados['distribuciones'][f'T={T}'][tok]:.4f}".rjust(col_w)
    print(fila)
print(separador)
fila_suma = "Suma  ".ljust(7)
for T in temperaturas:
    fila_suma += f"{resultados['suma_distribucion'][f'T={T}']:.4f}".rjust(col_w)
print(fila_suma)
fila_H = "H (bits)".ljust(7)
for T in temperaturas:
    fila_H += f"{resultados['entropias_bits'][f'T={T}']:.4f}".rjust(col_w)
print(fila_H)
print(separador)

# Cifras principales con 4 decimales
print()
print("Cifras principales:")
print(f"Logits: {[float(x) for x in logits]}")
for T in temperaturas:
    clave = f"T={T}"
    dist_str = ", ".join(
        f"{tok}={resultados['distribuciones'][clave][tok]:.4f}" for tok in tokens
    )
    print(f"T={T}: p=[{dist_str}]")
    print(
        f"        H={resultados['entropias_bits'][clave]:.4f} bits, "
        f"p_max={resultados['prob_max'][clave]:.4f}, "
        f"suma={resultados['suma_distribucion'][clave]:.4f}"
    )
