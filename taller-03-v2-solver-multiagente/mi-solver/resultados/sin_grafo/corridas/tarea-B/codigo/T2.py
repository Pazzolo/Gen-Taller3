# -*- coding: utf-8 -*-
"""
SUBTAREA T2: Muestreo de núcleo (top-p) con p=0.9 sobre la distribución T=1 de la Parte 1.

Pasos:
1. Recalcular la distribución softmax con T=1 a partir de los logits de la Parte 1
   (el script es independiente: no lee archivos de subtareas anteriores).
2. Ordenar los tokens por probabilidad descendente.
3. Conservar el conjunto mínimo cuya probabilidad acumulada alcanza p=0.9.
4. Renormalizar las probabilidades de los tokens supervivientes.
5. Guardar todas las cifras en resultados.json e imprimirlas con 4 decimales.

La subtarea indica "Figura: no", por lo que no se genera ningún PNG.
"""

import json
import numpy as np


def main():
    # ------------------------------------------------------------------
    # Datos de la Parte 1 (subtarea T1): logits y tokens
    # ------------------------------------------------------------------
    tokens = ["t0", "t1", "t2", "t3", "t4", "t5"]
    logits = np.array([2.0, 1.0, 0.5, 0.2, -1.0, -3.0], dtype=float)
    temperatura = 1.0
    p_nucleo = 0.9

    # ------------------------------------------------------------------
    # 1. Softmax con temperatura T=1 (recomputado, con estabilidad numérica)
    # ------------------------------------------------------------------
    z = logits / temperatura
    z = z - np.max(z)
    exp_z = np.exp(z)
    probs = exp_z / np.sum(exp_z)

    # ------------------------------------------------------------------
    # 2. Ordenar tokens por probabilidad descendente
    # ------------------------------------------------------------------
    orden = np.argsort(-probs, kind="stable")
    tokens_ordenados = [tokens[i] for i in orden]
    probs_ordenadas = probs[orden]
    acumuladas = np.cumsum(probs_ordenadas)

    # ------------------------------------------------------------------
    # 3. Conjunto mínimo cuya probabilidad acumulada alcanza p=0.9
    #    (se incluye el token que hace cruzar el umbral)
    # ------------------------------------------------------------------
    idx_cruce = int(np.searchsorted(acumuladas, p_nucleo, side="left"))
    k = min(idx_cruce + 1, len(tokens))  # tamaño del núcleo
    indices_nucleo = orden[:k]
    tokens_supervivientes = [tokens[i] for i in indices_nucleo]
    tokens_eliminados = [tokens[i] for i in orden[k:]]
    masa_nucleo = float(acumuladas[k - 1])
    masa_eliminada = float(1.0 - masa_nucleo)

    # ------------------------------------------------------------------
    # 4. Renormalizar las probabilidades del núcleo
    # ------------------------------------------------------------------
    probs_nucleo = probs[indices_nucleo]
    renorm = probs_nucleo / masa_nucleo

    # Distribución renormalizada en longitud completa (0.0 para eliminados)
    renorm_completa = np.zeros(len(tokens), dtype=float)
    renorm_completa[indices_nucleo] = renorm

    # ------------------------------------------------------------------
    # Cifras adicionales (se guardan todas)
    # ------------------------------------------------------------------
    suma_T1 = float(np.sum(probs))
    suma_renorm = float(np.sum(renorm))
    entropia_T1_bits = float(-np.sum(probs * np.log2(probs)))
    entropia_renorm_bits = float(-np.sum(renorm * np.log2(renorm)))

    # ------------------------------------------------------------------
    # Guardar TODAS las cifras en resultados.json (un solo objeto)
    # ------------------------------------------------------------------
    resultados = {
        "tokens": tokens,
        "logits": [float(x) for x in logits],
        "temperatura": float(temperatura),
        "p_nucleo": float(p_nucleo),
        "distribucion_T1": {t: float(v) for t, v in zip(tokens, probs)},
        "suma_distribucion_T1": suma_T1,
        "entropia_T1_bits": entropia_T1_bits,
        "orden_descendente_tokens": tokens_ordenados,
        "orden_descendente_probs": [float(v) for v in probs_ordenadas],
        "probs_acumuladas_ordenadas": [float(v) for v in acumuladas],
        "num_tokens_nucleo": int(k),
        "tokens_supervivientes": tokens_supervivientes,
        "tokens_eliminados": tokens_eliminados,
        "masa_nucleo_antes_renormalizar": masa_nucleo,
        "masa_eliminada": masa_eliminada,
        "distribucion_renormalizada": {
            t: float(v) for t, v in zip(tokens_supervivientes, renorm)
        },
        "distribucion_renormalizada_completa": {
            t: float(v) for t, v in zip(tokens, renorm_completa)
        },
        "suma_distribucion_renormalizada": suma_renorm,
        "entropia_distribucion_renormalizada_bits": entropia_renorm_bits,
        "figuras": [],
    }

    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Impresión de las cifras principales (4 decimales)
    # ------------------------------------------------------------------
    linea = "-" * 64
    print("Top-p (muestreo de núcleo) con p=0.9 sobre la distribución T=1")
    print(linea)
    print("Distribución softmax con T=1 (recomputada de los logits):")
    for t, v in zip(tokens, probs):
        print(f"  {t}: {v:.4f}")
    print(f"  suma = {suma_T1:.4f}")
    print(linea)
    print("Tokens ordenados por probabilidad descendente:")
    print(f"  {'token':<7}{'p':>10}{'acumulada':>12}   estado")
    for j, (t, v, c) in enumerate(zip(tokens_ordenados, probs_ordenadas, acumuladas)):
        marca = ""
        if j < k:
            marca = "* en el núcleo"
        if j == k - 1:
            marca += " (aquí se alcanza p=0.9)"
        print(f"  {t:<7}{v:>10.4f}{c:>12.4f}   {marca}")
    print(linea)
    print(f"Conjunto mínimo con probabilidad acumulada >= {p_nucleo}:")
    print(f"  Tokens supervivientes (k={k}): {tokens_supervivientes}")
    print(f"  Tokens eliminados ({len(tokens) - k}): {tokens_eliminados}")
    print(f"  Masa del núcleo antes de renormalizar: {masa_nucleo:.4f}")
    print(f"  Masa eliminada: {masa_eliminada:.4f}")
    print(linea)
    print("Distribución renormalizada (núcleo top-p):")
    for t, v in zip(tokens_supervivientes, renorm):
        print(f"  {t}: {v:.4f}")
    print(f"  suma = {suma_renorm:.4f}")
    print(f"  entropía = {entropia_renorm_bits:.4f} bits "
          f"(T=1 original: {entropia_T1_bits:.4f} bits)")
    print(linea)
    print("Resultados guardados en resultados.json")


if __name__ == "__main__":
    main()
