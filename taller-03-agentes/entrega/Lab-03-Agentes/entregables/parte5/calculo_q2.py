"""Parte 5, pregunta 2: costo acumulado de entrada si se duplica el máximo de pasos.

Modelo: la llamada k cuesta P0 + (k - 1) * D tokens de entrada, porque el historial se reenvía
completo y cada paso le añade una observación. Acumulado en n llamadas: n * P0 + D * n(n - 1) / 2.
P0 y D salen de las 92 trazas de MiAgente sobre el golden set (zai-org/GLM-5.3-Flash, 2026-10-05):
primera llamada mediana 1 142; incremento por paso mediana 481 y máximo observado 707.
"""
P0 = 1142
for etiqueta, D in (("incremento mediano", 481), ("peor incremento observado", 707)):
    a8 = 8 * P0 + D * 8 * 7 // 2
    a16 = 16 * P0 + D * 16 * 15 // 2
    print(f"{etiqueta} (D={D}): 8 pasos → {a8:,} · 16 pasos → {a16:,} · razón {a16 / a8:.2f}x")
print("presupuesto por corrida (freno 2): 120,000 tokens de entrada + salida")
