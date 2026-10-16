# Parte 4 — evidencia y notas para el informe

Mediciones: [`../parte4_mediciones.json`](../parte4_mediciones.json) (de `spans_base.jsonl`,
`spans_v2.jsonl` y los dos `resultados_*.csv`). Resolución del golden set: 32 casos → cada caso
vale **3,1 %**; ningún umbral de tasa puede estar por debajo de eso.

## 4.b — Tres alertas (umbral sacado de nuestras mediciones)

| Alerta | Medido (base, v1) | Umbral | Ventana | Qué mirar primero en los spans |
|---|---|---|---|---|
| **Latencia p95 de la corrida** (span `corrida`) | p50 1,77 s · p95 2,76 s · máx 4,72 s | **> 6 s** (≈ 2 × p95 y por encima del máximo observado) | últimas 30 corridas | los spans `llm`: si su duración subió pero los tokens de salida no, es el servidor (tokens/s); si subieron los tokens, es el prompt o la pregunta |
| **Tokens por corrida p95** (suma de los `llm`) | p50 5 188 · p95 7 548 · máx 8 017; llamadas al modelo: máx 4 | **> 10 000** (≈ 1,3 × p95; el freno de 120 000 queda muy lejos) | últimas 30 corridas | cuántos `llm` hay por corrida (más de 4 → bucle o repetición); `tamano_resultado` de las `herramienta` (una consulta que devuelve mucho); `prompt_version` del span `corrida` |
| **Tasa de bloqueos de entrada** (span `guardrail` `inyeccion_entrada` = bloqueada) | 3/32 = 9,4 % en el golden set, que tiene 5 adversariales de propósito | **> 20 %** (en 50 corridas son 10 bloqueos; cada corrida vale 2 %, muy por encima de la resolución) | últimas 50 corridas | la `regla` que saltó y el texto ya redactado del span: si es `escribir_en_base` sobre preguntas normales, es un falso positivo como L01/L03 de la 0.b; si son ataques reales, avisar |

### Por qué la primera alerta ya hubiera saltado
En la corrida con el prompt v2 la latencia p95 de la corrida fue **42,24 s** (máx 48,31 s): la
alerta de 6 s salta. Lo primero que dice la tabla es que los tokens por corrida casi no cambiaron
(p95 7 548 → 8 165) y que las llamadas al modelo por corrida tampoco (máx 4 en las dos); lo que
cambió es la velocidad del servidor: mediana de tokens de salida por segundo en los `llm`
**180,9 → 99,8**, y el percentil 5 **92,6 → 15,9**. La llamada más lenta tardó 43,9 s para 135
tokens. O sea: no fue el prompt, fue la H200. Sin los spans `llm` con sus tokens y su duración,
esta alerta habría llevado a revertir el prompt v2 por una causa que no era suya.
