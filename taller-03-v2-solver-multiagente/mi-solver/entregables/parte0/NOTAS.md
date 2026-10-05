# Parte 0 — tres errores que no dan error (v2)

Corrido el 2026-10-05 con los scripts del kit sin modificar (`solver-v2/parte0/`), Python 3.14.7 y
las versiones de `solver-v2/requirements.txt`. 0.a usa la H200 con `zai-org/GLM-5.3-Flash` (leído de
`/v1/models`); 0.b y 0.c no usan modelo.

## 0.a — El solver que no ejecuta ([0a_llm_sin_ejecutar.txt](0a_llm_sin_ejecutar.txt))
```
Medido: NB accuracy 0.9474 · NB F1 macro 0.9429 · LR accuracy 0.9883 · LR F1 macro 0.9875
LLM sin herramientas: zai-org/GLM-5.3-Flash, 12451 tokens de salida, 77.17 s, fin=stop
26 cifras con decimales; procedencia: 0 de 26; ¿dice que son estimadas? no
```
El reporte inventado (`solver-v2/parte0/reporte_sin_ejecutar.md`) contra lo medido. La curva medida
se calculó aquí con el protocolo del enunciado: submuestras estratificadas del entrenamiento con
`random_state=42`, evaluadas siempre sobre el mismo test.

| Fracción (n) | NB inventado | NB medido | LR inventada | LR medida |
|---|---|---|---|---|
| 5 % (~20) | 0,9123 | 0,9415 | 0,8889 | 0,9298 |
| 10 % (~40) | 0,9240 | 0,9474 | 0,9415 | 0,9357 |
| 25 % (~100) | 0,9298 | 0,9474 | 0,9649 | 0,9415 |
| 50 % (~200) | 0,9357 | 0,9357 | 0,9766 | 0,9766 |
| 100 % (398) | 0,9415 | 0,9474 | 0,9825 | 0,9883 |

Hechos para las tres frases:
- **Por qué no basta con que estén cerca:** NB 0,9415 contra 0,9474 es una diferencia de 0,6 puntos,
  que parece ruido. Pero ninguna cifra salió de una ejecución, así que estar cerca es suerte (el
  dataset es famoso), no medición. Además, la cercanía cifra por cifra no protege la conclusión
  (siguiente punto). El modelo repitió exactamente 0,9415, la misma cifra que cita el enunciado del
  taller, lo que sugiere un valor «memorizado» y no calculado.
- **Qué cifra inventada cambia una conclusión:** LR al 5 % = 0,8889, frente a 0,9298 medido. Con
  ella el reporte dice que **el cruce de Ng y Jordan ocurre entre 20 y 40 ejemplos**; con las cifras
  medidas, NB gana hasta n ≈ 100 y LR lo supera **entre ~100 y ~200**. La curva inventada confirma la
  tesis del enunciado en el lugar equivocado.
- **Qué comprobaría un verificador sin conocer las respuestas:** la procedencia. Cada cifra con
  decimales del entregable tiene que aparecer en algo que escribió una ejecución (un JSON, un CSV,
  la salida de un notebook). No hace falta saber cuál es la respuesta correcta para ver que no
  hubo ninguna ejecución. Es la comprobación `procedencia` de `evaluar_solver.py` y la regla 5.

## 0.b — El RAG que pierde la referencia ([0b_rag_plano.txt](0b_rag_plano.txt))
```
RAG plano (TF-IDF, k=2): Parte 4 (0.293), Parte 3 (0.243) → ✗ datos ✗ proporción ✗ prueba intocable
+ un salto: Parte 3 —depende_de→ Parte 1                  → ✓ datos ✓ proporción ✓ prueba intocable
RAG plano: 0 de 3 hechos. Con el salto: 3 de 3.
```
- **Fallo de la búsqueda, no del modelo:** ningún modelo intervino; el contexto que se le daría ya
  no contiene los datos. La Parte 1 no se parece a la pregunta («¿con qué datos…?»), así que la
  similitud nunca la trae.
- **Qué haría un LLM con el primer contexto:** inventar la partición (algún 80/20 o 70/30 sin
  semilla, quizá otro dataset) o, en el mejor caso, preguntar. Es el 0.a, pero causado por la búsqueda.
- **Por qué la conexión sale de una regla:** «Parte 1» es texto literal; un regex la encuentra
  siempre, igual, gratis y sin alucinar. Pedírsela al LLM añade costo, una fuente de error y algo
  que no es reproducible, para un dato que ya está escrito.

## 0.c — Terminar sin error no es acertar ([0c_exit_cero.txt](0c_exit_cero.txt))
```
stdout: Exactitud: 1.000
1) Evaluador ingenuo: APROBADO (returncode=0, resultados.json=sí)
2) estática: se predice sobre lo mismo que se ajustó: ['X'] · plausible: ['accuracy=1.0'] → RECHAZADO
```
- **La fuga que la revisión estática no ve:** `StandardScaler().fit(X)` con todos los datos antes de
  partir. El `.fit` es sobre `X`, y el `.predict` es sobre `X_test`, otra variable, así que la regla
  «misma variable en fit y predict» no salta. La información del test ya entró en la media y la
  desviación del escalador, y la exactitud resultante es plausible (no 1,000), así que la
  comprobación de plausibilidad tampoco salta.
- **Por qué se puede dejar la decisión final a un LLM revisor:** las comprobaciones con código van
  antes y no se pueden convencer; filtran lo que es mecánicamente falso (código de salida, JSON,
  NaN, fit/predict sobre lo mismo, métricas imposibles). Al LLM solo llega lo que ya pasó ese filtro,
  y su error queda acotado a juicios (¿responde esto a la subtarea?), no a aceptar un 1,000 imposible.
