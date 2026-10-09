# Tarea C — Recuperación léxica contra semántica latente sobre un corpus mínimo

## Objetivo

Comparar dos formas de recuperar documentos sobre un corpus mínimo de 10 textos y 6 consultas, cada una con un documento relevante: la línea base léxica (TF-IDF con similitud coseno) y la semántica latente (LSA, proyectando la misma matriz TF-IDF a 4 dimensiones). Se mide si el relevante queda en el puesto 1 (Hit@1), dentro de los 3 primeros (Hit@3) y el rango recíproco medio (MRR), y se explica por qué cada método falla donde falla.

## Método

- **Datos:** documentos d01–d10 y consultas q1–q6 del enunciado, con relevantes q1→d04, q2→d03, q3→d08, q4→d05, q5→d09 y q6→d02.
- **Línea base:** `TfidfVectorizer` de scikit-learn con parámetros por defecto (minúsculas, tokens de 2+ caracteres, sin lista de stop words en español, norma l2). El vocabulario resultante tiene 83 términos. Para cada consulta se ordenan los 10 documentos por similitud coseno.
- **LSA:** `TruncatedSVD(n_components=4, random_state=0)` sobre la misma matriz TF-IDF; se proyectan documentos y consultas y se repite el ranking con coseno. Las 4 componentes explican en conjunto 0.4026 (40.26 %) de la varianza.
- **Métricas** sobre las 6 consultas: Hit@1, Hit@3 y MRR (media del recíproco del rango del primer relevante).
- Todas las cifras salen de la ejecución registrada (resultados.json y su salida estándar). El código se entrega aparte, no se pega aquí.

## Resultados

**Tabla 1 — Métricas agregadas (Parte 2, 6 consultas):**

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF + coseno | 0.8333 | 0.8333 | 0.8519 |
| LSA-4D + coseno | 0.6667 | 0.8333 | 0.7667 |

**Tabla 2 — Posición (1-based) del primer relevante por consulta (Parte 3):**

| Consulta | Relevante | Pos. TF-IDF | Pos. LSA | ¿TF-IDF en top-3? | ¿LSA en top-3? |
|---|---|---|---|---|---|
| q1 | d04 | 1 | 1 | Sí | Sí |
| q2 | d03 | 1 | 1 | Sí | Sí |
| q3 | d08 | 1 | 2 | Sí | Sí |
| q4 | d05 | 1 | 1 | Sí | Sí |
| q5 | d09 | 9 | 10 | No | No |
| q6 | d02 | 1 | 1 | Sí | Sí |

Lectura: TF-IDF coloca el relevante en el puesto 1 en cinco de seis consultas; cinco aportan 1.0 al recíproco y q5 aporta 0.1111 (puesto 9), de ahí su MRR de 0.8519. LSA conserva el Hit@3 pero pierde un Hit@1: en q3 el relevante baja al puesto 2 porque otro documento del espacio latente lo supera (0.9697 frente a 0.9336), y en q5 cae al puesto 10; q3 aporta 0.5 y q5 aporta 0.1 al recíproco, de ahí su MRR de 0.7667.

**Nota de alcance (qué no se pudo completar y por qué).** El presupuesto de ejecución del programador se agotó (550,542 gastados más ~2,114 estimados, por encima de la reserva de 540,000). Quedaron sin ejecutar: (a) la subtarea T1, es decir, la Parte 1 como corrida y verificación independiente contra los juicios del enunciado (las métricas de TF-IDF reportadas provienen de la corrida de T2); y (b) la parte cuantitativa de la subtarea T3: el cálculo de los términos que comparten cada consulta y su relevante. De T3 solo pude derivar, a partir de los rangos ya calculados, qué consultas quedan fuera del top-3. Las posiciones de la Tabla 2 se copian de la salida de T2 sin verificación consulta a consulta, y tampoco pude confirmar contra el código (que va aparte) que el orden interno de documentos de la ejecución coincida con d01–d10. Por eso el análisis de términos que sigue es cualitativo y no incluye cifras de solapamiento.

## Discusión

**Por qué falla q5 en los dos métodos.** La ejecución da una similitud TF-IDF de 0.0 entre q5 y su relevante: no comparten ningún término del vocabulario. Mirando los textos, el diagnóstico es de paráfrasis: la pregunta dice "más determinista al elegir la siguiente palabra" y d09 responde con otro vocabulario ("La temperatura reescala los logits antes del softmax; valores bajos concentran la probabilidad"). El único cruce posible son funcionarios como "la", a los que el IDF casi anula. Sin coincidencia léxica, TF-IDF no tiene señal; y LSA, entrenada sobre la co-ocurrencia de este mismo corpus, tampoco: deja al relevante con similitud −0.1972 y en el puesto 10.

**Por qué LSA no mejora (y empeora q3).** Con 10 documentos y 83 términos, las 4 componentes latentes retienen solo 40.26 % de la varianza: la proyección descarta detalle léxico que, en un corpus tan pequeño, es justamente la señal. En q3 el relevante sigue siendo cercano (0.9336), pero la proyección acerca la consulta a otro documento (0.9697) y le quita el puesto 1; en q5 el espacio latente ni siquiera conserva el cero de TF-IDF y produce una similitud negativa. LSA aporta sinónimos cuando la co-ocurrencia se observa en muchos documentos; aquí no hay datos suficientes para eso.

**Parte 4 — por qué un recuperador denso resolvería q5.** Un codificador entrenado (por ejemplo, un bi-encoder tipo Sentence-BERT) aprende, a partir de millones de pares de textos, a acercar en su espacio documentos que se parafrasean: "hacer el modelo más determinista al elegir la siguiente palabra" y "valores bajos de temperatura concentran la probabilidad" quedarían cerca aunque no compartieran ninguna palabra. TF-IDF exige coincidencia léxica exacta, y LSA solo puede aprender la co-ocurrencia del corpus propio (10 documentos, 83 términos, 40.26 % de varianza en 4 dimensiones), muy lejos de contener esa equivalencia.

**Costo frente a TF-IDF.** (1) Cómputo: cada documento y cada consulta deben pasar por una red neuronal (inferencia), frente a contar términos; para escalar se necesita además búsqueda aproximada de vecinos. (2) Memoria: vectores densos de cientos o miles de dimensiones por documento, frente a un vector disperso de 83 términos. (3) Dependencias: requiere un modelo preentrenado (descarga, licencia, versiones) y su calidad depende del dominio en que fue entrenado. (4) Opacidad: es más difícil explicar por qué dos textos son similares, mientras que en TF-IDF se pueden listar los términos compartidos.
