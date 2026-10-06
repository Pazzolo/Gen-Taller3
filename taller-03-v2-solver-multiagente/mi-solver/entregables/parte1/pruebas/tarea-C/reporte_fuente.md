**MMIA 6013 · Tarea C — Recuperación léxica contra semántica latente sobre un corpus mínimo**

## Objetivo

Comparar un recuperador léxico (TF-IDF + similitud coseno) con uno de semántica latente (LSA: TruncatedSVD a 4 dimensiones sobre la misma matriz TF-IDF) en un corpus de diez documentos y seis consultas con un documento relevante cada una. Se miden Hit@1, Hit@3 y MRR, se identifica en qué consultas falla cada método mirando qué términos comparten la consulta y su documento relevante, y se discute qué aportaría un recuperador denso en el caso más difícil (q5).

## Método

Indexé los diez documentos con TfidfVectorizer de scikit-learn con sus parámetros por defecto (minúsculas, tokens alfanuméricos de dos o más caracteres, idf suavizada y normalización l2), y vectoricé las seis consultas con el mismo vocabulario. La puntuación de cada par consulta–documento es la similitud coseno entre sus vectores, y cada consulta ordena los diez documentos. Sobre ese ranking calculo Hit@k (proporción de consultas cuyo relevante queda entre los k primeros, con k = 1 y k = 3) y MRR (promedio del inverso del rank del relevante). Para LSA ajusté TruncatedSVD(n_components=4, random_state=0) sobre la matriz TF-IDF de documentos, proyecté documentos y consultas a ese espacio de 4 dimensiones y repetí la misma evaluación con los mismos juicios. La implementación (Python con scikit-learn) se adjunta aparte.

## Resultados

El vocabulario construido tiene 105 términos. La Tabla 1 resume los dos métodos (Parte 2).

**Tabla 1.** Métricas globales sobre las seis consultas.

| Método | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| TF-IDF (línea base léxica) | 0.8333 | 0.8333 | 0.8611 |
| LSA (TruncatedSVD, 4 componentes) | 0.6667 | 0.8333 | 0.7500 |

TF-IDF coloca al relevante en el puesto 1 en cinco de las seis consultas; LSA pierde Hit@1 (−0.1667) y MRR (−0.1111) y deja Hit@3 igual. Los 4 componentes explican en conjunto 0.3953 de la varianza.

**Tabla 2.** Análisis por consulta (Parte 3): rank del relevante por método y términos compartidos entre la consulta y su documento relevante.

| Consulta | Rank TF-IDF | Rank LSA | Términos compartidos consulta–relevante |
|---|---|---|---|
| q1 → d04 | 1 | 1 | función, ranking, léxica, pondera, frecuencia, términos |
| q2 → d03 | 1 | 1 | fragmentos, prompt; variantes recuperados/recuperación y añaden/añade |
| q3 → d08 | 1 | 1 | entrena, matrices, bajo, rango |
| q4 → d05 | 1 | 3 | vectores, orientación, compara |
| q5 → d09 | 6 | 6 | solo el artículo «la»; sin solape de contenido |
| q6 → d02 | 1 | 1 | autoatención, capas, feed-forward |

El único relevante fuera del top 3 en los dos métodos es el de q5. La consulta («…más determinista al elegir la siguiente palabra») y d09 («La temperatura reescala los logits…; valores bajos concentran la probabilidad») no comparten ninguna palabra de contenido, solo «la»: la similitud TF-IDF cae a 0.0790 y el rank a 6, mientras d03, que sí comparte «modelo» con la consulta, se queda con el primer puesto. LSA tampoco lo resuelve (rank 6, similitud 0.4999). Además, LSA degrada q4: d05 baja del puesto 1 al 3, superado por d01 (0.9220) y d10 (0.9206) frente a 0.8984; al proyectar a 4 dimensiones las similitudes se concentran y el orden lo deciden componentes latentes que no reflejan los términos concretos de la consulta. En el extremo opuesto, q1 comparte casi todos sus términos de contenido con d04 y logra la similitud más alta (0.6891).

## Discusión

LSA no mejora aquí porque se construye sobre las mismas co-ocurrencias del corpus: con diez documentos y 4 componentes que capturan 0.3953 de la varianza, no tiene de dónde aprender que «determinista» se relaciona con «temperatura» o «probabilidad». La proyección latente solo suaviza el ranking y, en q4, incluso borra la ventaja del solape léxico exacto de d05.

Un recuperador denso resolvería q5 mejor que los dos métodos de esta tarea porque aprende la relación entre consultas y pasajes a partir de ejemplos externos al corpus: puede saber que «elegir la siguiente palabra de forma más determinista» y «valores bajos [de temperatura] concentran la probabilidad» expresan la misma idea, y acercarlos en el espacio de embeddings aunque no compartan ninguna palabra de contenido. Eso es justo lo que ni TF-IDF (similitud 0.0790, rank 6) ni LSA sobre este corpus mínimo (rank 6) logran, porque ambos dependen, directa o indirectamente, del solape de términos dentro de los diez documentos.

El costo frente a TF-IDF es triple. Cómputo: hay que ejecutar un encoder neuronal sobre cada documento al indexar y sobre cada consulta al buscar, frente a contar términos y multiplicar matrices. Dependencia: requiere un modelo preentrenado y sus datos, mientras TF-IDF no entrena nada. Interpretabilidad: las dimensiones densas no se leen como términos, así que no se puede explicar un ranking mirando qué palabras comparten consulta y documento, como hicimos en la Tabla 2. A cambio, cubre los casos de vocabulario disjunto como q5, que son los que hacen caer el MRR de los métodos léxicos.
