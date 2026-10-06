# Parte 1 — El solver base (v2): arquitectura y notas para el informe

Modelo: `zai-org/GLM-5.3-Flash` en la H200 (vLLM, `:12555`; réplica en `:12559`), leído de `/v1/models`;
embeddings `bge-m3` en el Ollama de `:11434`. Fecha de las corridas: 2026-10-05. Python 3.14.7.

## Los roles

| Rol | Archivo | Qué hace | Qué revisa el código |
|---|---|---|---|
| Lector | `lector.py` | PDF → texto por página; secciones por tamaño de letra; tablas con `find_tables` en Markdown | ligaduras (NFKC), guiones de corte, palabras pegadas, páginas sin texto (PDF escaneado) |
| Indexador (GraphRAG) | `grafo.py` | nodos = secciones; `depende_de` por regex; entidades del LLM sobre el enunciado y las 83 notas del curso (caché) | REGLA 2: las aristas entre secciones nunca las decide el LLM |
| Planificador | `agentes.py` | subtareas con tipo, secciones, dependencias y figura; la especificación del entregable | REGLA 1: `validar_plan` |
| Investigador | `agentes.investigar` | preámbulo + sección literal + dependencias (siguiendo `depende_de`) + sección de entrega + notas del curso con entidades en común; cada fragmento con su cita | que vayan la sección y sus dependencias |
| Programador | `agentes.programar` | un script por subtarea de cálculo, que escribe `resultados.json` | el filtro del sandbox (REGLA 3) |
| Ejecutor | `sandbox.ejecutar` | corre el script, sin LLM | REGLA 3 |
| Revisor | `revisor.py` | comprobaciones con código y, si pasan, el LLM | REGLA 4 |
| Redactor | `redactor.py` | `.md`, `.pdf` (PyMuPDF, ajusta letra antes de pedir recortar) o `.ipynb` ejecutado dentro del sandbox | REGLA 5 y forma |

El orquestador (`solver.py`) es un `StateGraph` de LangGraph 1.2.12 (diagrama en el docstring de
`solver.py`): leer → indexar → planificar ⇄ validar → por subtarea [investigar → programar →
revisar_codigo → (aprobar_red) → ejecutar → revisar ⇄ programar, hasta aprobar o fallar] → redactar ⇄
comprobar → publicar.

## Las seis reglas en el código

| Regla | Dónde | Cómo |
|---|---|---|
| 1. El plan se valida con código | `agentes.validar_plan` (`agentes.py:77`), `solver.py:183` | ids únicos, dependencias que existen, sin ciclos (`networkx.find_cycle`), toda sección cubierta salvo la de entrega, tipos válidos; formato, archivo y límites contra lo que el enunciado dice explícitamente (`especificacion_por_reglas`). Hasta 3 vueltas al planificador con la lista de problemas |
| 2. GraphRAG con base fija | `grafo.py:38` (regex), `grafo.py:175`, `agentes.investigar` (`agentes.py:147`) | «Parte N», «Sección N», «la pregunta anterior», «las mismas …» → `depende_de` solo hacia atrás; las menciones hacia adelante quedan como `referencia_adelante` y no se siguen. Encima, entidades del LLM unidas con las de las notas por contención de palabras, peso IDF y un piso de coseno |
| 3. Sandbox | `sandbox.py` (`revisar`, `ejecutar`), `solver.py:252` | AST (red, procesos, borrado, `eval`, dunders, rutas fuera) → entorno mínimo sin variables del usuario → carpeta y `TMPDIR` propios → `start_new_session` + `killpg` al vencer el tiempo → `sandbox-exec` que niega red y escritura fuera de la carpeta |
| 4. Resultados fijos y revisión con código primero | `sandbox.py:35` (`resultados.json`), `revisor.revisar_con_codigo`, `solver.py:302` | salida, JSON, NaN, figura, `fuga_fit`, `fuga_escala`, plausibilidad; el LLM solo si todo eso pasa |
| 5. Origen de cada cifra | `redactor.sin_origen` / `comprobar`, `solver.py:381` | la lógica de `procedencia` de `evaluar_solver.py`: toda cifra con ≥ 2 decimales tiene que estar en un artefacto de una ejecución o en el enunciado; si no, vuelve al redactor |
| 6. Siempre se guarda la traza | `traza.py`, `llm.py`, `solver.py:105` | JSONL escrito línea a línea desde el primer evento; por llamada (agente, modelo, tokens, latencia, error), por ejecución (código de salida, duración, archivos creados) |

Pruebas sin red: `test_solver.py`, 20/20.

## Entregas de la Parte 1
- **Grafo de la Tarea A dibujado:** [grafo_tarea_a.png](grafo_tarea_a.png). Secciones en azul, `depende_de` en rojo
  (Parte 3 → Parte 1, El reporte → Partes 2 y 3), entidades en naranja y en verde las que también están en las notas.
- **Plan de una tarea:** [pruebas/tarea-A/plan.json](pruebas/tarea-A/plan.json), válido al primer intento.
- **Traza con un rechazo del revisor:** [pruebas/tarea-E/traza.jsonl](pruebas/tarea-E/traza.jsonl). El revisor rechazó
  dos scripts: T2 con código (`fuga_fit`, que resultó un falso positivo y se corrigió) y T4 con el LLM
  (la figura no quedaba visible en el notebook: `matplotlib.use("Agg")` + `plt.close()`). También
  [pruebas/tarea-A/traza.jsonl](pruebas/tarea-A/traza.jsonl), T3, rechazado por plausibilidad: otro falso
  positivo, que se corrigió.

## Primeras corridas reales (antes de la 2.b), comprobaciones del golden set
| Tarea | Formato | Resultado | Duración | Notas |
|---|---|---|---|---|
| A | md | 8/8 | 7 min | cruce medido entre 99 y 199 ejemplos (la 0.a inventó 20–40); la regla 5 atrapó 37,26/62,74 |
| B | ipynb | 9/9 | 7 min | notebook ejecutado dentro del sandbox, 42/42 cifras respaldadas |
| C | pdf | 8/8 | 18 min | el programador agotó 16k y 32k tokens razonando (respuesta vacía) antes de responder con 49k |
| E | ipynb | 7/8 | 7 min | no cayó en la trampa (α = 0,3162); falló E03 por un encabezado largo (corregido) |
| F | pdf | 6/6 | 5 min | no cayó en la trampa (ARI 0,8975) |

## Hallazgos honestos
- **El grafo no siempre gana en recuperación:** con `bge-m3`, el top-k plano también trae la Parte 1 en la
  Tarea A (la 0.b falla con TF-IDF). Lo que el grafo garantiza por construcción es traer el preámbulo
  (en la C, el corpus) y la sección de entrega.
- **Un defecto que el golden set no ve:** en la Tarea A el planificador escribió «estandarizar» también
  para Naive Bayes; la tabla dice NB 0,9357 (estandarizado) y la curva 0,9474 (crudo). A06 pasó porque
  0,9474 aparece igual cerca de «Naive». Candidato para la 2.c.
- **Falsos positivos del revisor con código** (plausibilidad por subcadena «acc» en «fracciones»;
  `fuga_fit` sobre `X_train`): cuestan un intento, no una respuesta equivocada, y se corrigieron.
