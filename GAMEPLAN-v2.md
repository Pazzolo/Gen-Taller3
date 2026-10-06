# Taller 03 v2 — Gameplan: un solver multiagente con GraphRAG

Fuente: `taller-03-v2-solver-multiagente.pdf` y el kit `taller-03-v2-solver-multiagente/solver-v2/`.
Pesos: P0 10 % · P1 35 % · P2 30 % (2.a/2.b/2.c, 10 % cada una) · P3 10 % · P4 5 % · P5 5 % · reproducibilidad 5 %.

> ⚠️ **Es opcional y sustituye al Taller 03, no se suma:** «Peso: 25 % de la nota final, en lugar
> del Taller 03: cada grupo entrega uno de los dos». El Taller 03 ya está terminado (`GAMEPLAN.md`).
> Hay que confirmar con el grupo y el profesor que se entrega el v2 en su lugar.
> Entrega propuesta: **10 de octubre de 2026, 23:59** (la fija el profesor en clase).

> **Dos reglas del enunciado:** construir → medir → mejorar → volver a medir. Y **una cifra que
> no salió de una ejecución no es un resultado**: un reporte con cifras sin origen, o tablas que
> no salen del CSV, pierden esos puntos aunque se lean bien.

---

## Reglas de trabajo (las mismas del Taller 03)

### 1. Mirar primero el código de referencia
Carpeta `202610.IM2.4971 - …/`. Qué sirve en cada parte:

| Parte | Mirar primero |
|---|---|
| Cliente H200 (LLM + embeddings, `finish_reason=length`) | `solver-v2/h200.py` (el kit), `mini-opencode/h200.py` |
| Orquestador como grafo de estados | `h200-langgraph-profesor.ipynb` (`StateGraph`, aristas condicionales, `interrupt`, `InMemorySaver`), `s3-mie-profesor.ipynb` (motor de grafo de 40 líneas) |
| Indexador GraphRAG | `solver-v2/parte0/b_rag_plano.py` (el regex de aristas `depende_de`), `h200-langgraph` §3 (RAG con `bge-m3`), `papers/edge-2024-graphrag.pdf` |
| Sandbox: revisión antes de ejecutar | `mini-opencode/guardia.py` (lista blanca, *fail closed*) + `test_guardia.py` (el estilo de pruebas) |
| Sandbox: ejecución | `mini-opencode/herramientas.py` (`Consola`: `env` limpio, `shell=False`, timeout que mata, *tripwire*) |
| Revisor: comprobaciones con código | `solver-v2/parte0/c_exit_cero.py` (`evalua_sobre_entrenamiento`, `implausibles`) |
| Traza, presupuesto, repetición | `mini-opencode/traza.py`, `agentes.py` (`Presupuesto`), `s3-mar-profesor.ipynb` |
| Procedencia de cifras (regla 5) | `solver-v2/evaluar_solver.py` (`numeros`, `artefactos`, la comprobación `procedencia`) |
| Lo ya hecho en el Taller 03 | `taller-03-agentes/Lab-03-Agentes/` (traza en `finally`, frenos, modelo de guion, `tabla_2b.py`) |

### 2. Commits a medida que se avanza
- Un commit por paso con sentido (`v2-parte1: sandbox con revisión AST y timeout que mata`).
- Antes de **cada** commit: `git status` y buscar claves en lo que se sube. Aquí el riesgo es mayor
  que en el Taller 03: **el modelo escribe código**, y si el script heredara el entorno podría
  imprimir una clave en un log. Ese es el motivo de la regla 3 (sin variables de entorno).
- El `.gitignore` del kit excluye `corridas/` y los CSV, pero son entregables: hay que sacarlos de
  ahí, como en el Taller 03.

---

## Fase 0 — Preparación (≈30 min)
1. venv con Python ≥ 3.11 en `solver-v2/` (el kit se probó en 3.14; verificar que los pines
   instalen en 3.11: `numpy==2.5.3`, `scikit-learn==1.9.1`, `pymupdf==1.28.2`…).
2. Comprobar con la VPN los **dos** servicios de la H200: el LLM en `:12555` (`/v1/models`, réplica
   en `:12559`) y los embeddings `bge-m3` en el Ollama de `:11434` (el `:12555` no da embeddings).
3. ⚠️ **Falta la Tarea D:** el `golden_tareas.json` y el README citan `enunciados/hackathon3_student/`
   (paquete con notebook, contrato y verificador), pero **no viene en el kit**. Hay que pedirlo o
   sacar D del golden set y decirlo en el informe.
4. Carpeta del solver: `taller-03-v2-solver-multiagente/mi-solver/` (el README del kit lo
   invoca con `--ruta ../mi-solver`).

## Fase 1 — Parte 0: tres errores que no dan error (≈45 min, 10 %)
`python parte0/b_rag_plano.py` y `c_exit_cero.py` (sin red); `a_llm_sin_ejecutar.py` (H200).

| Ítem | Qué entregar |
|---|---|
| 0.a | Salida, modelo y tres frases: por qué no basta con que las cifras estén cerca; qué cifra inventada cambiaría una conclusión (la curva de aprendizaje que confirma la tesis); qué comprobaría un verificador sin conocer las respuestas (procedencia: cada cifra debe aparecer en algo que escribió una ejecución) |
| 0.b | Salida (referencia: RAG plano 0 de 3 hechos, con un salto 3 de 3) y tres frases: falla de búsqueda y no del modelo; qué haría un LLM con el primer contexto (inventar la partición); por qué la arista sale de una regla (determinista, gratis, sin alucinación) |
| 0.c | Salida (ingenuo APROBADO, con dos comprobaciones RECHAZADO) y dos frases: la fuga que la revisión de código no ve (escalador ajustado con todos los datos antes de partir); por qué se puede dejar la decisión final a un LLM si el código revisa antes |

## Fase 2 — Parte 1: el solver base (≈8–12 h, 35 %)
`Solver().solve(pdf, salida) -> {status, entregables, subtareas, usage, model, trace}` y además
`.run(pregunta)` para el Taller 4.

**Roles** (un archivo por rol en `mi-solver/agentes/`):

| Rol | Qué hace | Qué revisa el código |
|---|---|---|
| Lector | PDF → texto por página con tablas (`pymupdf`, `find_tables`) | texto pegado o partido, PDF escaneado (sin texto → aviso) |
| Indexador (GraphRAG) | nodos = secciones; aristas `depende_de` por **regex** («Parte 1», «la pregunta anterior»); encima, entidades del LLM (datasets, métodos, métricas) unidas con las de `notas-teoricas/`; embeddings `bge-m3` para la entrada; caché en disco | secciones y dependencias |
| Planificador | subtareas con `id`, `tipo` (cálculo / redacción / figura), `seccion` y `depende_de` (JSON) | **regla 1** |
| Investigador | por subtarea: su sección literal + las de sus dependencias, cada una con su cita | que vayan la sección y sus dependencias |
| Programador | un script por subtarea de cálculo, que guarda sus cifras en `resultados.json` | el filtro del sandbox |
| Ejecutor | código, sin LLM: corre el script | **regla 3** |
| Revisor | primero código (0.c: código de salida, JSON, NaN, figuras, fugas, plausibilidad), después LLM | **regla 4** |
| Redactor | entregable en el formato pedido: `.md`, `.ipynb` ejecutado (`nbconvert`) o `.pdf` dentro de su límite de páginas | secciones, extensión, **regla 5** |

**Orquestador** en LangGraph (`langgraph==1.2.12`): leer → indexar → planificar ⇄ validar →
por subtarea [investigar → programar → ejecutar → revisar ⇄ (volver a programar | aprobar | fallida)]
→ redactar ⇄ comprobar procedencia → publicar. Hasta aprobar o llegar al máximo de intentos; se pasa a
redactar cuando se acaban las subtareas o el presupuesto.

**Las seis reglas** (marcadas `# REGLA n`):
1. El plan se valida con código: ids únicos, dependencias que existen, sin ciclos (`networkx`), cada sección cubierta. Si falla, vuelve al planificador con la lista de problemas.
2. GraphRAG con base fija: las aristas salen de una regla; las entidades del LLM van encima; cada fragmento lleva su cita.
3. Sandbox: revisión AST antes de ejecutar (sin red, sin `subprocess`/`os.system`, sin borrar, sin `eval`/`exec`, sin rutas fuera de la carpeta); `env` vacío, carpeta propia y timeout que mata **el grupo de procesos** (`start_new_session` + `killpg`).
4. Resultados en un JSON de nombre fijo; el revisor corre las comprobaciones de la 0.c y solo si pasan llama al LLM.
5. Cada cifra del entregable salió de una ejecución o está en el enunciado (reutilizar la lógica de `procedencia` del evaluador); las que no, vuelven al redactor.
6. Traza JSONL siempre, también si falla: por llamada (agente, modelo, tokens, latencia, error) y por ejecución (código de salida, duración, archivos creados).

Además: detectar `finish_reason=length` con contenido vacío y reintentar con más `max_tokens`
(el modelo razona siempre y consume el mismo cupo).

**Entregas:** el código; el diagrama del orquestador; **el grafo de la Tarea A dibujado** con las
aristas `depende_de` visibles; el plan de una tarea; el modelo y la fecha; **la traza de una corrida
en que el revisor rechazó al menos un script**.

Orden sugerido: lector + indexador (probar contra 0.b) → sandbox + revisor con código (probar contra
0.c) → planificador + validación → programador/ejecutor/revisor con modelo de guion → redactor por
formato → LangGraph de punta a punta en la Tarea A → B → C.

## Fase 3 — Parte 2: evaluación (≈4 h + tiempo de corrida, 30 %)
- **2.a Golden set:** `golden_tareas.json` (A–C, y D si aparece) **más dos tareas reales de la
  maestría en PDF** (con permiso si no son públicas), cada una con ≥ 5 comprobaciones: de forma,
  ≥ 2 de tipo `cifra` con `verdad_py` y 1 de `procedencia`. Entre todas: un notebook, un PDF y una
  **trampa** (no tocar la prueba, una semilla, un dataset que habría que descargar).
  ⏳ **Depende del grupo:** elegir y traer esas dos tareas.
- **2.b Completo contra recortado:** el mismo golden set y el mismo modelo, con el solver completo y
  con una versión **sin grafo** (top-k como la 0.b) o **sin revisor** (un intento, sin revisión).
  Por tarea: comprobaciones aprobadas, procedencia, subtareas fallidas, intentos, tokens por agente y
  duración, desde `resultados_solver.csv` y `resumen_solver.csv`.
- **2.c Tres peores comprobaciones:** qué agente falló (lector, grafo, plan, investigador,
  programador, sandbox, revisor o redactor), con la traza.
- **Tiempo:** según el enunciado, cada tarea toma entre 4 y 13 min y entre 79 000 y 138 000 tokens;
  la primera indexación de las notas suma ~6 min. Dos versiones × 5–6 tareas ≈ 1,5–2,5 h de corrida.

## Fase 4 — Parte 3: cuatro frenos (≈2 h, 10 %)
Cada uno activado con su traza (con modelo de guion para no gastar):
1. **Presupuesto de tokens** revisado antes de cada llamada, **con reserva para el redactor**: si se
   acaba, entrega lo que tiene y dice qué faltó.
2. **Máximo de intentos por subtarea + detector de repetición:** al máximo, la subtarea queda fallida
   y se sigue; un script idéntico (hash) a uno rechazado no se vuelve a ejecutar.
3. **Tiempo máximo del sandbox** que mata todos los procesos del script (probar con un script que
   lanza un hijo que duerme).
4. **Confirmación humana antes de usar la red:** `fetch_openml`, `load_dataset`, `download=True` no
   corren sin aprobación (`interrupt` de LangGraph o una pregunta en consola).

## Fase 5 — Parte 4: una mejora (≈3 h, 5 %)
Se elige después de medir. Candidatas: **A** paralelo con las dos réplicas (mide tiempo, que es el
costo real aquí), **C** memoria entre tareas, **E** juez `gemma3` con acuerdo y kappa contra las
comprobaciones. Probar con una tarea antes de correr todas.

## Fase 6 — Parte 5 e informe (≈2 h)
Tres respuestas de 100–150 palabras (quién decide que la tarea está resuelta; qué agente gasta más
tokens y qué ganó el grafo, con `resumen_solver.csv`; uso honesto y daño si falla el sandbox), y el
PDF: Parte 0, arquitectura con las seis reglas, grafo de la Tarea A, modelo y fecha, golden set
comentado, tabla 2.b, tres peores casos, cuatro frenos, mejora medida y Parte 5.
**Repositorio:** código, `requirements.txt`, `golden_tareas.json`, `corridas/tarea-*/` con
`traza.jsonl`, `plan.json` y el entregable, y los CSV de las dos versiones.

---

## Decisiones (2026-10-05)
1. **El v2 se hace para estudiar**, no para entregar en lugar del Taller 03, así que la fecha no presiona.
2. **Tareas propias provisionales:** mientras llegan las dos tareas reales de la maestría, se escriben
   dos tareas propias (E y F) con el mismo formato del kit, marcadas `"provisional": true` en el golden
   set. Cuando lleguen las reales, se sustituyen.
3. **La Tarea D queda fuera** hasta que aparezca `hackathon3_student/`.

## Estado
- **Fase 0 ✅** Python 3.14.7 con `uv`, porque `numpy==2.5.3` exige ≥ 3.12 y el 3.11 de Homebrew no
  sirve: `uv venv --python 3.14 solver-v2/.venv`. Además de los pines del kit se instalaron
  `ipykernel` y `matplotlib` (para notebooks y figuras). H200 verificada: LLM `zai-org/GLM-5.3-Flash`
  en 12555 y 12559, `bge-m3` (1024 dim.) en 11434, `gemma3:27b` disponible (opción E de la Parte 4).
- **Golden set ✅ (adelantado de la Fase 3)** `mi-solver/golden_tareas.json`: A–C del kit + E (notebook, trampa: alpha con la prueba) y F (PDF de 1 página, trampa: no estandarizar), provisionales. Validado con soluciones hechas a mano: correctas 8/8 y 6/6, trampas 4/8 y 4/6. Notas en `mi-solver/entregables/parte2/NOTAS_golden.md`.
- **Fase 2 🔨 (en curso)** Piezas hechas y probadas sin red (`test_solver.py`, 19/19):
  `lector.py` (secciones por tamaño de letra, tablas), `llm.py` (H200, reintento por `length`,
  presupuesto con reserva), `grafo.py` (regla 2: `depende_de` por regex solo hacia atrás, entidades del
  LLM, enlace con las notas por entidades específicas), `sandbox.py` (regla 3: AST + entorno mínimo +
  `killpg` + `sandbox-exec` sin red), `revisor.py` (regla 4, incluida la fuga del escalador),
  `agentes.py` (regla 1), `redactor.py` (regla 5, md/pdf/ipynb) y `solver.py` (LangGraph).
  Grafo de la Tarea A dibujado en `mi-solver/entregables/parte1/grafo_tarea_a.png`.
  Hallazgo: con `bge-m3`, el top-k plano también recupera la Parte 1 en la Tarea A (la 0.b falla con
  TF-IDF); la 2.b dirá si el grafo gana en otras tareas.
- **Parte 3 ✅** `frenos_demo.py`: los cuatro frenos activados en corridas completas (traza en `mi-solver/entregables/parte3/`).
- **2.b ⏳ interrumpida (2026-10-06 02:17 UTC):** la VPN se cayó durante la redacción de C (completo; A y B ya habían
  terminado). El evaluador quedó 75 min en `recv`, porque el `timeout` de urlopen es por lectura y no total.
  Arreglado con `PLAZO_CHAT_S` en `llm.py`. Hay que **reconectar GlobalProtect y repetir la evaluación completa**:
  el evaluador escribe el CSV solo al final, así que A y B de esa pasada no quedaron registradas.
- **Fase 1 ✅** Parte 0 reproducida. 0.a: 26 cifras, 0 con origen; el cruce inventado (20–40
  ejemplos) contra el medido (~100–200). 0.b: 0/3 → 3/3. 0.c: APROBADO → RECHAZADO. Evidencia y notas
  en `mi-solver/entregables/parte0/`.
