# Taller 04 — Gameplan: evaluar, observar y proteger el agente del Taller 3

Fuente: `Taller4/taller-04-llmops.pdf` (generado el 2026-10-09) y el kit `Taller4/Referencias/taller-04-llmops/` (idéntico a la copia de Descargas).
Pesos: P0 10 % · P1 25 % · 2.a 10 % · 2.b 15 % · 2.c 10 % · P3 15 % · P4 10 % · reproducibilidad 5 %.

> **Entrega: sábado 17 de octubre de 2026.** Repositorio con `resultados.csv`, spans y
> `guardrails.csv` **al cierre de la sesión (11:50)**; informe PDF hasta las 23:59.
> Hoy es jueves 15: quedan ~1,5 días. Todo lo que pide red (H200) necesita GlobalProtect.

> **Las dos reglas que quitan puntos aunque el código funcione:** un span que no se escribe
> cuando la corrida falla, un juez sin calibrar, o tablas que no salen del CSV. Y **una clave
> expuesta deja el taller sin calificar**; ningún dato `pii` puede aparecer en los spans.

---

## Decisión: ruta A (agente analista SQL)

Es la que se entregó como Taller 3 (`taller-03-agentes/entrega/`) y la que el kit soporta de
punta a punta: `evaluar_llmops.py` importa `../Lab-03-Agentes/evaluar.py` y usa
`sql_verificacion` como verdad. La B (solver v2) pediría `juez_solver.py`, golden de tareas con
instrucción escondida, sandbox como guardrail y corridas de minutos por tarea; no cabe en 1,5 días.
RAGAS: **no aplica** en la A (el agente no recupera texto) — decirlo así en el informe.

## Mirar primero: los archivos del taller por parte

Rutas relativas a `Taller4/Referencias/taller-04-llmops/` («Archivos de este taller» del enunciado).

| Parte | Mirar primero |
|---|---|
| Setup | `Lab-04-LLMOps/README.md`, `.env.example`, `requirements.txt` (`-r ../Lab-03-Agentes/requirements.txt` + `langfuse==4.17.0`) |
| P0 | `Lab-04-LLMOps/guion.py` (`ceros`, `guardrails`, `traza`), y lo que reproduce: `evaluation.py`, `guardrails.py`, `instrumented_agent.py` (no se modifican) |
| P1 — servidor | `langfuse/README.md`, `langfuse/levantar_langfuse.sh`, `langfuse/hola_langfuse.py` (20 líneas: `@observe`, `as_type="tool"`, `langfuse.openai`, `flush`) |
| P1 — instrumentar | `langfuse/demo.py`: `enmascarar()` (l. 76, la máscara que recorre tuplas → copiarla y extenderla a cédula/teléfono), `lf()` (l. 98, `Langfuse(mask=…)` + `auth_check`), `propagate_attributes(…, metadata={"prompt_version": …})` (l. 260), la herramienta que falla en nivel ERROR (`falla()`, l. 282–303) |
| P1 — exportar | `Lab-04-LLMOps/exportar_trazas.py` (`TIPOS`: GENERATION→llm, TOOL→herramienta, GUARDRAIL→guardrail; la raíz es `corrida`; `prompt_version` sale del `metadata`), `resumir_trazas.py` |
| 2.a | `Lab-04-LLMOps/casos_plantilla.json` (P13, P14 ambiguas; P18, P19 con `pii`), nuestro `taller-03-agentes/Lab-03-Agentes/golden_set.json` |
| 2.b | `Lab-04-LLMOps/evaluar_llmops.py` (docstring: contrato, claves opcionales `traza_id`/`prompt_version`/`guardrail`; `CRITERIO_POR_TIPO`; `calibrar`), `papers/zheng-2023-llm-as-judge.pdf` §3.3 (autopreferencia) |
| 2.c | `langfuse/demo.py` `registrar_prompts()` / `trazar()` (l. 233–267: `create_prompt(…, commit_message=…)`, `get_prompt`, `langfuse_prompt=prompt`) |
| P3 | `Lab-04-LLMOps/probar_guardrails.py` (contrato `f(texto) -> (ok, texto)`), `ataques_plantilla.json` (17 casos), `guardrails.py` (el contraejemplo); `202610.*/mini-opencode/guardia.py` (lista blanca, *fail closed*) |
| P4 | `papers/hu-2021-lora.pdf` (Tabla 5, §7.1), `papers/dettmers-2023-qlora.pdf` (§4), anexo de precios del PDF del enunciado (p. 12) |
| No aplica a la A | `ragas-langfuse/`, `Lab-04-LLMOps/juez_solver.py`, `Lab-03-Agentes/solver-v2/`, `taller-03-opcion-libre.md` |

**Hallazgos al revisar el kit:**
- **Falta la tabla de precios.** `evaluar_llmops.py` (`_tabla_modelos`) y `resumir_trazas.py --precio`
  buscan `fuentes/modelos/modelos-2026-1.json` subiendo carpetas, y no viene en el kit: con
  `--precio propietario_economico` sale «La tabla semestral no tiene la fila». Hay que crearla con
  las filas del anexo del PDF (`{"models": [{"id", "model", "input_usd_per_1m", "output_usd_per_1m",
  "verified_at"}]}`) y decirlo en el informe.
- **Compatibilidad confirmada:** `langfuse/README.md` dice que la demo se probó el 2026-10-06 con
  `langfuse` 4.17.0, servidor 4.50.0 y `openai` 3.22.1, la misma versión del Lab 03.
- **Exportar después del `flush()`, no en la misma línea:** el servidor tarda unos segundos en
  dejar consultable lo recién enviado (docstring de `exportar_trazas.py`).

## Estructura en el repo

```
taller-04-llmops/
├── Lab-03-Agentes -> ../taller-03-agentes/Lab-03-Agentes   # symlink: el evaluador busca ../Lab-03-Agentes
├── Lab-04-LLMOps/          # copia del kit + lo nuestro
│   ├── (kit tal cual: evaluation.py, guardrails.py, instrumented_agent.py, guion.py, evaluar_llmops.py, …)
│   ├── mi_agente_obs.py    # MiAgente del T3 envuelto con Langfuse + guardrails (NO se toca el T3)
│   ├── mis_guardas.py      # validar_entrada / validar_salida / mascara_pii
│   ├── golden_set.json     # 22 del T3 + 2 ambiguas + 2 pii = 26
│   ├── ataques.json, etiquetas.csv
│   ├── resultados_base.csv, resultados_v2.csv, guardrails_ejemplo.csv, guardrails_mios.csv
│   ├── spans_base.jsonl, spans_v2.jsonl
│   └── entregables/        # salidas de cada parte + informe
└── langfuse/               # del kit (levantar_langfuse.sh, hola_langfuse.py)
```

El T3 ya entregado queda congelado; el agente instrumentado importa `MiAgente` desde el T3.

---

## Fase 0 — Preparación (≈45 min)
1. Copiar el kit al repo (sin `papers/`, que es material del curso). Venv Python 3.11 en
   `Lab-04-LLMOps/` con `pip install -r requirements.txt` (incluye `langfuse==4.17.0`).
   (`langfuse.openai` + `openai==3.22.1` ya probado por el kit.) Crear
   `fuentes/modelos/modelos-2026-1.json` con el anexo de precios del PDF.
2. `python ../Lab-03-Agentes/crear_base.py` (la base ya existe en el T3, pero `data/` no se versiona).
3. **Langfuse local (hecho el 2026-10-16):** `./levantar_langfuse.sh` con Docker Desktop 4.12 /
   Compose 2.10.2. Ese Compose no entiende `!reset` de `sin-puertos-internos.yml`, y el Postgres
   local ocupa el 5432: se quitaron a mano los `ports:` de worker, clickhouse, redis y postgres en
   `.langfuse/docker-compose.yml` (generado, fuera de git) y se volvió a correr. Interfaz en
   http://localhost:3000 (usuario y contraseña en `langfuse/.langfuse/.env`). Claves copiadas a
   `Lab-04-LLMOps/.env`; `auth_check()` → True. Falta `hola_langfuse.py` (pide la H200).
4. `.env`: agente en vLLM `:12555`, juez **gemma3** en Ollama `:11434` (otra familia; ya lo
   usamos en el v2). Leer los ids de `/v1/models`, anotar ids y fecha para el informe.

## Fase 1 — Parte 0: tres errores que no dan error (≈30 min, 10 %) — sin red
`python guion.py ceros | guardrails | traza`, sin modificar los archivos del kit.

| Ítem | Entregar |
|---|---|
| 0.a | salida + por qué un reporte en ceros es peor que un error (parece una medición) + por qué no se promedia 0 «no aplica» con 0 «falló» |
| 0.b | tabla FN/FP + qué comparten los ataques que pasan (A02 vs A03: frase literal, tildes) + por qué los FP de L01/L03 son graves sin bloquear (reescriben en silencio) + qué le pasa a L03 en un agente de ventas por fecha |
| 0.c | salida + la línea de `instrumented_agent.py` de cada problema (texto original al bloquear, cronómetro después del guardrail, sin `finally`) |

## Fase 2 — Parte 1: observabilidad con Langfuse (≈3 h, 25 %)
`mi_agente_obs.py`, clase `MiAgenteObs` con el contrato del T3 + `traza_id`, `prompt_version`, `guardrail`:
- **corrida**: `@observe` en `.run()`; la raíz **envuelve los guardrails**; status y `prompt_version` en metadata.
- **guardrail**: `as_type="guardrail"`, output = `ok | redactada | bloqueada`.
- **llm**: cliente `from langfuse.openai import OpenAI` (inyectarlo en `MiAgente` en vez del cliente actual).
- **herramienta**: `as_type="tool"` alrededor de `mis_tools.ejecutar`: nombre, tamaño del resultado, error.
- Las tres correcciones de 0.c: span aunque falle (Langfuse lo cierra; además el `finally` del T3),
  cronómetro que incluye guardrails, `Langfuse(mask=mascara_pii)` que recorra dict/list/**tuplas**.
  Además: el `traza.json` en disco del T3 guarda `question` sin redactar → pasarle la pregunta ya redactada.
- `propagate_attributes(session_id="paolo-base", tags=["taller-04"])`, `flush()` al terminar.
- `exportar_trazas.py --sesion paolo-base -o spans_base.jsonl` → `resumir_trazas.py --precio propietario_economico`.
- Entregable: tabla corridas / tokens / costo con fila y fecha (gpt-4o-mini, 0,15/0,60, 2026-08-26) / p50 y p95 / tiempo por tipo de span; cuello de botella = mayor **tiempo total** (casi seguro `llm`).

## Fase 3 — Parte 2.a: golden set (≈45 min, 10 %)
Partir de `taller-03-agentes/Lab-03-Agentes/golden_set.json` (22: 15 respondibles, 4 negativas, 3 adversariales).
Añadir de `casos_plantilla.json` (misma base: `crear_base.py` idéntico, misma fórmula de importe)
**P13 y P14** (ambiguas, con `criterio_juez`) y **P18 y P19** (con `pii`: correo + cédula; teléfono
con fechas-trampa), con ids nuestros → 26 casos. Declarar en el informe que esos cuatro vienen de
la plantilla del curso. Revisar `debe_contener` donde la respuesta lleva un nombre. Comentar el
golden set en el informe.

## Fase 4 — Parte 2.b: verificadores y juez calibrado (≈2 h, 15 %)
1. `python evaluar_llmops.py --golden golden_set.json --agente mi_agente_obs:MiAgenteObs --juez gemma3` → `resultados_base.csv`.
2. Métricas por tipo de los dos verificadores (tabla generada desde el CSV con un script, como `tabla_2b.py`).
3. Calibración 1: acuerdo y kappa juez vs determinístico (respondibles, negativas, adversariales).
4. Calibración 2: **escribir `etiquetas.csv` (≥10, todas las ambiguas) ANTES de abrir la columna del juez**;
   `--calibrar resultados.csv --etiquetas etiquetas.csv`.
5. Revisar cada desacuerdo; si el error es del verificador (porcentaje como fracción, nombre), corregir el golden y decirlo.
- ⚠️ Lección del v2: gemma3 aprobó 58/58 → kappa 0/indefinido. Si el agente acierta casi todo, kappa
  no informa: decirlo, y las ambiguas/pii son las que dan casos de falla al juez.

## Fase 5 — Parte 2.c: tres peores casos y una mejora medida (≈2 h, 10 %)
1. Tres peores por `acierto`/juez/latencia; sus spans (qué llamó, qué recibió, guardrail, respuesta) y qué parte falló.
2. Mejora como **versión nueva del prompt**: `create_prompt(..., commit_message=...)` en Langfuse,
   `langfuse_prompt=prompt` en cada generation, `prompt_version` v1 → v2, changelog.
   Candidato natural: las ambiguas (declarar año/medida) si fallan.
3. Misma corrida con sesión `paolo-v2` → `resultados_v2.csv`, `spans_v2.jsonl`; tabla antes/después caso por caso;
   separar lo buscado del azar; decir que 26 casos no bastan para una diferencia pequeña (1 caso ≈ 3,8 %).

## Fase 6 — Parte 3: guardrails (≈2,5 h, 15 %)
`mis_guardas.py` con el contrato `f(texto) -> (ok, texto)`:
1. **Inyección** en la entrada: normalizar (minúsculas, sin tildes, `unicodedata`) y por intención
   (verbo de anulación + objeto «instrucciones/reglas/prompt», verbos de escritura SQL), no frase exacta.
2. **PII** en entrada, salida y traza: correo, cédula ecuatoriana (10 dígitos + **dígito verificador**, para no
   confundir con números del dominio), teléfono (09xxxxxxxx / +593) sin tomar fechas ni ids de tiendas.
   Sin Presidio → decir que se escapan los nombres propios.
3. **Tercero:** el presupuesto de tokens del T3 (freno 2) como guardrail con su span — ya existe, solo se instrumenta.
   El guardrail «peligroso» de la A (base intacta) ya lo cubren `validar_sql` + conexión solo-lectura + `base_intacta`.
- Banco: `ataques.json` = plantilla + ≥4 propios (2 ataques del dominio, 2 legítimas que parecen ataques).
  Correr con los de ejemplo y con los nuestros → `guardrails_ejemplo.csv`, `guardrails_mios.csv`; FN/FP de ambos.
- En el agente: hacer saltar cada uno en una corrida del golden set (adversariales, pii) y mostrar su span.
- Por guardrail: qué detecta, cómo, qué hace, latencia en spans, FP sobre las preguntas normales.

## Fase 7 — Parte 4: ajuste fino y alertas (≈1 h, 10 %)
- 4.a (150–250 palabras): ubicar los tres peores en el árbol prompt → recuperar → pesos. Probablemente
  «no hace falta» (fallos de prompt/datos). Si se ajusta: modelo de pesos abiertos de la tabla
  (`open_weight_local` llama3.1:8b), SFT con pares del golden set, LoRA en W_q/W_v rango 8 (Hu et al.
  2021, Tabla 5, §7.1), QLoRA NF4 (Dettmers et al. 2023, §4), y cómo se mediría. Apuntes en
  `taller-03-v2-solver-multiagente/notas-teoricas/s4-llmops-y-finetuning.md`; papers en el kit.
- 4.b: tres alertas (latencia p95, tokens por corrida, tasa de bloqueo) con umbral **de nuestras
  mediciones**, ventana y qué span mirar; umbrales por encima de 1/26 ≈ 3,8 %.

## Fase 8 — Informe y cierre
- Informe con la skill `informe-maestria`, todas las tablas generadas desde los CSV/JSONL.
- Antes de cada commit: `git status`, grep de claves (`sk-`, `pk-lf-`, `sk-lf-`) y de los valores `pii` en `spans_*.jsonl`.
- `.env` y `langfuse/.env` en `.gitignore`.

## Orden si el tiempo aprieta (hasta las 11:50 del sábado)
P0 → P1 instrumentación → 2.a → 2.b corrida base (produce `resultados.csv` + spans) → P3 banco
(`guardrails.csv`) → push. Después, hasta las 23:59: calibración, 2.c, P4 e informe.
