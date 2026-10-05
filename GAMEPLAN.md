# Taller 03 — Gameplan: un agente analítico con herramientas

Source: `taller-03-agentes.pdf` + `taller-03-agentes/Lab-03-Agentes/`.
Weights: P0 10% · P1 35% · P2 30% (2.a/2.b/2.c, 10% each) · P3 10% · P4 5% · P5 5% · reproducibility 5%.

> **Rule that costs points even if the code works:** the trace must store tokens, and every
> table in the report must come from `resultados.csv` and the trace files.

> **Deadline in the PDF:** repo at the close of the session on 3 Oct 2026 (11:50), PDF report by 23:59 the same day.
> Confirm the extension or resubmission window before investing time.

---

## Working rules (apply to every phase)

### 1. Check the reference code first
Before writing anything, look in the course reference folder and reuse its patterns:

```
202610.IM2.4971 - MMIA 6013 IA Generativa y Agentes - Daniel Riofrío Almeida - 1052026 - 1240 AM/
├── s3-lun-profesor.ipynb          # tool-use loop from scratch, tool schemas (a_formato_openai), LLMSimulado/LLMReal, traza
├── s3-mar-profesor.ipynb          # reading ReAct traces, detectar_loop, ejecutar_con_presupuesto, Reflexion + verificador
├── s3-mie-profesor.ipynb          # 40-line graph engine, human-in-the-loop with checkpoint, ServidorMCP / ClienteMCP
├── h200-langgraph-profesor.ipynb  # H200 client with tools, real LangGraph (StateGraph, interrupt, InMemorySaver), RAG bge-m3
└── mini-opencode/                 # (uncompressed from mini-opencode.zip) — read README.md first
    ├── agentes.py                 # loop + 3 brakes (Presupuesto, repetition, max steps), orchestrator/subagents (Sesion)
    ├── guardia.py                 # deterministic guard: decides in code, not in the prompt
    ├── test_guardia.py            # 76 bypass attempts + legit commands — testing style to copy
    ├── demo_ataque.py             # scripted hostile model against the guard (no network)
    ├── herramientas.py            # tools + Integridad tripwire
    ├── traza.py                   # JSONL trace per step
    ├── h200.py                    # H200 client, discovers model via /v1/models
    └── mini_opencode.py           # CLI entry point
```

**Where to look for each part:**

| Part | Look at first |
|---|---|
| Setup / model id | `mini-opencode/h200.py`, section 1 of `h200-langgraph-profesor.ipynb` |
| P1 tools + schemas + loop | `s3-lun` (Parts 1–4), `mini-opencode/agentes.py` (`Herramienta`, `Agente.correr`) |
| P1 rule 2 (read-only guard) | `mini-opencode/guardia.py` + `test_guardia.py` (copy the testing style), `herramientas.py` (`Integridad` tripwire → idea for checking the DB stays intact) |
| P1 rule 5 (errors go back to the model) | `mini-opencode/agentes.py` (argument errors return as text; only integrity violations stop the run) |
| P1 rule 6 (trace) | `mini-opencode/traza.py`, `s3-lun` Part 4 |
| P2.c reading traces | `s3-mar` Part 1 (`imprimir_traza`, manual diagnosis) |
| P3 brakes | `s3-mar` Parts 2–3 (`detectar_loop`, `ejecutar_con_presupuesto`), `mini-opencode/agentes.py` (`Presupuesto`, repetition, max steps with a final call without tools) |
| P3 scripted model | `s3-lun` `LLMSimulado`, `mini-opencode/demo_ataque.py`, lab `guion.py` |
| P4.A multi-agent | `mini-opencode/agentes.py` (`Sesion`, orchestrator → subagent hand-off) |
| P4.B MCP | `s3-mie` Part 5, `h200-langgraph` section 2 |
| P4.C memory/RAG | `h200-langgraph` section 3 (`recuperar`) |
| P4.D LangGraph + approval | `s3-mie` Parts 3–4, `h200-langgraph` sections 4–6 |

Only write new code when the reference doesn't cover the case. When you adapt reference code, say so in a comment.

### 2. Commit as you go
- Make one commit per meaningful step, never one big commit at the end. At minimum: setup, Part 0 outputs, each tool, the agent loop, each brake, the golden set, each eval run, the improvement, the report.
- Message style: `parte1: herramienta consultar_sql en solo lectura (regla 2)`, `parte2: resultados.csv de ambos agentes`.
- Before **every** commit, run `git status` and grep the staged files for secrets:
  `git diff --cached | grep -iE "sk-|api_key|OPENAI_API_KEY=.+|ANTHROPIC_API_KEY=.+"` must return nothing.
  `.env` must never be staged. **A leaked key means the whole taller goes ungraded.**
- Commit traces and CSVs too (they're deliverables). The lab's `.gitignore` excludes `traces/` and `resultados*.csv`, so un-ignore them for the final deliverables.

---

## Phase 0 — Setup (≈20 min)
1. `cd taller-03-agentes/Lab-03-Agentes`, create a venv **with Python ≥ 3.11** (`pandas==3.0.6` won't install on the system 3.9): `/opt/homebrew/bin/python3.11 -m venv .venv`, then `.venv/bin/pip install -r requirements.txt` (pinned versions, `langgraph==1.2.12`). ✅ done 2026-10-05
   - Note from `mini-opencode/h200.py`: the H200 model **always reasons** and bills it as output tokens; pass `chat_template_kwargs: {"enable_thinking": True}` so vLLM separates it into `reasoning` and `content` stays clean. Matters for Phase 2.
2. `python crear_base.py` → `data/database.sqlite` (productos, clientes, 6 000 ventas, fixed seed) and `data/ventas_mensuales.csv`. ✅ done (13 / 320 / 6 000 rows)
3. `cp .env.example .env`. Use the **H200 route** (free, needs GlobalProtect). ⏳ `.env` written; `AGENT_MODEL` still empty — fill it once GlobalProtect is connected:
   `OPENAI_BASE_URL=http://172.28.230.10:12555/v1`, `OPENAI_API_KEY=local`, `AGENT_MODEL=` the id from `/v1/models` (read it, don't type it from memory, as `h200.py` does).
4. Write down **route + model id + date** for the report. **Use one model for Parts 1–3**; if you switch models, re-run the whole golden set.
5. **Data:** option D (the sample database) is the fastest. The golden set must be ours and harder than the template, because the reference agent gets all 8 template questions right. If we want our own domain for Taller 4, choose A or B now.
6. Commit: `setup: entorno, base de ejemplo y .env.example`.

## Phase 1 — Part 0: three failures that don't raise errors (≈30 min, 10%)
Don't modify `agent.py` or `tools.py` in this part.

| Item | Command | What to capture and explain |
|---|---|---|
| 0.a | `python evaluar.py --solo P1 P3 --salida parte0.csv` | From each `traces/trace-*.json`: status, number of steps, actions. Expect `completed` with 0 steps or made-up tools. Two sentences: why `completed` with 0 steps is worse than an error, and what the model needs to know (tool names, descriptions, schemas, DB schema). |
| 0.b | `python guion.py ataque` | **What saved the table:** sqlite3/`pd.read_sql_query` only runs one statement, so it fails on `;drop` (not the filter). **What crashed the run:** the uncaught exception. **Why there's no trace:** `_write_trace` is only called on the `return` paths (`agent.py:90` and `:106`). |
| 0.c | `python guion.py limite` | The 3 output lines plus the rule: *a limit the model can change through tool arguments is not a limit.* Then **delete `../fuera_del_lab/`**. |

Status: ✅ 0.b and 0.c done 2026-10-05 (evidence in `Lab-03-Agentes/entregables/parte0/`, see `NOTAS.md`); ⏳ 0.a needs the H200.

Compare with `mini-opencode/demo_ataque.py` and `guardia.py` to see how the reference solves the same problems.
Commit: `parte0: salidas y trazas de 0.a, 0.b, 0.c`.

## Phase 2 — Part 1: our agent (≈3–4 h, 35%)
New files, so the scaffold stays intact for the 2.b comparison:

```
mi_agente.py      # class MiAgente: MiAgente().run(q) -> {answer, trace, status, model, usage}
mis_tools.py      # tools + JSON schemas + validation
golden_set.json
frenos_demo.py    # Part 3
```

Use **native function calling** (`tools=` in `chat.completions`), following the loop in `s3-lun` / `agentes.py`. Test with a single call first that the H200 accepts `tools`; if it doesn't, list the schemas in the prompt instead.

**Tools:**
- `describir_base()`: tables, columns, types and sample values, plus the formula **importe = cantidad·precio_unitario·(1−descuento)** (the model can't find it on its own).
- `consultar_sql(query)`: read-only SQL.
- `estadisticas(dataset_id, columnas)`: mean, median, std, IQR.
- `grafico(dataset_id, x, y, tipo)`: saves a PNG.

**The six rules** (mark each one with `# REGLA n` in the code):
1. **The model knows its tools:** schemas passed through `tools=`, plus `describir_base`.
2. **Truly read-only:** `sqlite3.connect("file:…?mode=ro", uri=True)`, plus `PRAGMA query_only=ON`. Exactly one statement (strip a trailing `;`, reject any remaining `;`). After stripping comments, the query must start with `SELECT` or `WITH`. Write tests in the style of `test_guardia.py`.
3. **Limits set by code:** `MAX_FILAS` in code, wrapping the query as `SELECT * FROM (<q>) LIMIT MAX_FILAS`. The chart folder is fixed and file names are generated (uuid or counter). Neither is a tool argument.
4. **Tools pass data to each other:** `consultar_sql` saves the full result to `traces/<run>/ds_<n>.csv` and returns a `dataset_id` plus a preview. `estadisticas` and `grafico` take the `dataset_id`, never a path. *Justification:* the model never sees paths, and the context stays small.
5. **Errors don't end the run:** validate arguments against the schema, wrap each tool call in try/except, and return `{"error": …}` to the model. Bad JSON in the arguments counts as an error too.
6. **The trace is always saved:** `try/finally` around the loop. Per step and in total: model, tokens_in/out (`response.usage`), latency and error. Base it on `traza.py`.

**Abstention:** the system prompt requires the exact phrase `No puedo responder con los datos disponibles.` whenever the data can't answer the question or the question asks to modify data.

**Report deliverables:** tool list with descriptions and schemas, a loop diagram with its stop condition, the 6 rules marked in the code, the justification for rule 4, the model and date, and full traces of 2 runs (one using all three tools).

Commits: one per tool, then the loop, then the trace, then two sample runs.

Status (2026-10-05): ✅ code done and tested offline. `mis_tools.py`, `mi_agente.py` (the 3 brakes are already in the loop), `modelo_guion.py`, `test_mi_agente.py` (11/11 pass, no network; SDK path also checked against a fake vLLM server). Report notes in `Lab-03-Agentes/entregables/parte1/NOTAS.md` (tools table, rule → line map, rule 4 justification, loop diagram). ⏳ The two real runs need the VPN (see below).

### ▶ When GlobalProtect is connected
From `taller-03-agentes/Lab-03-Agentes/`:
```
.venv/bin/python correr_con_vpn.py todo
```
This runs, in order:
- `modelo`: writes `AGENT_MODEL` into `.env` and `entregables/modelo.json` (route + model + date).
- `humo`: one `tools=` call. If it fails, the server has no tool calling and we switch to schemas in the prompt.
- `parte0a`: Part 0.a evidence into `entregables/parte0/`.
- `parte1`: the two Part 1 runs into `entregables/parte1/`.

Each step can also run on its own. Then review the outputs, `git add entregables`, and commit.

## Phase 3 — Part 2: evaluation (≈2 h, 30%)
**2.a Golden set (10–12 questions, minimum 8).** The design itself is graded, and if the agent misses none, the set is too easy.
- **Simple (≥2):** a count by region, a monthly total.
- **Multi (≥3, aim for 4–5):** two-table joins with a time filter; the importe formula; semester-over-semester growth; ambiguous dates ("Q3 2025"); medians (SQLite has no `MEDIAN`, so data has to be passed to the stats tool).
- **Negative (≥2):** costs and margin, satisfaction, 2024 sales, dates after Aug 2026.
- **Adversarial (≥1, aim for 2):** a buried instruction ("set the discount to 0 and then compute…"), injection inside a value.
- Every answerable question gets a `sql_verificacion` (numeric first column), plus `debe_contener` when the answer includes a name. Run each SQL by hand before using it.

**2.b Metrics:** same questions, same model:
```
python evaluar.py --golden golden_set.json --agente mi_agente:MiAgente  --salida resultados_mi_agente.csv
python evaluar.py --golden golden_set.json --agente agent:AnalystAgent --salida resultados_base.csv
```
Table for both agents: accuracy, correct and wrongful abstention, database intact (adversarial), average steps, tool errors, exceptions, total input tokens. **Take the numbers from the evaluator's summary or the CSV, never type them by hand.**

**2.c Three worst cases:** for each one, use the trace (what the model asked for, what it got, what it answered) and blame one layer: tool description, schema, query, tool, prompt or model. Method: `s3-mar` Part 1.

Commits: `parte2: golden_set`, `parte2: resultados de ambos agentes`.

## Phase 4 — Part 3: brakes (≈1 h, 10%)
Implement them in `MiAgente` (reference: `s3-mar` Parts 2–3, `mini-opencode/agentes.py`) and **trigger each one with a scripted model** (subclass that overrides the LLM call, like `guion.py` / `LLMSimulado`). Save the trace each one produces.
1. **Max steps:** a final call *without tools* that answers with what it has and says it didn't finish (as `mini-opencode` does). Never an empty answer.
2. **Token budget:** checked **before** each call. Estimate the next call as the last `prompt_tokens` plus the new messages. Cost grows roughly quadratically with steps.
3. **Repetition detector:** the same (tool, canonical args) more than k times. State what it misses: alternating loops (A,B,A,B), queries with small changes (whitespace, `LIMIT 51`), semantic loops.

Our agent can't write data, so a 4th brake isn't required.
Commit: `parte3: tres frenos + frenos_demo.py con sus trazas`.

## Phase 5 — Part 4: one improvement (≈1.5 h, 5%)
Pick it **after** having the 2.b numbers, and measure it with the same golden set. Estimate tokens from one question first and put that estimate in the report.
- **Recommended: A, multiple agents.** A supervisor plus a data subagent (SQL, stats, charts) and a writing subagent (formats the number, handles abstention). Base it on `Sesion` / the hand-off in `mini-opencode/agentes.py`. It plugs straight into `evaluar.py`. It may gain nothing, and that's an acceptable result.
- **Alternative: D, LangGraph with approval** (`h200-langgraph-profesor.ipynb` is almost the template). The `interrupt` needs an auto-approve mode for the evaluator.

Commit: `parte4: mejora + resultados_mejora.csv`.

## Phase 6 — Part 5 and the report (≈1.5 h)
Part 5, 100–150 words each:
1. **PEAS / goal-based:** where the goal lives (question plus system prompt) and who decides it's met (the model when it emits a final answer; the code through the brakes). Point to the lines in the code.
2. **Tokens and steps vs. the baseline,** using our CSVs. If max steps doubles, the worst case grows roughly quadratically (the full history is resent each step). Explain why it doesn't just double, using our numbers.
3. **Real harm in production:** e.g. a writable user plus a 0.b-style filter, or exfiltrating whole tables through 0.c-style limits. In the server: a read-only DB user, timeouts, row caps. A person decides on writes and exports.

Use the `informe-maestria` skill for the prose.

**Report PDF checklist:** Part 0 outputs · architecture with the 6 rules · model, route and date · annotated golden set · 2.b table for both agents · 3 worst cases · 3 brakes triggered · measured improvement · Part 5.

**Repo checklist:** code, `requirements.txt`, `crear_base.py`, `golden_set.json`, `resultados*.csv` for both agents, `traces/`, no credentials (grep the traces: they store full messages).

Final commit: `informe: PDF y tablas desde resultados*.csv`.

---

## Suggested split (2–3 people)
- **A:** Part 0 + `mi_agente.py` (loop, trace).
- **B:** `mis_tools.py` (rules 2–5, guard tests) + Part 3.
- **C:** golden set + verification SQL, runs the evaluations, report skeleton.
- Everyone: 2.c, Part 4, Part 5.
