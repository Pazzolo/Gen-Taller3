# Entrega — Taller 04: Evaluar, observar y proteger el agente del Taller 3

Paolo Arrata · MMIA 6013 · USFQ · corridas del 2026-10-16. Opción A del Taller 3 (agente analista SQL).

- **Informe:** `Taller04_LLMOps_Paolo_Arrata.pdf`
- **Repositorio** (`Lab-04-LLMOps/`): agente instrumentado `mi_agente_obs.py`, guardrails `mis_guardas.py`,
  `requirements.txt`, `golden_set.json` (32), `ataques.json` (23), `etiquetas.csv` (18), y `prompts.py`
  (versiones del prompt en Langfuse).
- **Archivos crudos:** `resultados_base.csv` y `resultados_v2.csv`; `spans_base.jsonl` y `spans_v2.jsonl`
  (exportados de Langfuse); `guardrails_ejemplo.csv` y `guardrails_mios.csv`. La primera corrida con 26
  casos queda como `*_26`. Las tablas del informe salen de estos archivos con `python tablas_t4.py`.
- **Evidencia por parte:** `Lab-04-LLMOps/entregables/parte0/` … `parte4/`, cada una con su `NOTAS.md`.
- `Lab-03-Agentes/`: el agente del Taller 3 que se evalúa (sin cambios). `langfuse/`: cómo levantar Langfuse
  con Docker. `fuentes/modelos/`: la tabla de precios que lee `--precio`.

Sin credenciales: ningún `.env` va en la entrega.

---

## Cómo correrlo, paso a paso

Todos los comandos se escriben desde esta carpeta (`entrega/`). Los pasos 1–3 no necesitan red y se
comprobaron el 2026-10-17 en un entorno virtual nuevo; los pasos 4–6 necesitan la H200 (GlobalProtect)
y Docker.

### 1. Requisitos
- Python **3.11** o más nuevo (se usó 3.11.3).
- Solo para los pasos 4–6: la VPN **GlobalProtect** conectada y **Docker Desktop** abierto.

### 2. Entorno y base de datos

```bash
cd Lab-04-LLMOps
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt        # incluye las del Lab 03 (-r ../Lab-03-Agentes/requirements.txt) y langfuse==4.17.0
cd ../Lab-03-Agentes && ../Lab-04-LLMOps/.venv/bin/python crear_base.py && cd ../Lab-04-LLMOps
```

`crear_base.py` escribe `Lab-03-Agentes/data/database.sqlite` (semilla 6013), siempre igual.

### 3. Lo que se puede comprobar sin red

Desde `Lab-04-LLMOps/`:

| Comando | Qué reproduce | Qué tiene que salir |
|---|---|---|
| `.venv/bin/python guion.py ceros` | Parte 0.a | las mismas líneas que `entregables/parte0/0_ceros.txt` |
| `.venv/bin/python guion.py guardrails` | Parte 0.b | 9/17 correctos, 5 falsos negativos, 3 falsos positivos |
| `.venv/bin/python guion.py traza` | Parte 0.c | el correo en la traza, `latency_seconds=0.0`, 2 filas antes y después |
| `.venv/bin/python probar_guardrails.py --casos ataques.json --csv guardrails_ejemplo.csv` | Parte 3, guardrails de ejemplo | 12 correctos de 23 |
| `.venv/bin/python probar_guardrails.py --entrada mis_guardas:validar_entrada --salida mis_guardas:validar_salida --casos ataques.json --csv guardrails_mios.csv` | Parte 3, los nuestros | 22 correctos de 23 (falla D02) |
| `.venv/bin/python evaluar_llmops.py --calibrar resultados_base.csv --etiquetas etiquetas.csv` | Parte 2.b, juez contra la persona | acuerdo 0,944, kappa 0, discrepancia AM03 |
| `.venv/bin/python tablas_t4.py` | todas las tablas del informe | `entregables/tablas.md`, igual a la entregada |

### 4. Levantar Langfuse (Docker)

```bash
cd ../langfuse && ./levantar_langfuse.sh && cd ../Lab-04-LLMOps
```

La primera vez baja unos 4 GB. Si falla con «Ports are not available: … 5432» (un Postgres local ocupa
ese puerto) y Docker Compose es anterior a la 2.24 (no entiende el `!reset` de `sin-puertos-internos.yml`),
borra las líneas `ports:` de `langfuse-worker`, `clickhouse`, `redis` y `postgres` en
`langfuse/.langfuse/docker-compose.yml` y vuelve a correr el script. La interfaz queda en
<http://localhost:3000>; el usuario y la contraseña están en `langfuse/.langfuse/.env`.

### 5. Configurar `.env`

```bash
cp .env.example .env
```

Y en `Lab-04-LLMOps/.env`:
- `AGENT_MODEL=` el id que devuelve `http://172.28.230.10:12555/v1/models` (se usó `zai-org/GLM-5.3-Flash`);
- `JUEZ_MODEL=` uno de otra familia de `http://172.28.230.10:11434/v1/models` (se usó `gemma3:27b`);
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` y `LANGFUSE_BASE_URL`, copiadas de `../langfuse/.env`.

Comprobar: `.venv/bin/python -c "from dotenv import load_dotenv; load_dotenv(); from langfuse import get_client; print(get_client().auth_check())"` → `True`.

### 6. Las corridas (cada una ≈ 30 min con el juez)

Escriben encima de los archivos crudos entregados; para comparar, cambia `--salida` y `-o`.

```bash
# Partes 1 y 2.b: corrida base (prompt v1)
LANGFUSE_SESION=paolo-base32 .venv/bin/python evaluar_llmops.py --golden golden_set.json \
    --agente mi_agente_obs:MiAgenteObs --juez gemma3 --salida resultados_base.csv
sleep 5   # Langfuse tarda unos segundos en dejar consultable lo recién enviado
.venv/bin/python exportar_trazas.py --sesion paolo-base32 -o spans_base.jsonl
.venv/bin/python resumir_trazas.py spans_base.jsonl --precio propietario_economico

# Parte 2.c: registrar v1 y v2 en Langfuse (una sola vez) y correr con v2
.venv/bin/python prompts.py registrar
LANGFUSE_SESION=paolo-v2 PROMPT_VERSION=v2 PROMPT_LANGFUSE=1 .venv/bin/python evaluar_llmops.py \
    --golden golden_set.json --agente mi_agente_obs:MiAgenteObs --juez gemma3 --salida resultados_v2.csv
.venv/bin/python exportar_trazas.py --sesion paolo-v2 -o spans_v2.jsonl

# Las tablas, otra vez
.venv/bin/python tablas_t4.py
```

El modelo muestrea, así que una corrida nueva no da exactamente las mismas cifras; lo que se mantiene en
las repeticiones está en `entregables/parte2c/tabla_repeticiones.md`.

### 7. Regenerar el PDF del informe (opcional)

```bash
.venv/bin/pip install markdown
.venv/bin/python entregables/informe/construir_pdf.py     # usa Google Chrome en modo headless
```

Escribe `entregables/informe/informe.html` e `informe.pdf` a partir de `informe.md`.
