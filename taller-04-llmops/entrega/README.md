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

Para correr: `python Lab-03-Agentes/crear_base.py`, después en `Lab-04-LLMOps/`
`pip install -r requirements.txt`, `cp .env.example .env` con las rutas de la H200 y las claves de Langfuse,
y los comandos de la sección «Reproducibilidad» del informe. Sin credenciales: ningún `.env` va en la entrega.
