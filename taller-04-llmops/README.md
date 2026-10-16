# Taller 04 — Evaluar, observar y proteger el agente del Taller 3

MMIA 6013 · IA Generativa y Agentes · USFQ. Enunciado: `../Taller4/taller-04-llmops.pdf`. Plan: `../GAMEPLAN-T4.md`.

**Informe:** [`Lab-04-LLMOps/entregables/informe/informe.pdf`](Lab-04-LLMOps/entregables/informe/informe.pdf)
(fuente `informe.md`; comandos completos en su sección «Reproducibilidad»).

| Qué | Dónde |
|---|---|
| Agente instrumentado (Langfuse) y guardrails | `Lab-04-LLMOps/mi_agente_obs.py`, `mis_guardas.py` |
| Versiones del prompt (Langfuse) | `Lab-04-LLMOps/prompts.py`, `entregables/parte2c/CHANGELOG_prompt.md` |
| Golden set, banco de ataques, etiquetas | `golden_set.json` (32), `ataques.json` (23), `etiquetas.csv` (18) |
| Archivos crudos | `resultados_base.csv`, `resultados_v2.csv`, `spans_base.jsonl`, `spans_v2.jsonl`, `guardrails_ejemplo.csv`, `guardrails_mios.csv` |
| Tablas del informe | `python tablas_t4.py` → `entregables/tablas.md` |
| Evidencia por parte | `entregables/parte0/` … `parte4/`, con su `NOTAS.md` |
| Kit del curso | `Lab-04-LLMOps/` (los archivos del kit, sin modificar) y `langfuse/`; `Lab-03-Agentes` es un enlace a `../taller-03-agentes/Lab-03-Agentes` |
| Precios | `fuentes/modelos/modelos-2026-1.json`, transcrito del anexo del enunciado (el kit no lo traía) |

Credenciales solo en `.env` (fuera de git). Langfuse corre en Docker en esta máquina (`langfuse/levantar_langfuse.sh`).
