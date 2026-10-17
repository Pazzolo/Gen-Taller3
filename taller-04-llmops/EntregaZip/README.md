# Taller 04 — Entrega · Paolo Arrata · MMIA 6013 · USFQ

Opción A del Taller 3 (agente analista SQL). Corridas del 2026-10-16 en la H200 (agente
`zai-org/GLM-5.3-Flash`, juez `gemma3:27b`), con Langfuse 4.50.0.

**Informe en PDF:** `Taller04_LLMOps_Paolo_Arrata.pdf`

**Repositorio** (`repositorio/`)

| Archivo | Qué es |
|---|---|
| `mi_agente_obs.py` | el agente instrumentado: `MiAgente` del Taller 3 con spans de Langfuse (corrida, guardrail, llm, herramienta) |
| `prompts.py` | las versiones del prompt (v1, v2) registradas en Langfuse; `mi_agente_obs.py` la usa para la mejora |
| `mis_guardas.py` | los guardrails: inyección por intención, datos personales, claves en la salida, y la máscara de Langfuse |
| `requirements.txt` | dependencias con versiones fijadas |
| `golden_set.json` | 32 casos: 4 simples (2 con `pii`), 15 multi, 4 ambiguas, 4 negativas, 5 adversariales |
| `ataques.json` | banco de guardrails: los 17 de la plantilla + 6 propios (D01–D06) |
| `etiquetas.csv` | 18 etiquetas aprueba/falla de una persona, para calibrar al juez |

**Archivos crudos** (`archivos_crudos/`) — de aquí salen las tablas del informe

| Archivo | Qué es |
|---|---|
| `resultados_base.csv` | `evaluar_llmops.py`, versión base (prompt v1), 32 casos |
| `resultados_v2.csv` | ídem, versión mejorada (prompt v2) |
| `guardrails_mios.csv` | `probar_guardrails.py` con `mis_guardas.py` sobre `ataques.json` |
| `guardrails_ejemplo.csv` | ídem con los guardrails de ejemplo (`guardrails.py`) |
| `spans_base.jsonl`, `spans_v2.jsonl` | los spans de esas dos corridas, exportados de Langfuse con `exportar_trazas.py` |

Sin credenciales: ningún `.env` va en la entrega. El código completo, con el kit del curso y la
evidencia de cada parte, está en el repositorio (`taller-04-llmops/`).
