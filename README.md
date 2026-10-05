# Gen-Taller3 — Taller 03: un agente analítico con herramientas

MMIA 6013 · IA Generativa y Agentes · USFQ. Enunciado: `taller-03-agentes.pdf`.

**Informe:** [`taller-03-agentes/Lab-03-Agentes/entregables/informe/informe.pdf`](taller-03-agentes/Lab-03-Agentes/entregables/informe/informe.pdf)
(fuente: `informe.md`). Plan de trabajo: [`GAMEPLAN.md`](GAMEPLAN.md).

## Qué hay en `taller-03-agentes/Lab-03-Agentes/`

| Archivo | Qué es |
|---|---|
| `agent.py`, `tools.py`, `guion.py`, `evaluar.py`, `crear_base.py`, `preguntas_plantilla.json` | el laboratorio tal como se repartió (sin modificar) |
| `mis_tools.py`, `mi_agente.py` | Parte 1: el agente corregido (seis reglas marcadas `# REGLA n`) y sus tres frenos |
| `mi_agente_multi.py` | Parte 4.A: supervisor + subagentes de datos y redacción |
| `modelo_guion.py`, `test_mi_agente.py` | modelo sin LLM y 15 pruebas sin red |
| `golden_set.json`, `estres_adversarial.json` | Parte 2 (22 preguntas comentadas) y prueba de estrés de la Parte 4 |
| `resultados_*.csv` | salidas de `evaluar.py` para cada agente y corrida |
| `frenos_demo.py`, `correr_con_vpn.py`, `tabla_2b.py` | Parte 3, corridas en la H200 y tablas generadas desde los CSV |
| `traces/`, `entregables/parteN/` | trazas de todas las corridas y la evidencia de cada parte, con `NOTAS.md` |

## Correr

```bash
cd taller-03-agentes/Lab-03-Agentes
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python crear_base.py
cp .env.example .env    # descomentar OPENAI_BASE_URL de la H200, OPENAI_API_KEY=local (requiere GlobalProtect)
.venv/bin/python correr_con_vpn.py modelo     # escribe AGENT_MODEL leyendo /v1/models
.venv/bin/python test_mi_agente.py            # sin red
.venv/bin/python evaluar.py --golden golden_set.json --agente mi_agente:MiAgente
```

Comandos completos por parte: sección «Reproducibilidad» del informe. Sin credenciales en el repositorio:
`.env` está en `.gitignore`, y la carpeta de material de referencia del curso (`202610.*`) no se versiona.
