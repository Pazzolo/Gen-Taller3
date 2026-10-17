# Lab-03-Agentes (lo mínimo para correr el Taller 4)

El agente del Taller 3 que evalúa el Taller 4, sin cambios: `mi_agente.py` (`MiAgente`), `mis_tools.py`
(sus herramientas), `evaluar.py` (el verificador determinístico que importa `evaluar_llmops.py`) y
`crear_base.py`. Además `agent.py` y `tools.py`, el agente de ejemplo del kit sin modificar, porque
`guion.py` (Parte 0) los importa a través de `instrumented_agent.py`. El Taller 3 completo (trazas, golden set de 22, informe) se entregó aparte.

    python crear_base.py      # crea data/database.sqlite, que el Taller 4 usa
