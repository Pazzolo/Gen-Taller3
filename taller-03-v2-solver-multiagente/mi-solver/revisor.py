"""Revisor: primero comprobaciones con código y, solo si pasan, el LLM. REGLA 4.

El orden importa (Parte 0.c): un `returncode == 0` con un `resultados.json` no es un experimento.
Las comprobaciones con código no se pueden convencer, así que van primero y filtran lo que es
mecánicamente falso. Al LLM llega solo lo que ya pasó ese filtro, y su error queda acotado a juicios
(«¿esto responde a la subtarea?»), no a aceptar una exactitud de 1,000.

Comprobaciones con código (las de `solver-v2/parte0/c_exit_cero.py` y algunas más):
    salida       código de salida 0 y sin tiempo vencido
    resultados   `resultados.json` existe, es un objeto JSON y no está vacío
    nan          ningún número del JSON es NaN o infinito
    figura       si la subtarea pide figura, hay un PNG no vacío
    fuga_fit     se predice sobre la misma variable con que se ajustó (`.fit(X)` … `.predict(X)`)
    fuga_escala  un escalador o transformador se ajusta con datos que después se parten
                 (`StandardScaler().fit_transform(X)` antes de `train_test_split(X, …)`)
    plausible    métricas de desempeño ≥ 0,999 en un problema con ruido
"""
from __future__ import annotations

import ast
import math
import re
from pathlib import Path

from sandbox import Ejecucion, leer_resultados

# Solo métricas de clasificación o regresión sobre un conjunto de prueba con ruido: ahí un 1,000 es
# sospechoso. En recuperación (Hit, MRR por consulta) o en agrupamiento un 1,0 puede ser legítimo
# (en la Tarea C, cuatro de las seis consultas tienen el relevante en el rango 1).
METRICAS = ("acc", "exact", "f1", "r2", "auc", "precision", "recall")


def es_metrica(ruta: str) -> bool:
    """La clave nombra una métrica si alguna de sus PALABRAS empieza como una métrica.

    Con subcadenas, «fracciones» contenía «acc» y la fracción 1,0 (el 100 % del entrenamiento) se
    rechazó como «exactitud implausible» en la primera corrida real de la Tarea A (2026-10-05)."""
    return any(p.startswith(METRICAS) for p in re.split(r"[^a-z0-9]+", ruta.lower()) if p)


def _numeros(x, ruta=""):
    if isinstance(x, bool):
        return
    if isinstance(x, (int, float)):
        yield ruta, x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from _numeros(v, f"{ruta}.{k}" if ruta else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _numeros(v, f"{ruta}[{i}]")


def fuga_fit(codigo: str) -> list[str]:
    """Variables que aparecen en `.fit(X…)` y otra vez en `.predict(X)`/`.score(X…)` (0.c)."""
    ajustadas, evaluadas = set(), []
    for n in ast.walk(ast.parse(codigo)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args and isinstance(n.args[0], ast.Name):
            if n.func.attr in {"fit", "fit_transform", "fit_predict"}:
                ajustadas.add(n.args[0].id)
            elif n.func.attr in {"predict", "predict_proba", "score", "decision_function"}:
                evaluadas.append(n.args[0].id)
    return sorted({v for v in evaluadas if v in ajustadas})


def fuga_escala(codigo: str) -> list[str]:
    """Un `fit`/`fit_transform` sobre una variable que después se parte con train_test_split.

    Es la fuga que la comprobación anterior no ve (pregunta de la 0.c): el escalador aprende la media
    y la desviación del conjunto de prueba, y la exactitud resultante es plausible. Sigue siendo una
    heurística por nombres: si el script renombra la variable entre medio, no la ve."""
    arbol = ast.parse(codigo)
    llamadas = sorted((n for n in ast.walk(arbol) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))),
                      key=lambda n: (n.lineno, n.col_offset))
    ajustes: dict[str, int] = {}
    hallazgos = []
    for n in llamadas:
        nombre = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
        primero = n.args[0].id if n.args and isinstance(n.args[0], ast.Name) else None
        if nombre in {"fit", "fit_transform"} and primero:
            ajustes.setdefault(primero, n.lineno)
        if nombre == "train_test_split" and primero in ajustes and ajustes[primero] < n.lineno:
            hallazgos.append(f"«{primero}» se usa para ajustar en la línea {ajustes[primero]} y se parte en la {n.lineno}")
    return hallazgos


def revisar_con_codigo(ejec: Ejecucion, carpeta: Path, codigo: str, pide_figura: bool = False) -> tuple[list[str], dict | None]:
    """Motivos de rechazo (vacío = pasa) y los resultados leídos."""
    motivos: list[str] = []
    if ejec.timeout:
        motivos.append("salida: superó el tiempo máximo y se mató")
    elif ejec.codigo_salida != 0:
        motivos.append(f"salida: código {ejec.codigo_salida}; stderr: {ejec.stderr.strip()[-600:]}")
    resultados, error = leer_resultados(carpeta)
    if error:
        motivos.append(f"resultados: {error}")
    else:
        raros = [r for r, v in _numeros(resultados) if isinstance(v, float) and (math.isnan(v) or math.isinf(v))]
        if raros:
            motivos.append(f"nan: valores NaN o infinitos en {raros[:5]}")
        altos = [f"{r}={v}" for r, v in _numeros(resultados)
                 if isinstance(v, float) and es_metrica(r) and 0.999 <= v <= 1.0]
        if altos:
            motivos.append(f"plausible: métricas ≥ 0,999 en un problema con ruido: {altos[:5]}")
    if pide_figura and not any(p.stat().st_size > 0 for p in Path(carpeta).glob("*.png")):
        motivos.append("figura: la subtarea pide una figura y no hay ningún PNG")
    try:
        if f := fuga_fit(codigo):
            motivos.append(f"fuga_fit: se predice sobre lo mismo que se ajustó: {f}")
        if f := fuga_escala(codigo):
            motivos.append(f"fuga_escala: se ajusta antes de partir: {f}")
    except SyntaxError:
        pass
    return motivos, resultados


SISTEMA_REVISOR = """Eres el revisor de un solver que resuelve tareas de maestría. Un script ya pasó las comprobaciones
automáticas (terminó bien, escribió resultados.json, sin NaN ni fugas evidentes). Decide si cumple
LITERALMENTE lo que pide la subtarea y su sección del enunciado: datos, partición, semillas,
parámetros, métricas y figuras. No juzgues el estilo. No pidas cosas que el enunciado no pide.
Responde solo JSON: {"aprobado": true|false, "problemas": ["..."], "correccion": "qué cambiar, concreto"}"""


def revisar_con_llm(llm, subtarea: dict, contexto: str, codigo: str, ejec: Ejecucion, resultados: dict) -> dict:
    import json
    r = llm.json("revisor", [
        {"role": "system", "content": SISTEMA_REVISOR},
        {"role": "user", "content": f"SUBTAREA: {json.dumps(subtarea, ensure_ascii=False)}\n\n"
                                    f"CONTEXTO DEL ENUNCIADO:\n{contexto[:6000]}\n\nSCRIPT:\n```python\n{codigo[:8000]}\n```\n\n"
                                    f"STDOUT:\n{ejec.stdout[-2000:]}\n\nresultados.json:\n{json.dumps(resultados, ensure_ascii=False)[:3000]}"}])
    if not isinstance(r, dict):
        return {"aprobado": False, "problemas": ["el revisor no devolvió un objeto"], "correccion": ""}
    return {"aprobado": bool(r.get("aprobado")), "problemas": list(r.get("problemas") or [])[:8],
            "correccion": str(r.get("correccion") or "")[:1500]}
