"""Planificador (con la validación de la REGLA 1), investigador y programador.

El planificador PROPONE con el LLM; `validar_plan` DECIDE con código. Si el plan falla, vuelve al
planificador con la lista de problemas (el orquestador cuenta los intentos).
"""
from __future__ import annotations

import json
import re

import networkx as nx

from grafo import Grafo, es_seccion_de_entrega
from lector import Documento
from llm import LLM

TIPOS_SUBTAREA = ("calculo", "redaccion")
FORMATOS = ("md", "pdf", "ipynb")
MAX_SUBTAREAS = 10
BIBLIOTECAS = "numpy, scipy, scikit-learn, matplotlib y la biblioteca estándar (json, math, statistics)"
_NUMEROS = {"una": 1, "un": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5}


# ── La especificación del entregable: lo que el enunciado dice explícitamente lo decide el código ──
def especificacion_por_reglas(doc: Documento) -> dict:
    """Formato, nombre y límites que se pueden leer del enunciado sin interpretar."""
    texto = " ".join(doc.texto.split())
    spec: dict = {}
    if m := re.search(r"\b([\w-]+\.(md|pdf|ipynb))\b", texto):
        spec["archivo"], spec["formato"] = m.group(1), m.group(2)
    elif re.search(r"(?i)\bnotebook\b|jupyter|\.ipynb", texto):
        spec["formato"] = "ipynb"
    elif re.search(r"(?i)\ben\s+\**pdf\b|\breporte\s+en\s+pdf", texto):
        spec["formato"] = "pdf"
    if m := re.search(r"(?i)m[aá]ximo\s*\**\s*([\d][\d\s.  ]*)\s*\**\s*palabras", texto):
        spec["max_palabras"] = int(re.sub(r"\D", "", m.group(1)))
    if m := re.search(r"(?i)(\d+|una|dos|tres|cuatro|cinco)\s*\**\s*p[aá]ginas?\**\s*(como\s+m[aá]ximo)?", texto):
        crudo = m.group(1).lower()
        spec["max_paginas"] = int(crudo) if crudo.isdigit() else _NUMEROS[crudo]
    return spec


SISTEMA_PLANIFICADOR = f"""Eres el planificador de un solver que resuelve tareas de maestría a partir de su enunciado.
Divide la tarea en subtareas. Responde solo JSON:
{{"entregable": {{"formato": "md|pdf|ipynb", "archivo": "nombre.ext", "secciones": ["..."],
                 "max_palabras": null|int, "max_paginas": null|int}},
 "subtareas": [{{"id": "T1", "tipo": "calculo|redaccion", "secciones": ["S1"], "descripcion": "...",
                "depende_de": [], "figura": false}}]}}
Reglas:
- "calculo": algo que se resuelve ejecutando un script Python ({BIBLIOTECAS}). Una por parte del
  enunciado que pida cifras, tablas o figuras. "figura": true si esa parte pide una figura.
- "redaccion": preguntas conceptuales o de discusión que se responden con texto (usando cifras ya calculadas).
- "secciones": los ids de las secciones del enunciado que cubre (S1, S2…). Toda sección con contenido
  tiene que quedar cubierta por al menos una subtarea, salvo la que describe el entregable.
- "depende_de": ids de subtareas cuyos resultados necesita (sin ciclos).
- "entregable": el formato y el archivo que pide el enunciado y las secciones del entregable EN ORDEN,
  con el nombre exacto que usa el enunciado (si no pide secciones, usa los títulos de las partes).
- Como mucho {MAX_SUBTAREAS} subtareas. Las descripciones son concretas: datos, partición, semillas y
  parámetros tal como los dice el enunciado."""


def planificar(llm: LLM, doc: Documento, grafo: Grafo, problemas: list[str] | None = None,
               anterior: dict | None = None) -> dict:
    secciones = "\n\n".join(f"[{s.id}] {s.titulo}\n{s.texto}" for s in doc.secciones)
    aristas = "\n".join(f"- {r}" for r in grafo.resumen()["aristas_regla"]) or "- (ninguna)"
    mensajes = [{"role": "system", "content": SISTEMA_PLANIFICADOR},
                {"role": "user", "content": f"ENUNCIADO, por secciones:\n\n{secciones}\n\n"
                                            f"DEPENDENCIAS entre secciones (extraídas por regla):\n{aristas}"}]
    if problemas:
        mensajes += [{"role": "assistant", "content": json.dumps(anterior, ensure_ascii=False)},
                     {"role": "user", "content": "El plan no pasó la validación. Corrige estos problemas y "
                                                 "devuelve el plan completo:\n- " + "\n- ".join(problemas)}]
    return llm.json("planificador", mensajes)


def validar_plan(plan: dict, doc: Documento) -> list[str]:
    """REGLA 1: el plan se valida con código. Devuelve la lista de problemas (vacía = válido)."""
    problemas: list[str] = []
    if not isinstance(plan, dict) or not isinstance(plan.get("subtareas"), list) or not plan["subtareas"]:
        return ["el plan no tiene una lista «subtareas» no vacía"]
    subs = plan["subtareas"]
    if len(subs) > MAX_SUBTAREAS:
        problemas.append(f"hay {len(subs)} subtareas; el máximo es {MAX_SUBTAREAS}")
    ids = [s.get("id") for s in subs if isinstance(s, dict)]
    if len(ids) != len(subs) or any(not isinstance(i, str) or not i for i in ids):
        problemas.append("toda subtarea necesita un «id» de texto")
    repetidos = sorted({i for i in ids if ids.count(i) > 1})
    if repetidos:
        problemas.append(f"ids repetidos: {repetidos}")
    secciones_validas = {s.id for s in doc.secciones}
    g = nx.DiGraph()
    g.add_nodes_from(i for i in ids if isinstance(i, str))
    for s in subs:
        if not isinstance(s, dict):
            continue
        if s.get("tipo") not in TIPOS_SUBTAREA:
            problemas.append(f"{s.get('id')}: tipo «{s.get('tipo')}» no es uno de {TIPOS_SUBTAREA}")
        if not str(s.get("descripcion", "")).strip():
            problemas.append(f"{s.get('id')}: falta la descripción")
        secs = s.get("secciones") or []
        if not isinstance(secs, list) or not secs:
            problemas.append(f"{s.get('id')}: no dice qué secciones cubre")
        for x in secs if isinstance(secs, list) else []:
            if x not in secciones_validas:
                problemas.append(f"{s.get('id')}: la sección «{x}» no existe (hay {sorted(secciones_validas)})")
        for d in s.get("depende_de") or []:
            if d not in ids:
                problemas.append(f"{s.get('id')}: depende de «{d}», que no existe")
            elif d == s.get("id"):
                problemas.append(f"{s.get('id')}: depende de sí misma")
            else:
                g.add_edge(s["id"], d)
    try:
        ciclo = nx.find_cycle(g)
        problemas.append(f"hay un ciclo de dependencias: {' → '.join(a for a, _ in ciclo)}")
    except nx.NetworkXNoCycle:
        pass
    cubiertas = {x for s in subs if isinstance(s, dict) for x in (s.get("secciones") or []) if isinstance(x, str)}
    for sec in doc.secciones:
        if sec.nivel >= 2 and not es_seccion_de_entrega(sec) and sec.id not in cubiertas:
            problemas.append(f"la sección {sec.id} «{sec.titulo}» no la cubre ninguna subtarea")
    ent = plan.get("entregable") or {}
    if ent.get("formato") not in FORMATOS:
        problemas.append(f"entregable.formato «{ent.get('formato')}» no es uno de {FORMATOS}")
    if not ent.get("archivo"):
        problemas.append("entregable.archivo está vacío")
    reglas = especificacion_por_reglas(doc)
    for clave in ("formato", "archivo", "max_palabras", "max_paginas"):
        if clave in reglas and ent.get(clave) != reglas[clave]:
            problemas.append(f"entregable.{clave} = {ent.get(clave)!r}, pero el enunciado dice {reglas[clave]!r}")
    return problemas


def orden_de_ejecucion(plan: dict) -> list[dict]:
    """Las subtareas en un orden que respeta las dependencias (el plan ya está validado)."""
    g = nx.DiGraph()
    por_id = {s["id"]: s for s in plan["subtareas"]}
    g.add_nodes_from(por_id)
    for s in plan["subtareas"]:
        for d in s.get("depende_de") or []:
            g.add_edge(d, s["id"])
    return [por_id[i] for i in nx.lexicographical_topological_sort(g, key=lambda i: list(por_id).index(i))]


# ── Investigador ───────────────────────────────────────────────────────────────────────────
def investigar(grafo: Grafo, subtarea: dict, resultados_previos: dict, con_grafo: bool = True) -> tuple[str, list[str]]:
    """El contexto de una subtarea, cada fragmento con su cita. REGLA 2 (última parte).

    Con grafo: el preámbulo (suele traer los datos y las tablas), las secciones de la subtarea en su
    texto literal, las secciones de las que dependen (aristas depende_de, siguiéndolas hasta el
    final), la sección del entregable y hasta dos fragmentos de las notas del curso que comparten
    entidades específicas. Sin grafo (versión recortada de la 2.b): los 3 fragmentos más parecidos a
    la descripción de la subtarea, y nada más."""
    doc = grafo.doc
    partes: list[tuple[str, str]] = []
    if con_grafo:
        ids: list[str] = []
        pre = doc.secciones[0]
        if pre.nivel == 1:
            ids.append(pre.id)
        for sid in subtarea.get("secciones", []):
            for x in [sid, *grafo.dependencias(sid)]:
                if x not in ids:
                    ids.append(x)
        ent = grafo.seccion_de_entrega()
        if ent and ent.id not in ids:
            ids.append(ent.id)
        for sid in ids:
            s = grafo.secciones[sid]
            partes.append((s.cita(doc.nombre), f"{s.titulo}\n{s.texto}"))
        vistas = set()
        for sid in subtarea.get("secciones", []):
            for f, comunes in grafo.notas_relacionadas(sid):
                if f.id not in vistas:
                    vistas.add(f.id)
                    partes.append((f.cita() + f" (entidades en común: {', '.join(comunes)})", f.texto))
    else:
        partes = [(c, t) for c, t, _ in grafo.top_k(subtarea.get("descripcion", ""), k=3)]
    for d in subtarea.get("depende_de") or []:
        if d in resultados_previos:
            partes.append((f"[resultados de la subtarea {d}]", json.dumps(resultados_previos[d], ensure_ascii=False)[:3000]))
    texto = "\n\n".join(f"{cita}\n{t}" for cita, t in partes)
    return texto, [c for c, _ in partes]


# ── Programador ────────────────────────────────────────────────────────────────────────────
SISTEMA_PROGRAMADOR = f"""Eres el programador de un solver. Escribe UN script de Python completo y autocontenido que
resuelva la subtarea, siguiendo el enunciado al pie de la letra (datos, partición, semillas, parámetros).
Reglas del entorno (las aplica el código, no tú):
- Solo {BIBLIOTECAS}. No hay pandas ni internet: no descargues nada; los datos vienen de
  scikit-learn (load_*) o del propio enunciado.
- Sin subprocess, os.system, eval/exec, borrar archivos ni rutas fuera de la carpeta actual.
- Guarda TODAS las cifras que calcules en "resultados.json" (un objeto; números en float de Python,
  sin redondear) con claves descriptivas. Imprime las cifras principales con 4 decimales.
- Si la subtarea pide una figura: guárdala como PNG en la carpeta actual con plt.savefig y luego plt.close().
- El script es independiente: si necesita algo de una subtarea anterior, vuelve a calcularlo.
- Nunca uses el conjunto de prueba para elegir hiperparámetros ni para ajustar escaladores.
- No ejecutes ni simules el script en tu razonamiento y no calcules los resultados a mano: el sandbox lo
  ejecuta después y el revisor mira lo que imprime. Tu trabajo es solo escribirlo.
Responde solo con el script en un bloque ```python."""


def programar(llm: LLM, subtarea: dict, contexto: str, previo: str | None = None,
              motivos: list[str] | None = None) -> str:
    mensajes = [{"role": "system", "content": SISTEMA_PROGRAMADOR},
                {"role": "user", "content": f"SUBTAREA {subtarea['id']}: {subtarea['descripcion']}\n"
                                            f"Figura: {'sí' if subtarea.get('figura') else 'no'}\n\nCONTEXTO:\n{contexto}"}]
    if previo is not None:
        mensajes += [{"role": "assistant", "content": f"```python\n{previo}\n```"},
                     {"role": "user", "content": "El revisor rechazó el script. Corrige esto y devuelve el script completo:\n- "
                                                 + "\n- ".join(motivos or [])}]
    texto = llm.chat("programador", mensajes)
    m = re.search(r"```(?:python)?\s*\n(.*?)```", texto, re.S)
    return (m.group(1) if m else texto).strip() + "\n"
