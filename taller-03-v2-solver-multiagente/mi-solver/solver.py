"""Solver multiagente con GraphRAG — Taller 03 v2. El orquestador es un grafo de estados de LangGraph.

    python solver.py ../solver-v2/enunciados/tarea-a-generativo-discriminativo.pdf corridas/tarea-A
    cd ../solver-v2 && python evaluar_solver.py --golden ../mi-solver/golden_tareas.json \\
        --solver solver:Solver --ruta ../mi-solver --corridas ../mi-solver/corridas

Contrato (enunciado, Parte 1): `Solver().solve(pdf, salida)` devuelve status, entregables, subtareas,
usage, model y trace; `.run(pregunta)` es lo mismo para el Taller 4, con la ruta del enunciado.

    leer ─► indexar ─► planificar ─► validar ─┬─(problemas, < 3 intentos)─► planificar
                                              └─(válido)─► elegir ◄─────────────────────────────┐
    elegir ─┬─(cálculo)─► investigar ─► programar ─► revisar_codigo ─┬─(rechazo)─► programar  │
            │                                                        ├─(red)─► aprobar_red ─┐  │
            │                                                        └─(ok)──► ejecutar ◄───┘  │
            │                                           ejecutar ─► revisar ─┬─(aprobado)──────┤
            │                                                                ├─(rechazo, < máx)─► programar
            │                                                                └─(máx. o repetido)─► fallida ─┘
            └─(sin subtareas o sin presupuesto)─► redactar ─► comprobar ─┬─(problemas, < 3)─► redactar
                                                                         └─► publicar ─► FIN

Frenos (Parte 3): presupuesto revisado antes de cada llamada, con reserva para el redactor (`llm.py`);
máximo de intentos por subtarea y detector de repetición (`_hash` de los scripts rechazados); tiempo
máximo del sandbox, que mata el grupo de procesos (`sandbox.py`); y confirmación humana antes de usar
la red (`aprobar_red`).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

import agentes
import redactor
import revisor
import sandbox
from grafo import Grafo, es_seccion_de_entrega
from lector import leer
from llm import LLM, Presupuesto, PresupuestoAgotado
from traza import Traza

MAX_INTENTOS = 3          # por subtarea (freno 2)
MAX_PLANES = 3            # REGLA 1: intentos del planificador
MAX_RONDAS_REDACCION = 3  # REGLA 5: vueltas al redactor


def _hash(codigo: str) -> str:
    return hashlib.sha256("\n".join(l.rstrip() for l in codigo.strip().splitlines()).encode()).hexdigest()[:16]


def negar_red(subtarea: dict, motivos: list[str]) -> bool:
    """Por defecto nadie aprueba: sin una persona delante, un script que quiere red no corre."""
    return False


def preguntar_en_consola(subtarea: dict, motivos: list[str]) -> bool:
    print(f"\n[aprobación] La subtarea {subtarea['id']} quiere usar la red: {motivos}")
    return input("¿Lo permites? [s/N] ").strip().lower() == "s"


class Estado(TypedDict, total=False):
    intentos_plan: int
    problemas_plan: list[str]
    cola: list[str]
    actual: str
    intento: int
    motivos: list[str]
    red_aprobada: bool
    sin_presupuesto: str
    rondas: int
    problemas_entrega: list[str]


class Solver:
    def __init__(self, con_grafo: bool = True, con_revisor: bool = True, max_intentos: int = MAX_INTENTOS,
                 presupuesto: int = 600_000, reserva_redactor: int = 60_000, timeout_s: float = sandbox.TIMEOUT_S,
                 llm_factory: Callable[[Traza, Presupuesto], LLM] | None = None,
                 aprobar_red: Callable[[dict, list[str]], bool] = negar_red):
        self.con_grafo, self.con_revisor = con_grafo, con_revisor
        self.max_intentos = max_intentos if con_revisor else 1
        self.presupuesto_cfg = (presupuesto, reserva_redactor)
        self.timeout_s = timeout_s
        self.llm_factory = llm_factory or (lambda traza, p: LLM(traza, p))
        self.aprobar_red = aprobar_red

    # ── contrato ─────────────────────────────────────────────────────────────────────────────
    def run(self, pregunta: str) -> dict:
        salida = Path("corridas") / Path(pregunta).stem
        r = self.solve(pregunta, str(salida))
        texto = Path(r["entregables"][0]).name if r["entregables"] else "(sin entregable)"
        return {"answer": f"Entregable: {texto} ({r['status']})", "trace": r["trace"], "status": r["status"],
                "model": r["model"], "usage": r["usage"], **r}

    def solve(self, ruta_pdf: str, salida: str) -> dict:
        self.salida = Path(salida).resolve()
        if self.salida.exists():
            shutil.rmtree(self.salida)
        self.salida.mkdir(parents=True)
        self.traza = Traza(self.salida / "traza.jsonl")             # REGLA 6: desde el primer evento
        self.presupuesto = Presupuesto(*self.presupuesto_cfg)
        self.pdf, self.doc, self.grafo, self.plan = ruta_pdf, None, None, None
        self.subtareas: dict[str, dict] = {}
        self.resultados: dict[str, dict] = {}
        self.scripts: dict[str, str] = {}
        self.rechazados: dict[str, set[str]] = {}
        self.entregable: Path | None = None
        self.llm: LLM | None = None
        t0 = time.perf_counter()
        error = None
        self.traza("inicio", pdf=str(ruta_pdf), salida=str(self.salida), con_grafo=self.con_grafo,
                   con_revisor=self.con_revisor, presupuesto=self.presupuesto.max_tokens)
        try:
            self.llm = self.llm_factory(self.traza, self.presupuesto)
            self.traza("modelo", modelo=self.llm.modelo)
            self._grafo().invoke({}, {"recursion_limit": 400})
        except Exception as err:   # el contrato devuelve un dict; la traza dice qué pasó
            error = f"{type(err).__name__}: {err}"
            self.traza("error", error=error[:2000])
        finally:
            info = self._informe(error, time.perf_counter() - t0)
            self.traza("fin", **{k: v for k, v in info.items() if k != "trace"})
        return info

    # ── el grafo de estados ──────────────────────────────────────────────────────────────────
    def _grafo(self):
        g = StateGraph(Estado)
        for nombre in ("leer", "indexar", "planificar", "validar", "elegir", "investigar", "programar",
                       "revisar_codigo", "aprobar_red", "ejecutar", "revisar", "fallida", "redactar", "comprobar", "publicar"):
            g.add_node(nombre, self._nodo(nombre, getattr(self, f"n_{nombre}")))
        g.add_edge(START, "leer")
        g.add_edge("leer", "indexar")
        g.add_edge("indexar", "planificar")
        g.add_edge("planificar", "validar")
        g.add_conditional_edges("validar", self.r_validar, ["planificar", "elegir", "publicar"])
        g.add_conditional_edges("elegir", self.r_elegir, ["investigar", "redactar"])
        g.add_edge("investigar", "programar")
        g.add_conditional_edges("programar", self.r_programar, ["revisar_codigo", "redactar"])
        g.add_conditional_edges("revisar_codigo", self.r_revisar_codigo, ["programar", "aprobar_red", "ejecutar", "fallida"])
        g.add_conditional_edges("aprobar_red", self.r_aprobar_red, ["ejecutar", "programar", "fallida"])
        g.add_edge("ejecutar", "revisar")
        g.add_conditional_edges("revisar", self.r_revisar, ["elegir", "programar", "fallida", "redactar"])
        g.add_edge("fallida", "elegir")
        g.add_conditional_edges("redactar", self.r_redactar, ["comprobar", "publicar"])
        g.add_conditional_edges("comprobar", self.r_comprobar, ["redactar", "publicar"])
        g.add_edge("publicar", END)
        return g.compile()

    def _nodo(self, nombre: str, f):
        def envuelto(estado: Estado) -> dict:
            self.traza("nodo", nodo=nombre, subtarea=estado.get("actual"), intento=estado.get("intento"))
            try:
                return f(estado) or {}
            except PresupuestoAgotado as err:   # freno 1: se pasa a redactar con lo que hay
                return {"sin_presupuesto": str(err)}
        return envuelto

    # ── nodos ────────────────────────────────────────────────────────────────────────────────
    def n_leer(self, e: Estado) -> dict:
        self.doc = leer(self.pdf)
        self.traza("lector", "lector", paginas=len(self.doc.paginas), secciones=[s.titulo for s in self.doc.secciones],
                   avisos=self.doc.avisos)
        if not any(s.texto.strip() for s in self.doc.secciones):
            raise RuntimeError(f"el PDF no tiene texto legible: {self.doc.avisos}")
        return {}

    def n_indexar(self, e: Estado) -> dict:
        self.grafo = Grafo(self.doc, self.llm, con_entidades=self.con_grafo)
        self.grafo.guardar(self.salida / "grafo.json")
        self.traza("grafo", "indexador", **self.grafo.resumen())
        return {}

    def n_planificar(self, e: Estado) -> dict:
        self.plan = agentes.planificar(self.llm, self.doc, self.grafo, e.get("problemas_plan"), self.plan)
        return {"intentos_plan": e.get("intentos_plan", 0) + 1}

    def n_validar(self, e: Estado) -> dict:
        problemas = agentes.validar_plan(self.plan, self.doc)               # REGLA 1
        self.traza("plan", "planificador", intento=e.get("intentos_plan"), valido=not problemas, problemas=problemas)
        (self.salida / "plan.json").write_text(json.dumps(self.plan, ensure_ascii=False, indent=2), encoding="utf-8")
        if problemas:
            return {"problemas_plan": problemas}
        for s in agentes.orden_de_ejecucion(self.plan):
            self.subtareas[s["id"]] = {"id": s["id"], "tipo": s["tipo"], "status": "pendiente", "intentos": 0, "def": s}
        return {"problemas_plan": [], "cola": [s["id"] for s in agentes.orden_de_ejecucion(self.plan)]}

    def r_validar(self, e: Estado) -> str:
        if e.get("sin_presupuesto"):
            return "publicar"
        if e.get("problemas_plan"):
            return "planificar" if e.get("intentos_plan", 0) < MAX_PLANES else "publicar"
        return "elegir"

    def n_elegir(self, e: Estado) -> dict:
        cola = list(e.get("cola", []))
        while cola:
            sid = cola.pop(0)
            st = self.subtareas[sid]
            if st["tipo"] == "redaccion":
                st["status"] = "para_redactor"
                continue
            fallidas = [d for d in st["def"].get("depende_de") or [] if self.subtareas.get(d, {}).get("status") == "fallida"]
            self.traza("subtarea", sid, descripcion=st["def"]["descripcion"][:300], dependencias_fallidas=fallidas)
            return {"cola": cola, "actual": sid, "intento": 0, "motivos": [], "red_aprobada": False}
        return {"cola": [], "actual": None}

    def r_elegir(self, e: Estado) -> str:
        return "investigar" if e.get("actual") and not e.get("sin_presupuesto") else "redactar"

    def n_investigar(self, e: Estado) -> dict:
        st = self.subtareas[e["actual"]]
        contexto, citas = agentes.investigar(self.grafo, st["def"], self.resultados, con_grafo=self.con_grafo)
        st["contexto"] = contexto
        self.traza("investigacion", "investigador", subtarea=st["id"], citas=citas, caracteres=len(contexto))
        return {}

    def n_programar(self, e: Estado) -> dict:
        st = self.subtareas[e["actual"]]
        st["intentos"] += 1
        previo = st.get("codigo") if e.get("motivos") else None
        st["codigo"] = agentes.programar(self.llm, st["def"], st["contexto"], previo, e.get("motivos"))
        return {"intento": st["intentos"]}

    def r_programar(self, e: Estado) -> str:
        return "redactar" if e.get("sin_presupuesto") else "revisar_codigo"

    def n_revisar_codigo(self, e: Estado) -> dict:
        st = self.subtareas[e["actual"]]
        h = _hash(st["codigo"])
        if h in self.rechazados.get(st["id"], set()):                       # freno 2: repetición
            motivo = "script idéntico a uno ya rechazado: no se vuelve a ejecutar"
            self.traza("freno", st["id"], freno="repeticion", intento=st["intentos"], motivo=motivo, hash=h)
            return {"motivos": [motivo], "red_aprobada": False}
        rev = sandbox.revisar(st["codigo"])                                # REGLA 3, capa 1
        self.traza("revision_estatica", st["id"], intento=st["intentos"], rechazos=rev.rechazos, requiere_red=rev.requiere_red)
        if rev.rechazos:
            self.rechazados.setdefault(st["id"], set()).add(h)
            return {"motivos": [f"sandbox: {r}" for r in rev.rechazos], "red_aprobada": False}
        if rev.requiere_red:
            return {"motivos": rev.requiere_red, "red_aprobada": None}
        return {"motivos": [], "red_aprobada": False}

    def r_revisar_codigo(self, e: Estado) -> str:
        if not e.get("motivos"):
            return "ejecutar"
        if e.get("red_aprobada") is None:
            return "aprobar_red"
        return "programar" if self.subtareas[e["actual"]]["intentos"] < self.max_intentos else "fallida"

    def n_aprobar_red(self, e: Estado) -> dict:
        st = self.subtareas[e["actual"]]
        ok = bool(self.aprobar_red(st["def"], e["motivos"]))             # freno 4: decide una persona
        self.traza("freno", st["id"], freno="red", aprobado=ok, motivos=e["motivos"])
        if ok:
            return {"red_aprobada": True, "motivos": []}
        self.rechazados.setdefault(st["id"], set()).add(_hash(st["codigo"]))
        return {"red_aprobada": False, "motivos": [f"una persona NO aprobó usar la red ({'; '.join(e['motivos'])}): "
                                                   "usa datos incluidos en la biblioteca o en el enunciado"]}

    def r_aprobar_red(self, e: Estado) -> str:
        if e.get("red_aprobada"):
            return "ejecutar"
        return "programar" if self.subtareas[e["actual"]]["intentos"] < self.max_intentos else "fallida"

    def n_ejecutar(self, e: Estado) -> dict:
        st = self.subtareas[e["actual"]]
        carpeta = self.salida / "trabajo" / f"{st['id']}-intento{st['intentos']}"
        ej = sandbox.ejecutar(st["codigo"], carpeta, timeout=self.timeout_s, con_red=bool(e.get("red_aprobada")))
        st["ejecucion"], st["carpeta"] = ej, carpeta
        self.traza("ejecucion", st["id"], intento=st["intentos"], codigo_salida=ej.codigo_salida, duracion_s=ej.duracion_s,
                   archivos_creados=ej.archivos_creados, timeout=ej.timeout, con_sandbox_exec=ej.con_sandbox_exec,
                   stderr=ej.stderr[-800:] if ej.codigo_salida != 0 else "")
        if ej.timeout:
            self.traza("freno", st["id"], freno="tiempo_sandbox", intento=st["intentos"], duracion_s=ej.duracion_s)
        return {}

    def n_revisar(self, e: Estado) -> dict:
        st = self.subtareas[e["actual"]]
        ej, carpeta = st["ejecucion"], st["carpeta"]
        if not self.con_revisor:                       # versión recortada de la 2.b: sin revisión
            res, _ = sandbox.leer_resultados(carpeta)
            self._aprobar(st, res or {}, "sin revisor")
            return {"motivos": []}
        motivos, res = revisor.revisar_con_codigo(ej, carpeta, st["codigo"], bool(st["def"].get("figura")))  # REGLA 4
        origen = "codigo"
        if not motivos:
            juicio = revisor.revisar_con_llm(self.llm, st["def"], st["contexto"], st["codigo"], ej, res)
            origen = "llm"
            if not juicio["aprobado"]:
                motivos = [f"revisor LLM: {p}" for p in juicio["problemas"]] + ([f"corrección: {juicio['correccion']}"] if juicio["correccion"] else [])
        self.traza("revision", "revisor", subtarea=st["id"], intento=st["intentos"], aprobado=not motivos,
                   decidio=origen, motivos=motivos)
        if motivos:
            self.rechazados.setdefault(st["id"], set()).add(_hash(st["codigo"]))
            return {"motivos": motivos}
        self._aprobar(st, res, origen)
        return {"motivos": []}

    def _aprobar(self, st: dict, res: dict, por: str) -> None:
        st["status"] = "aprobada"
        self.resultados[st["id"]] = {"resultados": res, "stdout": st["ejecucion"].stdout[-2500:],
                                     "figuras": [f for f in st["ejecucion"].archivos_creados if f.endswith(".png")]}
        self.scripts[st["id"]] = st["codigo"]

    def r_revisar(self, e: Estado) -> str:
        if not e.get("motivos"):
            return "elegir"
        if e.get("sin_presupuesto"):
            return "redactar"
        return "programar" if self.subtareas[e["actual"]]["intentos"] < self.max_intentos else "fallida"

    def n_fallida(self, e: Estado) -> dict:
        st = self.subtareas[e["actual"]]
        st["status"] = "fallida"
        self.traza("freno", st["id"], freno="max_intentos", intentos=st["intentos"], ultimos_motivos=e.get("motivos", [])[:5])
        return {"actual": None}

    def n_redactar(self, e: Estado) -> dict:
        spec = dict(self.plan.get("entregable") or {}) if self.plan else {}
        spec.update(agentes.especificacion_por_reglas(self.doc))
        spec.setdefault("archivo", {"md": "reporte.md", "pdf": "reporte.pdf", "ipynb": "solucion.ipynb"}.get(spec.get("formato"), "reporte.md"))
        self.spec = spec
        figuras = []
        (self.salida / "codigo").mkdir(exist_ok=True)
        for sid, r in self.resultados.items():
            (self.salida / "codigo" / f"{sid}.py").write_text(self.scripts[sid], encoding="utf-8")
            for f in r["figuras"]:
                origen = self.subtareas[sid]["carpeta"] / f
                destino = self.salida / f"{sid}_{Path(f).name}"
                shutil.copy(origen, destino)
                figuras.append(destino.name)
        faltan = [f"{s['id']}: {s['def']['descripcion'][:120]}" for s in self.subtareas.values() if s["status"] in ("fallida", "pendiente")]
        redaccion = [{"id": s["id"], "descripcion": s["def"]["descripcion"]} for s in self.subtareas.values() if s["tipo"] == "redaccion"]
        if faltan or e.get("sin_presupuesto"):
            redaccion.append({"id": "AVISO", "descripcion": "Di explícitamente en el entregable qué no se pudo completar y por qué: "
                                                           + "; ".join(faltan) + (f" (presupuesto: {e['sin_presupuesto']})" if e.get("sin_presupuesto") else "")})
        es_nb = spec.get("formato") == "ipynb"
        md = redactor.escribir_markdown(self.llm, "\n\n".join(f"{s.titulo}\n{s.texto}" for s in self.doc.secciones), spec,
                                        self.resultados, redaccion, figuras, e.get("problemas_entrega"),
                                        getattr(self, "_md_previo", None), para_notebook=es_nb)
        self._md_previo = md
        destino = self.salida / spec["archivo"]
        if spec.get("formato") == "pdf":
            escala, paginas = 1.0, redactor.md_a_pdf(md, destino, self.salida)
            while spec.get("max_paginas") and paginas > spec["max_paginas"] and escala > 0.72:
                escala -= 0.07                                           # primero achica el código, no el LLM
                paginas = redactor.md_a_pdf(md, destino, self.salida, escala)
            (self.salida / "reporte_fuente.md").write_text(md, encoding="utf-8")
        elif es_nb:
            redactor.armar_notebook(destino, self.doc, self.plan, self.scripts, md)
            ok, err = redactor.ejecutar_notebook(destino)
            self.traza("ejecucion", "redactor", subtarea="notebook", codigo_salida=0 if ok else 1, stderr=err[-600:])
        else:
            destino.write_text(md, encoding="utf-8")
        self.entregable = destino
        return {"rondas": e.get("rondas", 0) + 1}

    def r_redactar(self, e: Estado) -> str:
        return "publicar" if e.get("sin_presupuesto") and not self.entregable else "comprobar"

    def n_comprobar(self, e: Estado) -> dict:
        texto_enunciado = "\n".join(self.doc.paginas)
        problemas = redactor.comprobar(self.entregable, self.spec, self.salida, texto_enunciado)   # REGLA 5
        self.traza("comprobacion", "redactor", ronda=e.get("rondas"), problemas=problemas)
        return {"problemas_entrega": problemas}

    def r_comprobar(self, e: Estado) -> str:
        if e.get("problemas_entrega") and e.get("rondas", 0) < MAX_RONDAS_REDACCION and not e.get("sin_presupuesto"):
            return "redactar"
        return "publicar"

    def n_publicar(self, e: Estado) -> dict:
        self.problemas_finales = e.get("problemas_entrega", []) + e.get("problemas_plan", [])
        self.sin_presupuesto = e.get("sin_presupuesto")
        return {}

    # ── informe ──────────────────────────────────────────────────────────────────────────────
    def _informe(self, error: str | None, segundos: float) -> dict:
        calc = [s for s in self.subtareas.values() if s["tipo"] == "calculo"]
        problemas = getattr(self, "problemas_finales", [])
        if not self.entregable or not self.entregable.exists() or error:
            status = "fallido"
        elif problemas or any(s["status"] != "aprobada" for s in calc) or getattr(self, "sin_presupuesto", None):
            status = "parcial"
        else:
            status = "completado"
        uso = self.traza.uso_por_agente()
        return {
            "status": status,
            "entregables": [str(self.entregable)] if self.entregable and self.entregable.exists() else [],
            "subtareas": [{"id": s["id"], "tipo": s["tipo"], "status": s["status"], "intentos": s["intentos"]}
                          for s in self.subtareas.values()],
            "usage": {"tokens_entrada": sum(a["tokens_entrada"] for a in uso.values()),
                      "tokens_salida": sum(a["tokens_salida"] for a in uso.values()), "por_agente": uso},
            "model": self.llm.modelo if self.llm else "",
            "trace": str(self.salida / "traza.jsonl"),
            "problemas": problemas, "error": error, "duracion_s": round(segundos, 1),
            "sin_presupuesto": getattr(self, "sin_presupuesto", None),
        }


if __name__ == "__main__":
    pdf = sys.argv[1]
    salida = sys.argv[2] if len(sys.argv) > 2 else f"corridas/{Path(pdf).stem}"
    r = Solver().solve(pdf, salida)
    print(json.dumps({k: v for k, v in r.items() if k != "usage"} | {"usage": {k: v for k, v in r["usage"].items() if k != "por_agente"}},
                     ensure_ascii=False, indent=2))
