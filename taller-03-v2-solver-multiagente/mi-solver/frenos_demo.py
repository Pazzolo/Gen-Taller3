"""Parte 3 — los cuatro frenos, cada uno activado en una corrida completa del solver, con su traza.

    ../solver-v2/.venv/bin/python frenos_demo.py

Con un modelo de guion (`llm.LLMGuion`) para no gastar (lo permite el enunciado): solo cambian las
respuestas del LLM; el orquestador, el sandbox, el revisor y la traza son los de `solver.py`. Cada
demostración usa el enunciado de la Tarea A y copia su carpeta de corrida a entregables/parte3/<freno>/.

    1_presupuesto    el programador se queda sin presupuesto; el redactor usa su reserva, entrega y dice qué faltó
    2_intentos       el programador repite el script rechazado: no se vuelve a ejecutar; al máximo de
                     intentos la subtarea queda fallida y el solver sigue con la siguiente
    3_tiempo         un script lanza un hijo y se cuelga: el sandbox mata el grupo entero al vencer el tiempo
    4_red_negada     un script quiere descargar datos; nadie lo aprueba: no corre y vuelve al programador
    4_red_aprobada   el mismo script con una persona que aprueba: corre (con la red habilitada en el sandbox)
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from llm import LLMGuion
from solver import Solver

AQUI = Path(__file__).resolve().parent
PDF = str(AQUI.parent / "solver-v2" / "enunciados" / "tarea-a-generativo-discriminativo.pdf")
DESTINO = AQUI / "entregables" / "parte3"
TMP = Path(tempfile.mkdtemp(prefix="frenos_"))

SECCIONES = ["Introducción", "Metodología", "Resultados", "Discusión", "Conclusiones"]
ENTREGABLE = {"formato": "md", "archivo": "reporte.md", "secciones": SECCIONES, "max_palabras": 1200, "max_paginas": None}


def plan(*subtareas):
    return {"entregable": ENTREGABLE, "subtareas": list(subtareas)}


def sub(i, secs, deps=()):
    return {"id": i, "tipo": "calculo", "secciones": list(secs), "descripcion": f"subtarea {i}", "depende_de": list(deps), "figura": False}


def reporte(texto: str = "x") -> str:
    return "\n".join(f"## {s}\n{texto}\n" for s in SECCIONES)


BIEN = ("import json\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.naive_bayes import GaussianNB\n"
        "from sklearn.model_selection import train_test_split\nX, y = load_breast_cancer(return_X_y=True)\n"
        "Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)\n"
        "acc = GaussianNB().fit(Xtr, ytr).score(Xte, yte)\njson.dump({'nb_accuracy': acc}, open('resultados.json', 'w'))\n")
FUGA = ("import json\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.naive_bayes import GaussianNB\n"
        "X, y = load_breast_cancer(return_X_y=True)\nacc = GaussianNB().fit(X, y).score(X, y)\n"
        "json.dump({'nb_accuracy': acc}, open('resultados.json', 'w'))\n")
CUELGA = ("import json, time, threading\nimport numpy as np\nhijo = threading.Thread(target=time.sleep, args=(600,))\nhijo.start()\n"
          "time.sleep(600)\njson.dump({'nunca': 1}, open('resultados.json', 'w'))\n")
DESCARGA = ("import json\n\ndef cargar(nombre, download=False):\n    return [1.0, 2.0, 3.0]   # sin conexión real: basta para el freno\n\n"
            "datos = cargar('mnist', download=True)\njson.dump({'media': sum(datos) / len(datos)}, open('resultados.json', 'w'))\n")


def py(codigo: str) -> str:
    return f"```python\n{codigo}```"


def correr(nombre: str, guion: dict, **kw) -> dict:
    salida = TMP / nombre
    r = Solver(llm_factory=lambda t, p: LLMGuion(t, guion, p, tokens=kw.pop("tokens", (1000, 200))), **kw).solve(PDF, str(salida))
    eventos = [json.loads(l) for l in open(salida / "traza.jsonl", encoding="utf-8")]
    frenos = [e for e in eventos if e["tipo"] == "freno"]
    print(f"\n── {nombre}: status={r['status']} · subtareas={[(s['id'], s['status'], s['intentos']) for s in r['subtareas']]}")
    for f in frenos:
        print(f"   freno «{f['freno']}» ({f['agente']}): " +
              json.dumps({k: v for k, v in f.items() if k not in ('t', 'hora', 'tipo', 'agente', 'freno')}, ensure_ascii=False)[:260])
    for e in eventos:
        if e["tipo"] == "ejecucion" and e["agente"] != "redactor":
            print(f"   ejecución {e['agente']} intento {e['intento']}: código {e['codigo_salida']}, {e['duracion_s']} s, timeout={e['timeout']}")
    destino = DESTINO / nombre
    if destino.exists():
        shutil.rmtree(destino)
    shutil.copytree(salida, destino, ignore=shutil.ignore_patterns("tmp", ".mpl", "grafo.json"))
    return {"r": r, "frenos": frenos, "eventos": eventos}


def base(**extra) -> dict:
    g = {"indexador": lambda m: {"entidades": []}, "revisor": lambda m: {"aprobado": True, "problemas": [], "correccion": ""},
         "redactor": lambda m: reporte("Sin cifras calculadas: ver qué faltó.")}
    g.update(extra)
    return g


def main() -> int:
    DESTINO.mkdir(parents=True, exist_ok=True)
    resumen = {}

    # 1 · presupuesto: 14 000 tokens con 4 000 de reserva para el redactor; cada llamada cuesta 1 200.
    #     Indexar (6 llamadas) y planificar (1) gastan 8 400; el programador hace un intento y el
    #     segundo ya no cabe en los 10 000 que no son reserva. El redactor sí cabe en la reserva.
    d = correr("1_presupuesto", base(planificador=[plan(sub("T1", ["S1", "S2"]), sub("T2", ["S3", "S4"]))],
                                     programador=lambda m: py(FUGA)),
               presupuesto=14_000, reserva_redactor=4_000)
    resumen["1_presupuesto"] = (d["r"]["status"], bool(d["r"]["entregables"]),
                                any(f["freno"] == "presupuesto" and f["agente"] != "redactor" for f in d["frenos"]))

    # 2 · intentos y repetición: el programador entrega SIEMPRE el script con fuga
    d = correr("2_intentos", base(planificador=[plan(sub("T1", ["S1", "S2"]), sub("T2", ["S3", "S4"]))],
                                  programador=lambda m: py(FUGA if "SUBTAREA T1" in m[1]["content"] else BIEN)))
    resumen["2_intentos"] = ([s["status"] for s in d["r"]["subtareas"]],
                             sum(e["tipo"] == "ejecucion" and e["agente"] == "T1" for e in d["eventos"]),
                             [f["freno"] for f in d["frenos"]])

    # 3 · tiempo: el script lanza un hilo y se cuelga; tiempo máximo de 4 s
    d = correr("3_tiempo", base(planificador=[plan(sub("T1", ["S1", "S2", "S3", "S4"]))], programador=[py(CUELGA), py(BIEN)]),
               timeout_s=4)
    resumen["3_tiempo"] = [f["freno"] for f in d["frenos"]]

    # 4 · red: el script quiere descargar; primero nadie lo aprueba, después una persona sí
    d = correr("4_red_negada", base(planificador=[plan(sub("T1", ["S1", "S2", "S3", "S4"]))], programador=[py(DESCARGA), py(BIEN)]))
    resumen["4_red_negada"] = [(f["freno"], f.get("aprobado")) for f in d["frenos"]]
    d = correr("4_red_aprobada", base(planificador=[plan(sub("T1", ["S1", "S2", "S3", "S4"]))], programador=[py(DESCARGA)]),
               aprobar_red=lambda subtarea, motivos: True)
    resumen["4_red_aprobada"] = [(f["freno"], f.get("aprobado")) for f in d["frenos"]]

    (DESTINO / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(resumen, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
