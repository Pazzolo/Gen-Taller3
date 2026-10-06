"""Pruebas del solver sin red ni LLM (estilo `mini-opencode/test_guardia.py`).

    ../solver-v2/.venv/bin/python test_solver.py

Comprueban las piezas que son código: lector, aristas por regla (REGLA 2), sandbox (REGLA 3) y
revisión con código (REGLA 4). Las trazas y carpetas van a una carpeta temporal.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
KIT = AQUI.parent / "solver-v2"
TMP = Path(tempfile.mkdtemp(prefix="test_solver_"))
fallos: list[str] = []


def caso(nombre: str):
    def deco(f):
        try:
            f()
            print(f"  ✓ {nombre}")
        except AssertionError as e:
            fallos.append(nombre)
            print(f"  ✗ {nombre}: {e}")
        return f
    return deco


# ── Lector y reglas del grafo ───────────────────────────────────────────────────────────────
from grafo import referencias  # noqa: E402
from lector import leer  # noqa: E402


def _aristas(pdf: str) -> set[tuple[str, str, str]]:
    d = leer(pdf)
    t = {s.id: s.titulo.split(" —")[0] for s in d.secciones}
    return {(t[o], t[x], r) for o, x, _, r in referencias(d.secciones)}


@caso("lector: secciones de las cinco tareas por tamaño de letra, sin avisos")
def _():
    for pdf, n in ((KIT / "enunciados/tarea-a-generativo-discriminativo.pdf", 6), (KIT / "enunciados/tarea-c-recuperacion-lexica-lsa.pdf", 6),
                   (AQUI / "enunciados/tarea-e-ridge-validacion-cruzada.pdf", 7)):
        d = leer(pdf)
        assert len(d.secciones) == n and not d.avisos, (pdf.name, [s.titulo for s in d.secciones], d.avisos)


@caso("lector: la tabla del corpus de C queda como tabla (una fila por documento)")
def _():
    texto = leer(KIT / "enunciados/tarea-c-recuperacion-lexica-lsa.pdf").secciones[0].texto
    assert "|d04|BM25 es una función de ranking léxica que pondera la frecuencia de términos y la longitud del documento.|" in texto


@caso("regla 2: Tarea A, Parte 3 —depende_de→ Parte 1 (la arista de la 0.b)")
def _():
    assert ("Parte 3", "Parte 1", "depende_de") in _aristas(KIT / "enunciados/tarea-a-generativo-discriminativo.pdf")


@caso("regla 2: «las mismas consultas» (C) es depende_de; una mención hacia adelante (E) no")
def _():
    assert ("Parte 2", "Parte 1", "depende_de") in _aristas(KIT / "enunciados/tarea-c-recuperacion-lexica-lsa.pdf")
    e = _aristas(AQUI / "enunciados/tarea-e-ridge-validacion-cruzada.pdf")
    assert ("Parte 1", "Parte 3", "referencia_adelante") in e and ("Parte 1", "Parte 3", "depende_de") not in e


# ── Sandbox ─────────────────────────────────────────────────────────────────────────────────
from sandbox import SANDBOX_EXEC, ejecutar, revisar  # noqa: E402

RECHAZAR = {
    "import socket": "socket", "import subprocess": "subprocess", "from urllib.request import urlopen": "urllib",
    "import requests": "requests", "import os\nos.system('ls')": "os.system", "import os\nos.remove('a')": "os.remove",
    "import shutil": "shutil", "eval('1+1')": "eval", "exec('x=1')": "exec", "__import__('os')": "__import__",
    "open('/etc/passwd').read()": "ruta fuera", "open('../fuera.txt', 'w')": "ruta fuera",
    "from pathlib import Path\nPath('a').unlink()": "borra", "import os\nprint(os.environ)": "os.environ",
    "x = __builtins__.__dict__['__imp' + 'ort__']('socket')": "__builtins__",
    "x = ().__class__.__bases__[0].__subclasses__()": "__subclasses__", "import multiprocessing": "multiprocessing",
    "import ctypes": "ctypes", "def f(:\n  pass": "no compila",
}
PERMITIR = [
    "import json, numpy as np\njson.dump({'a': float(np.mean([1, 2]))}, open('resultados.json', 'w'))",
    "import os\nos.makedirs('figs', exist_ok=True)\nprint(os.path.join('figs', 'a.png'))",
    "import matplotlib.pyplot as plt\nplt.plot([1, 2]); plt.savefig('curva.png')",
    "from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes\nX, y = load_wine(return_X_y=True)",
    "print('ratio a/b', 3 / 4)",
]


@caso(f"regla 3: la revisión estática rechaza {len(RECHAZAR)} intentos peligrosos")
def _():
    malos = [c for c, esperado in RECHAZAR.items() if not any(esperado in r for r in revisar(c).rechazos)]
    assert not malos, malos


@caso(f"regla 3: y deja pasar {len(PERMITIR)} scripts legítimos")
def _():
    malos = [(c, revisar(c).rechazos) for c in PERMITIR if not revisar(c).ok]
    assert not malos, malos


@caso("freno 4 (marca): fetch_openml, download=True y URLs piden red, no se rechazan")
def _():
    r = revisar("from sklearn.datasets import fetch_openml\nX = fetch_openml('mnist_784')\n"
                "import pandas as pd\nd = pd.read_csv('https://x.org/a.csv')\nf(download=True)")
    assert r.ok and len(r.requiere_red) == 3, (r.rechazos, r.requiere_red)


@caso("regla 3: el script no ve las variables del usuario (ninguna clave)")
def _():
    os.environ["OPENAI_API_KEY"] = "sk-no-debe-llegar"
    e = ejecutar("import json, os\njson.dump(sorted(getattr(os, 'env' + 'iron')), open('resultados.json','w'))", TMP / "env")
    vistas = (TMP / "env" / "resultados.json").read_text()
    assert e.codigo_salida == 0 and "OPENAI_API_KEY" not in vistas and "HOME" in vistas, vistas


@caso("regla 3: la capa del sistema niega la red y la escritura fuera de la carpeta" + ("" if SANDBOX_EXEC else " (sin sandbox-exec: se omite)"))
def _():
    if not SANDBOX_EXEC:
        return
    red = ("import json\nm = __builtins__.__dict__['__imp' + 'ort__']('socket')\ns = m.socket(); s.settimeout(3)\n"
           "try:\n    s.connect(('1.1.1.1', 80)); r = 'CONECTÓ'\nexcept Exception as e:\n    r = type(e).__name__\n"
           "json.dump({'r': r}, open('resultados.json', 'w'))")
    ejecutar(red, TMP / "red")
    assert "PermissionError" in (TMP / "red" / "resultados.json").read_text()
    fuera = TMP / "fuera.txt"
    ejecutar(f"try:\n    open({str(fuera)!r}, 'w').write('x')\nexcept Exception:\n    pass", TMP / "escribe")
    assert not fuera.exists()


@caso("freno 3: el tiempo máximo mata al script y a sus hijos")
def _():
    codigo = ("import subprocess, time\np = subprocess.Popen(['sleep', '300'])\n"
              "open('hijo.pid', 'w').write(str(p.pid))\ntime.sleep(300)")
    e = ejecutar(codigo, TMP / "hijo", timeout=3)
    time.sleep(0.5)
    pid = int((TMP / "hijo" / "hijo.pid").read_text())
    try:
        os.kill(pid, 0)
        vivo = True
    except ProcessLookupError:
        vivo = False
    assert e.timeout and e.codigo_salida is None and not vivo and e.duracion_s < 10, (e.timeout, vivo, e.duracion_s)


@caso("regla 6 (ejecución): código de salida, duración y archivos creados")
def _():
    e = ejecutar("import json, matplotlib.pyplot as plt\nplt.plot([1]); plt.savefig('f.png')\n"
                 "json.dump({'a': 1}, open('resultados.json', 'w'))", TMP / "creados")
    assert e.codigo_salida == 0 and e.archivos_creados == ["f.png", "resultados.json"] and e.duracion_s > 0, e.archivos_creados


# ── Revisión con código (REGLA 4) ───────────────────────────────────────────────────────────
from revisor import revisar_con_codigo  # noqa: E402

_spec = importlib.util.spec_from_file_location("c0", KIT / "parte0/c_exit_cero.py")
_c0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_c0)
ESCALA = ("import json\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.preprocessing import StandardScaler\n"
          "from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\n"
          "X, y = load_breast_cancer(return_X_y=True)\nX = StandardScaler().fit_transform(X)\n"
          "Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)\n"
          "acc = LogisticRegression(max_iter=5000).fit(Xtr, ytr).score(Xte, yte)\n"
          "json.dump({'lr_accuracy': acc}, open('resultados.json', 'w'))")
BIEN = ESCALA.replace("X = StandardScaler().fit_transform(X)\n", "").replace(
    "acc = LogisticRegression(max_iter=5000).fit(Xtr, ytr).score(Xte, yte)",
    "sc = StandardScaler().fit(Xtr)\nacc = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr).score(sc.transform(Xte), yte)")


def _revisar(codigo: str, nombre: str, figura: bool = False) -> list[str]:
    e = ejecutar(codigo, TMP / nombre)
    return revisar_con_codigo(e, TMP / nombre, codigo, pide_figura=figura)[0]


@caso("regla 4: el script de la 0.c (árbol evaluado sobre su entrenamiento) se rechaza por fuga y plausibilidad")
def _():
    m = _revisar(_c0.CODIGO, "c0")
    assert any(x.startswith("fuga_fit") for x in m) and any(x.startswith("plausible") for x in m), m


@caso("regla 4: el escalador ajustado antes de partir (la fuga que la 0.c no ve) se rechaza")
def _():
    m = _revisar(ESCALA, "escala")
    assert any(x.startswith("fuga_escala") for x in m), m


@caso("regla 4: el mismo experimento bien hecho pasa al LLM")
def _():
    assert _revisar(BIEN, "bien") == []


@caso("regla 4: código de salida, JSON ausente, NaN y figura faltante se rechazan")
def _():
    assert any(x.startswith("salida") for x in _revisar("raise ValueError('x')", "falla"))
    m = _revisar("import json\njson.dump({'media': float('nan')}, open('resultados.json', 'w'))", "nan", figura=True)
    assert any(x.startswith("nan") for x in m) and any(x.startswith("figura") for x in m), m
    assert any(x.startswith("resultados") for x in _revisar("print('hola')", "sinjson"))


# ── Plan (REGLA 1) ──────────────────────────────────────────────────────────────────────────
from agentes import validar_plan  # noqa: E402

DOC_A = leer(KIT / "enunciados/tarea-a-generativo-discriminativo.pdf")
ENT_A = {"formato": "md", "archivo": "reporte.md", "secciones": ["Introducción"], "max_palabras": 1200}


def _sub(i, secs, deps=(), tipo="calculo"):
    return {"id": i, "tipo": tipo, "secciones": list(secs), "descripcion": "d", "depende_de": list(deps)}


PLAN_OK = {"entregable": ENT_A, "subtareas": [_sub("T1", ["S1"]), _sub("T2", ["S2"], ["T1"]), _sub("T3", ["S3"], ["T1"]),
                                              _sub("T4", ["S4"], ["T2", "T3"], "redaccion")]}


@caso("regla 1: un plan correcto pasa")
def _():
    assert validar_plan(PLAN_OK, DOC_A) == []


@caso("regla 1: ids repetidos, dependencia inexistente, ciclo, sección sin cubrir y formato equivocado se detectan")
def _():
    malos = {
        "ids repetidos": {"entregable": ENT_A, "subtareas": [_sub("T1", ["S1", "S2", "S3"]), _sub("T1", ["S4"])]},
        "no existe": {"entregable": ENT_A, "subtareas": [_sub("T1", ["S1", "S2", "S3", "S4"], ["T9"])]},
        "ciclo": {"entregable": ENT_A, "subtareas": [_sub("T1", ["S1", "S2"], ["T2"]), _sub("T2", ["S3", "S4"], ["T1"])]},
        "no la cubre": {"entregable": ENT_A, "subtareas": [_sub("T1", ["S1", "S2"])]},
        "pero el enunciado dice": {"entregable": {**ENT_A, "formato": "pdf"}, "subtareas": PLAN_OK["subtareas"]},
    }
    sin_detectar = [k for k, plan in malos.items() if not any(k in p for p in validar_plan(plan, DOC_A))]
    assert not sin_detectar, sin_detectar


# ── El orquestador de punta a punta con un modelo de guion ──────────────────────────────────
from llm import LLMGuion  # noqa: E402
from solver import Solver  # noqa: E402

_FUGA = ("import json\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.naive_bayes import GaussianNB\n"
         "X, y = load_breast_cancer(return_X_y=True)\nacc = GaussianNB().fit(X, y).score(X, y)\n"
         "json.dump({'nb_accuracy': acc}, open('resultados.json', 'w'))")
_BIEN = ("import json\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.naive_bayes import GaussianNB\n"
         "from sklearn.model_selection import train_test_split\nX, y = load_breast_cancer(return_X_y=True)\n"
         "Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)\n"
         "acc = GaussianNB().fit(Xtr, ytr).score(Xte, yte)\njson.dump({'nb_accuracy': acc}, open('resultados.json', 'w'))\nprint(f'NB {acc:.4f}')")
_PLAN = {"entregable": {"formato": "md", "archivo": "reporte.md", "secciones": ["Introducción", "Metodología", "Resultados", "Discusión", "Conclusiones"],
                        "max_palabras": 1200, "max_paginas": None},
         "subtareas": [_sub("T1", ["S1", "S2", "S3"]), _sub("T2", ["S4"], ["T1"], "redaccion")]}


def _secciones(cuerpo_resultados: str) -> str:
    return f"## Introducción\nx\n## Metodología\nx\n## Resultados\n{cuerpo_resultados}\n## Discusión\nx\n## Conclusiones\nx\n"


def _guion(programas, redacciones):
    return {"indexador": lambda m: {"entidades": []}, "planificador": [_PLAN],
            "programador": [f"```python\n{p}\n```" for p in programas],
            "revisor": lambda m: {"aprobado": True, "problemas": [], "correccion": ""}, "redactor": redacciones}


@caso("orquestador: el revisor rechaza la fuga, el segundo script se aprueba, el reporte se publica")
def _():
    r = Solver(llm_factory=lambda t, p: LLMGuion(t, _guion([_FUGA, _BIEN], [_secciones("NB obtuvo 0.9474.")]), p)).solve(
        str(KIT / "enunciados/tarea-a-generativo-discriminativo.pdf"), str(TMP / "e2e"))
    import json as _j
    revs = [_j.loads(l) for l in open(TMP / "e2e" / "traza.jsonl") if '"revision"' in l]
    assert r["status"] == "completado" and r["subtareas"][0]["intentos"] == 2, (r["status"], r["subtareas"], r["error"])
    assert [(x["aprobado"], x["decidio"]) for x in revs] == [(False, "codigo"), (True, "llm")], revs
    assert (TMP / "e2e" / "plan.json").exists() and (TMP / "e2e" / "codigo" / "T1.py").exists()


@caso("regla 5: una cifra inventada vuelve al redactor; la versión corregida se publica")
def _():
    r = Solver(llm_factory=lambda t, p: LLMGuion(t, _guion([_BIEN], [_secciones("NB obtuvo 0.9415."), _secciones("NB obtuvo 0.9474.")]), p)).solve(
        str(KIT / "enunciados/tarea-a-generativo-discriminativo.pdf"), str(TMP / "e2e5"))
    import json as _j
    comp = [_j.loads(l)["problemas"] for l in open(TMP / "e2e5" / "traza.jsonl") if '"comprobacion"' in l]
    assert len(comp) == 2 and "0.9415" in comp[0][0] and comp[1] == [] and r["status"] == "completado", comp


@caso("orquestador: una respuesta vacía del programador cuenta como intento fallido, no tumba la corrida")
def _():
    r = Solver(llm_factory=lambda t, p: LLMGuion(t, _guion(["", _BIEN], [_secciones("NB obtuvo 0.9474.")]), p)).solve(
        str(KIT / "enunciados/tarea-a-generativo-discriminativo.pdf"), str(TMP / "vacia"))
    assert r["status"] == "completado" and r["subtareas"][0]["intentos"] == 2 and r["error"] is None, (r["status"], r["subtareas"], r["error"])
    assert (TMP / "vacia" / "trabajo" / "T1-intento2" / "script.py").exists()


if __name__ == "__main__":
    print(f"{len(fallos)} fallos" if fallos else "todas las pruebas pasaron")
    sys.exit(1 if fallos else 0)
