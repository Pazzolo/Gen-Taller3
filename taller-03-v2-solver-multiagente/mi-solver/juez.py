"""Parte 4, mejora E: un juez de otra familia (gemma3, en el Ollama de la H200) califica cada entregable con
los criterios del enunciado, y se compara con las comprobaciones del golden set: acuerdo y kappa de Cohen.

    ../solver-v2/.venv/bin/python juez.py                 # califica y compara todo
    ../solver-v2/.venv/bin/python juez.py --solo-comparar # recalcula acuerdo y kappa desde juez_resultados.csv

El juez solo lee texto, así que se compara en las comprobaciones que se pueden decidir leyendo: secciones,
extensión en palabras, cifras y contenido. Las de archivo, notebook ejecutado, páginas y procedencia
necesitan mirar archivos, no leer, y quedan fuera. El juez no ejecuta nada y no conoce las respuestas:
juzga si cada cifra «está bien» igual que lo haría un lector del reporte.

Para que el kappa tenga sentido hacen falta casos que fallen, y el solver completo aprueba casi todo. Por
eso entran también las soluciones de la validación del golden set (correcta y con la trampa, Tareas E y F).
Escribe entregables/parte4/: juez_resultados.csv (una fila por comprobación), juez_resumen.json y la
traza juez_traza.jsonl (por llamada: modelo, tokens, latencia, error).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "solver-v2"))
import evaluar_solver as ev  # noqa: E402

from llm import HOST, PUERTO_EMB  # noqa: E402  (el Ollama de la H200: el mismo de los embeddings)

MODELO = "gemma3:27b"
SALIDA = AQUI / "entregables" / "parte4"
MAX_CAR = 24_000          # del entregable; ninguno de los actuales llega a esto

# Qué pide cada comprobación, dicho como lo diría la rúbrica (las de secciones, palabras y contenido se
# describen solas a partir del golden set).
CIFRAS = {
    "A05": "Reporta correctamente cuántos ejemplos tiene el conjunto de datos.",
    "A06": "Reporta correctamente la exactitud en prueba de Naive Bayes gaussiano entrenado con todo el entrenamiento.",
    "A07": "Reporta correctamente la exactitud en prueba de la regresión logística entrenada con todo el entrenamiento.",
    "B03": "Reporta correctamente la entropía en bits de la distribución con T = 0.5.",
    "B04": "Reporta correctamente la entropía en bits de la distribución con T = 1.",
    "B05": "Reporta correctamente la entropía en bits de la distribución con T = 2.",
    "B06": "Reporta correctamente la probabilidad renormalizada de t0 después de aplicar top-p con p = 0.9.",
    "C04": "Reporta correctamente el MRR de la búsqueda TF-IDF.",
    "C05": "Reporta correctamente el hit@3 de la búsqueda TF-IDF.",
    "C06": "Reporta correctamente el MRR de la búsqueda con LSA.",
    "E04": "Reporta correctamente el alpha elegido por validación cruzada sobre el entrenamiento.",
    "E05": "Reporta correctamente el R² en prueba del modelo Ridge con el alpha elegido.",
    "E06": "Reporta correctamente el RMSE en prueba del modelo Ridge con el alpha elegido.",
    "F04": "Reporta correctamente el ARI entre los grupos del k elegido y las clases reales.",
    "F05": "Reporta correctamente la silueta del k elegido.",
}
JUZGABLES = {"secciones", "palabras_max", "cifra", "cifra_presente", "contiene"}

CONJUNTOS = {   # nombre → (carpeta con tarea-*/, CSV del evaluador)
    "completo": (AQUI / "resultados/completo/corridas", AQUI / "resultados/completo/parciales"),
    "sin_grafo": (AQUI / "resultados/sin_grafo/corridas", AQUI / "resultados/sin_grafo/parciales"),
    "validacion_correcta": (AQUI / "entregables/parte2/validacion/correcta",
                            AQUI / "entregables/parte2/validacion/resultados_correcta.csv"),
    "validacion_trampa": (AQUI / "entregables/parte2/validacion/trampa",
                          AQUI / "entregables/parte2/validacion/resultados_trampa.csv"),
}


def criterio(c: dict) -> str:
    if c["tipo"] == "secciones":
        orden = ", en ese orden" if c.get("en_orden") else ""
        return f"Tiene como encabezados las secciones {', '.join(c['lista'])}{orden}."
    if c["tipo"] == "palabras_max":
        return f"Tiene como máximo {c['valor']} palabras, sin contar el código."
    if c["tipo"] == "contiene":
        return f"Menciona y analiza explícitamente: {', '.join(c['lista'])}."
    return CIFRAS[c["id"]]


def resultados_checks(ruta: Path) -> dict[tuple[str, str], int]:
    archivos = sorted(ruta.glob("resultados_solver_*.csv")) if ruta.is_dir() else [ruta]
    return {(f["tarea"], f["check"]): int(f["ok"]) for a in archivos for f in csv.DictReader(a.open(encoding="utf-8"))}


def enunciado(tarea: dict) -> str:
    pdf = (AQUI / tarea["pdf"]).resolve()
    md = pdf.with_suffix(".md")
    return md.read_text(encoding="utf-8") if md.exists() else ev.texto_de(pdf)


def preguntar(texto_enunciado: str, texto_entregable: str, criterios: dict[str, str], traza, **meta) -> dict:
    lista = "\n".join(f"- {i}: {d}" for i, d in criterios.items())
    sistema = ("Eres un profesor que califica la entrega de un estudiante con la rúbrica del enunciado. "
               "No puedes ejecutar código: juzga leyendo. Para las cifras, decide si el valor reportado es "
               "correcto según lo que pide el enunciado; si no puedes saberlo con seguridad, di lo que creas más "
               "probable y explícalo en el motivo. Responde SOLO con JSON.")
    usuario = (f"## Enunciado\n{texto_enunciado}\n\n## Entrega del estudiante\n{texto_entregable[:MAX_CAR]}\n\n"
               f"## Criterios\n{lista}\n\n"
               'Responde con este JSON: {"criterios": [{"id": "...", "cumple": true o false, "motivo": "una frase"}], '
               '"resuelve_la_tarea": true o false, "nota": entero de 1 a 5, "comentario": "dos frases"}')
    cuerpo = {"model": MODELO, "stream": False, "format": "json",
              "options": {"temperature": 0, "seed": 0, "num_ctx": 32768},
              "messages": [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}]}
    t0, error, datos = time.perf_counter(), None, {}
    try:
        req = urllib.request.Request(f"http://{HOST}:{PUERTO_EMB}/api/chat", json.dumps(cuerpo).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            datos = json.loads(r.read())
        respuesta = json.loads(datos["message"]["content"])
    except Exception as err:  # la traza lo registra y la tarea queda sin juicio, no se inventa uno
        error, respuesta = f"{type(err).__name__}: {err}", {}
    traza.write(json.dumps({"hora": datetime.now(timezone.utc).isoformat(timespec="seconds"), "agente": "juez",
                            "modelo": datos.get("model", MODELO), "tokens_entrada": datos.get("prompt_eval_count"),
                            "tokens_salida": datos.get("eval_count"), "latencia_s": round(time.perf_counter() - t0, 2),
                            "error": error, **meta}, ensure_ascii=False) + "\n")
    traza.flush()
    return respuesta


def calificar() -> list[dict]:
    golden = json.loads((AQUI / "golden_tareas.json").read_text(encoding="utf-8"))
    filas = []
    SALIDA.mkdir(parents=True, exist_ok=True)
    with (SALIDA / "juez_traza.jsonl").open("w", encoding="utf-8") as traza:
        for nombre, (carpeta, csv_checks) in CONJUNTOS.items():
            if not csv_checks.exists():
                print(f"· {nombre}: sin CSV del evaluador, se salta")
                continue
            checks = resultados_checks(csv_checks)
            for t in golden["tareas"]:
                salida = carpeta / f"tarea-{t['id']}"
                principal = ev.entregable(salida, t["entregable"]) if salida.exists() else None
                if principal is None:
                    continue
                crit = {c["id"]: criterio(c) for c in t["checks"] if c["tipo"] in JUZGABLES}
                r = preguntar(enunciado(t), ev.texto_de(principal), crit, traza, conjunto=nombre, tarea=t["id"])
                juicio = {j.get("id"): j for j in r.get("criterios", []) if isinstance(j, dict)}
                for c in t["checks"]:
                    if c["tipo"] not in JUZGABLES:
                        continue
                    j = juicio.get(c["id"], {})
                    filas.append({"conjunto": nombre, "tarea": t["id"], "check": c["id"], "tipo": c["tipo"],
                                  "ok_check": checks.get((t["id"], c["id"]), ""),
                                  "ok_juez": "" if "cumple" not in j else int(bool(j["cumple"])),
                                  "motivo_juez": str(j.get("motivo", ""))[:300],
                                  "nota_juez": r.get("nota", ""), "resuelve_juez": r.get("resuelve_la_tarea", "")})
                print(f"· {nombre} {t['id']}: nota {r.get('nota')} · {len(juicio)}/{len(crit)} criterios juzgados")
    with (SALIDA / "juez_resultados.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader(); w.writerows(filas)
    return filas


def comparar(filas: list[dict]) -> dict:
    from sklearn.metrics import cohen_kappa_score
    pares = [(int(f["ok_check"]), int(f["ok_juez"])) for f in filas if f["ok_check"] != "" and f["ok_juez"] != ""]

    def resumen(ps):
        if not ps:
            return {"n": 0}
        a, b = zip(*ps)
        return {"n": len(ps), "acuerdo": round(sum(x == y for x, y in ps) / len(ps), 4),
                "kappa": round(float(cohen_kappa_score(a, b)), 4) if len(set(a)) > 1 or len(set(b)) > 1 else None,
                "check_ok_juez_ok": sum(x == 1 and y == 1 for x, y in ps), "check_ok_juez_no": sum(x == 1 and y == 0 for x, y in ps),
                "check_no_juez_ok": sum(x == 0 and y == 1 for x, y in ps), "check_no_juez_no": sum(x == 0 and y == 0 for x, y in ps)}

    cifra = {"cifra", "cifra_presente"}
    sin_juicio = sum(f["ok_juez"] == "" for f in filas)
    res = {"modelo_juez": MODELO, "fecha": datetime.now(timezone.utc).date().isoformat(),
           "comprobaciones": len(filas), "sin_juicio": sin_juicio,
           "todas": resumen(pares),
           "cifras": resumen([(int(f["ok_check"]), int(f["ok_juez"])) for f in filas
                              if f["tipo"] in cifra and f["ok_check"] != "" and f["ok_juez"] != ""]),
           "forma": resumen([(int(f["ok_check"]), int(f["ok_juez"])) for f in filas
                             if f["tipo"] not in cifra and f["ok_check"] != "" and f["ok_juez"] != ""]),
           "por_conjunto": {c: resumen([(int(f["ok_check"]), int(f["ok_juez"])) for f in filas if f["conjunto"] == c
                                        and f["ok_check"] != "" and f["ok_juez"] != ""])
                            for c in dict.fromkeys(f["conjunto"] for f in filas)}}
    (SALIDA / "juez_resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-comparar", action="store_true")
    args = ap.parse_args()
    filas = (list(csv.DictReader((SALIDA / "juez_resultados.csv").open(encoding="utf-8")))
             if args.solo_comparar else calificar())
    print(json.dumps(comparar(filas), indent=2, ensure_ascii=False))
