"""Las tablas de la 2.b, desde los CSV del evaluador y las trazas de las corridas (no a mano).

    ../solver-v2/.venv/bin/python tablas_2b.py completo=resultados/completo sin_grafo=resultados/sin_grafo

Cada argumento es `nombre=carpeta`, con `resultados_solver.csv`, `resumen_solver.csv` y `corridas/`
dentro. Escribe entregables/parte2/tablas_2b.md con:
    1. por tarea y versión: comprobaciones aprobadas, procedencia, subtareas fallidas, intentos de
       código, tokens de entrada y salida, duración (de los dos CSV del evaluador)
    2. tokens de entrada y salida por agente (de corridas/tarea-*/traza.jsonl, que el CSV no trae)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def _filas(ruta: Path) -> list[dict]:
    return list(csv.DictReader(ruta.open(encoding="utf-8")))


def leer_version(carpeta: Path) -> dict:
    res = _filas(carpeta / "resultados_solver.csv")
    resumen = _filas(carpeta / "resumen_solver.csv")
    por_tarea = {}
    for r in resumen:
        t = r["tarea"]
        checks = [x for x in res if x["tarea"] == t]
        proc = next((x["detalle"] for x in checks if x["tipo"] == "procedencia"), "—")
        traza = carpeta / "corridas" / f"tarea-{t}" / "traza.jsonl"
        fallidas, agentes = 0, {}
        if traza.exists():
            for l in traza.open(encoding="utf-8"):
                ev = json.loads(l)
                if ev["tipo"] == "fin":
                    fallidas = sum(s["status"] != "aprobada" for s in ev.get("subtareas", []) if s["tipo"] == "calculo")
                if ev["tipo"] == "llm":
                    a = agentes.setdefault(ev["agente"], [0, 0])
                    a[0] += ev.get("tokens_entrada", 0)
                    a[1] += ev.get("tokens_salida", 0)
        por_tarea[t] = {"aprobadas": f"{r['aprobadas']}/{r['total']}", "fallos": [x["check"] for x in checks if x["ok"] == "0"],
                        "procedencia": proc, "status": r["status"], "subtareas_fallidas": fallidas,
                        "intentos": r["intentos_codigo"], "subtareas": r["subtareas"],
                        "tokens": f"{int(r['tokens_entrada']):,} / {int(r['tokens_salida']):,}".replace(",", " "),
                        "duracion": f"{float(r['duracion_s']) / 60:.1f} min", "agentes": agentes}
    total = (sum(int(r["aprobadas"]) for r in resumen), sum(int(r["total"]) for r in resumen))
    return {"tareas": por_tarea, "total": total}


def main() -> int:
    versiones = {}
    for arg in sys.argv[1:]:
        nombre, _, carpeta = arg.partition("=")
        versiones[nombre] = leer_version(Path(carpeta))
    tareas = sorted({t for v in versiones.values() for t in v["tareas"]})
    md = ["<!-- generado por tablas_2b.py desde resultados_solver.csv, resumen_solver.csv y las trazas -->",
          "## Por tarea", "", "| Tarea | Versión | Aprobadas | Fallan | Procedencia | Status | Subtareas fallidas | Intentos de código | Tokens entrada / salida | Duración |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t in tareas:
        for nombre, v in versiones.items():
            x = v["tareas"].get(t)
            if x:
                md.append(f"| {t} | {nombre} | {x['aprobadas']} | {', '.join(x['fallos']) or '—'} | {x['procedencia']} | {x['status']} | "
                          f"{x['subtareas_fallidas']} de {x['subtareas']} | {x['intentos']} | {x['tokens']} | {x['duracion']} |")
    md += ["", "| Versión | Comprobaciones aprobadas |", "|---|---|"]
    md += [f"| {n} | {v['total'][0]}/{v['total'][1]} |" for n, v in versiones.items()]
    agentes = sorted({a for v in versiones.values() for x in v["tareas"].values() for a in x["agentes"]})
    md += ["", "## Tokens por agente (suma de todas las tareas; entrada / salida)", "",
           "| Agente | " + " | ".join(versiones) + " |", "|---|" + "---|" * len(versiones)]
    for a in agentes:
        celdas = []
        for v in versiones.values():
            e = sum(x["agentes"].get(a, [0, 0])[0] for x in v["tareas"].values())
            s = sum(x["agentes"].get(a, [0, 0])[1] for x in v["tareas"].values())
            celdas.append(f"{e:,} / {s:,}".replace(",", " "))
        md.append(f"| {a} | " + " | ".join(celdas) + " |")
    destino = AQUI / "entregables" / "parte2" / "tablas_2b.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
