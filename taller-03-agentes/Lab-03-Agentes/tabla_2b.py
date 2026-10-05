"""
Taller 03 — La tabla de la 2.b, calculada desde los CSV que escribe evaluar.py (no a mano).

    python tabla_2b.py                       # resultados_base.csv vs resultados_mi_agente*.csv
    python tabla_2b.py a.csv b.csv ...       # cualquier conjunto de CSV
    python tabla_2b.py --salida X.md a.csv … # escribe la tabla en X.md (Parte 4)

Reproduce las métricas de `evaluar.resumen()` leyendo los CSV, y escribe la tabla en markdown
en entregables/parte2/tabla_2b.md.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RESPONDIBLES = {"simple", "multi"}


def _bool(x: str) -> bool:
    return x == "True"


def metricas(ruta: Path) -> dict:
    filas = list(csv.DictReader(ruta.open(encoding="utf-8")))
    resp = [f for f in filas if f["tipo"] in RESPONDIBLES]
    neg = [f for f in filas if f["tipo"] not in RESPONDIBLES]
    adv = [f for f in neg if f["tipo"] == "adversarial"]

    def tasa(xs, clave):
        return f"{sum(_bool(x[clave]) for x in xs)}/{len(xs)} = {sum(_bool(x[clave]) for x in xs) / len(xs):.3f}" if xs else "—"

    pasos = [int(f["pasos"]) for f in filas if f["pasos"].isdigit()]
    tokens = [int(f["tokens_entrada"]) for f in filas if f["tokens_entrada"].isdigit()]
    tokens_sal = [int(f["tokens_salida"]) for f in filas if f["tokens_salida"].isdigit()]
    return {
        "preguntas": len(filas),
        "acierto global": tasa(filas, "acierto"),
        "exactitud (respondibles)": tasa(resp, "acierto"),
        "abstención correcta (neg. + adv.)": tasa(neg, "abstencion"),
        "abstención indebida (respondibles)": tasa(resp, "abstencion"),
        "base intacta (adversariales)": tasa(adv, "base_intacta"),
        "pasos promedio": f"{sum(pasos) / len(pasos):.2f}" if pasos else "—",
        "errores de herramienta": sum(int(f["errores_herramienta"] or 0) for f in filas if f["estado"] != "excepcion"),
        "excepciones": sum(1 for f in filas if f["estado"] == "excepcion"),
        "tokens de entrada totales": f"{sum(tokens):,}".replace(",", " ") if tokens else "la traza no trae tokens",
        "tokens de salida totales": f"{sum(tokens_sal):,}".replace(",", " ") if tokens_sal else "la traza no trae tokens",
        "segundos totales": f"{sum(float(f['segundos']) for f in filas):.1f}",
        "modelo": next((f["modelo"] for f in filas if f["modelo"]), "(no lo registra)"),
    }


def tabla(rutas: list[Path]) -> str:
    cols = [metricas(r) for r in rutas]
    lineas = ["| Métrica | " + " | ".join(f"`{r.name}`" for r in rutas) + " |",
              "|---|" + "---|" * len(rutas)]
    for clave in cols[0]:
        lineas.append(f"| {clave} | " + " | ".join(str(c[clave]) for c in cols) + " |")
    return "\n".join(lineas)


if __name__ == "__main__":
    args = sys.argv[1:]
    destino = Path("entregables/parte2/tabla_2b.md")
    if args[:1] == ["--salida"]:
        destino, args = Path(args[1]), args[2:]
    rutas = [Path(a) for a in args] or \
        [Path("resultados_base.csv"), *sorted(Path(".").glob("resultados_mi_agente*.csv"))]
    md = tabla(rutas)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("<!-- generado por tabla_2b.py desde los CSV de evaluar.py -->\n" + md + "\n",
                       encoding="utf-8")
    print(md)
