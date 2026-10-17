"""Tabla de repeticiones de la Parte 2.c: 5 casos × 3 corridas × prompt v1 y v2 (más la corrida de 32).

    python entregables/parte2c/tabla_repeticiones.py   # desde Lab-04-LLMOps/
"""
import csv, re
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent / "repeticiones"
FRASE = "No puedo responder con los datos disponibles"
LECTURA = re.compile(r"(medida|interpret|lectura|n[uú]mero de ventas|por (?:n[uú]mero de )?ventas|si se mide|si mid)", re.I)
R = defaultdict(list)
for v, completa in (("v1", "resultados_base.csv"), ("v2", "resultados_v2.csv")):
    fuentes = [Path(completa)] + sorted(D.glob(f"rep_{v}_r*.csv"))
    for ruta in fuentes:
        for f in csv.DictReader(open(ruta, encoding="utf-8")):
            if f["id"] in ("A04", "A05", "AM03", "G04", "G07"):
                R[(f["id"], v)].append(f)
print("| caso | prompt | corridas | acierto automático | frase al inicio / declara la lectura | juez aprueba |")
print("|---|---|---|---|---|---|")
for cid in ("A04", "A05", "AM03", "G04", "G07"):
    for v in ("v1", "v2"):
        fs = R[(cid, v)]
        if not fs:
            continue
        ac = [f["acierto"] for f in fs if f["acierto"] in ("True", "False")]
        acierto = f"{ac.count('True')}/{len(ac)}" if ac else "— (ambigua)"
        if cid in ("A04", "A05"):
            extra = f"frase al inicio {sum(f['respuesta'].strip().startswith(FRASE) for f in fs)}/{len(fs)}"
        elif cid == "AM03":
            extra = f"declara la lectura {sum(bool(LECTURA.search(f['respuesta'])) for f in fs)}/{len(fs)}"
        else:
            extra = "—"
        print(f"| {cid} | {v} | {len(fs)} | {acierto} | {extra} | "
              f"{sum(f['juez_veredicto'] == 'aprueba' for f in fs)}/{len(fs)} |")
