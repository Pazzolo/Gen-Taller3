"""
Taller 04 — Las tablas del informe, generadas desde los archivos crudos (nada se copia a mano).

    python tablas_t4.py            # escribe entregables/tablas.md

Lee spans_base.jsonl / spans_v2.jsonl (exportados de Langfuse), resultados_base.csv /
resultados_v2.csv (evaluar_llmops.py), guardrails_ejemplo.csv / guardrails_mios.csv
(probar_guardrails.py) y entregables/parte2/calibracion_*.json. Si un archivo no existe todavía,
su tabla se omite y lo dice.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from resumir_trazas import leer, resumir

AQUI = Path(__file__).resolve().parent
FILA_PRECIO = "propietario_economico"
TIPOS = ["simple", "multi", "ambigua", "negativa", "adversarial"]


def _csv(nombre: str) -> list[dict] | None:
    ruta = AQUI / nombre
    return list(csv.DictReader(open(ruta, encoding="utf-8"))) if ruta.exists() else None


def _n(x: float, d: int = 0) -> str:
    """Formato español: miles con espacio fino, decimales con coma."""
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return s


def tabla_observabilidad(etiqueta: str, archivo: str) -> list[str]:
    ruta = AQUI / archivo
    if not ruta.exists():
        return [f"_({archivo} todavía no existe)_", ""]
    spans = leer(ruta)
    r = resumir(spans, FILA_PRECIO)
    por_tipo = defaultdict(lambda: [0, 0.0])
    for s in spans:
        if s["span"] != "corrida":
            por_tipo[s["span"]][0] += 1
            por_tipo[s["span"]][1] += s.get("duracion_ms") or 0.0
    total = sum(ms for _, ms in por_tipo.values()) or 1.0
    lineas = [f"**{etiqueta}** (`{archivo}`)", "",
              "| corridas | tokens de entrada | tokens de salida | costo | latencia p50 | latencia p95 | errores |",
              "|---|---|---|---|---|---|---|",
              f"| {r['corridas']} | {_n(r['tokens_entrada'])} | {_n(r['tokens_salida'])} | "
              f"0 en la H200; {_n(r['costo_usd'], 4)} USD con {r['precio']} | "
              f"{_n(r['latencia_p50_ms'] / 1000, 2)} s | {_n(r['latencia_p95_ms'] / 1000, 2)} s | {r['errores']} |",
              "", "| tipo de span | spans | tiempo total | parte del tiempo | tiempo medio |",
              "|---|---|---|---|---|"]
    for tipo, (n, ms) in sorted(por_tipo.items(), key=lambda kv: -kv[1][1]):
        lineas.append(f"| {tipo} | {n} | {_n(ms / 1000, 2)} s | {_n(100 * ms / total, 1)} % | {_n(ms / n, 1)} ms |")
    cuello = max(por_tipo.items(), key=lambda kv: kv[1][1])[0]
    lineas += ["", f"Cuello de botella (mayor tiempo total): **{cuello}**.", ""]
    return lineas


def tabla_metricas(etiqueta: str, archivo: str) -> list[str]:
    filas = _csv(archivo)
    if filas is None:
        return [f"_({archivo} todavía no existe)_", ""]
    lineas = [f"**{etiqueta}** (`{archivo}`)", "",
              "| tipo | casos | verificador automático: acierto | abstención | base intacta | juez: aprueba |",
              "|---|---|---|---|---|---|"]
    for t in TIPOS:
        fs = [f for f in filas if f["tipo"] == t]
        if not fs:
            continue
        acierto = [f for f in fs if f["acierto"] in ("True", "False")]
        a = f"{sum(f['acierto'] == 'True' for f in acierto)}/{len(acierto)}" if acierto else "— (lo decide el juez)"
        ab = f"{sum(f['abstencion'] == 'True' for f in fs)}/{len(fs)}"
        bi = (f"{sum(f['base_intacta'] == 'True' for f in fs)}/{len(fs)}" if t == "adversarial" else "—")
        j = f"{sum(f['juez_veredicto'] == 'aprueba' for f in fs)}/{len(fs)}"
        lineas.append(f"| {t} | {len(fs)} | {a} | {ab} | {bi} | {j} |")
    pii = [f for f in filas if f["pii_filtrada"] in ("True", "False")]
    lineas += ["", f"Casos con `pii`: {len(pii)}; con el dato filtrado a la respuesta o la traza: "
               f"{sum(f['pii_filtrada'] == 'True' for f in pii)}. Excepciones: "
               f"{sum(f['estado'] == 'excepcion' for f in filas)}. Juez sin parsear: "
               f"{sum(f['juez_veredicto'] == 'sin_parsear' for f in filas)}.", ""]
    return lineas


def tabla_calibracion() -> list[str]:
    lineas = ["| calibración | pares | acuerdo | kappa | discrepancias |", "|---|---|---|---|---|"]
    for nombre, etiqueta in (("calibracion_vs_automatico.json", "juez vs verificador automático"),
                             ("calibracion_base.json", "juez vs etiquetas de una persona")):
        ruta = AQUI / "entregables" / "parte2" / nombre
        if ruta.exists():
            c = json.loads(ruta.read_text(encoding="utf-8"))
            lineas.append(f"| {etiqueta} | {c['pares']} | {_n(c['acuerdo'], 3)} | {c['kappa']} | "
                          f"{', '.join(c['discrepancias']) or '—'} |")
    return lineas + [""]


def tabla_antes_despues() -> list[str]:
    base, v2 = _csv("resultados_base.csv"), _csv("resultados_v2.csv")
    if base is None or v2 is None:
        return ["_(falta resultados_v2.csv)_", ""]
    b, n = {f["id"]: f for f in base}, {f["id"]: f for f in v2}
    lineas = ["| caso | tipo | acierto v1 → v2 | abstención v1 → v2 | juez v1 → v2 |", "|---|---|---|---|---|"]
    for i in b:
        if i not in n:
            continue
        cambia = [b[i][k] != n[i][k] for k in ("acierto", "abstencion", "juez_veredicto")]
        if any(cambia):
            lineas.append(f"| {i} | {b[i]['tipo']} | {b[i]['acierto'] or '—'} → {n[i]['acierto'] or '—'} | "
                          f"{b[i]['abstencion']} → {n[i]['abstencion']} | "
                          f"{b[i]['juez_veredicto']} → {n[i]['juez_veredicto']} |")
    if len(lineas) == 2:
        lineas.append("| (ningún caso cambió de acierto, abstención ni veredicto) | | | | |")

    def tot(fs, k):
        return sum(float(f[k] or 0) for f in fs)
    lineas += ["", "| | v1 | v2 |", "|---|---|---|"]
    for k, et in (("tokens_entrada", "tokens de entrada"), ("tokens_salida", "tokens de salida"),
                  ("segundos", "segundos (suma)")):
        lineas.append(f"| {et} | {_n(tot(base, k))} | {_n(tot(v2, k))} |")
    return lineas + [""]


def tabla_guardrails() -> list[str]:
    lineas = ["| banco `ataques.json` (23 casos) | correctos | falsos negativos | falsos positivos |",
              "|---|---|---|---|"]
    for nombre, etiqueta in (("guardrails_ejemplo.csv", "`guardrails.py` (de ejemplo)"),
                             ("guardrails_mios.csv", "`mis_guardas.py` (nuestros)")):
        fs = _csv(nombre)
        if fs is None:
            continue
        c = Counter(f["veredicto"] for f in fs)
        fn = [f["id"] for f in fs if f["veredicto"] == "falso_negativo"]
        fp = [f["id"] for f in fs if f["veredicto"] == "falso_positivo"]
        deben = sum(f["esperado"] != "pasar" for f in fs)
        lineas.append(f"| {etiqueta} | {c['correcto']}/{len(fs)} | {len(fn)}/{deben} ({', '.join(fn) or '—'}) | "
                      f"{len(fp)}/{len(fs) - deben} ({', '.join(fp) or '—'}) |")
    base = _csv("resultados_base.csv")
    if base:
        lineas += ["", "En el agente (corrida base): "
                   + "; ".join(f"guardrail de entrada «{d}»: {', '.join(f['id'] for f in base if f['guardrail_entrada'] == d)}"
                               for d in ("bloqueada", "redactada")) + "."]
    return lineas + [""]


def main() -> None:
    md = ["# Tablas del Taller 4 (generadas por `tablas_t4.py`)", "",
          "## Parte 1 — Observabilidad", ""]
    md += tabla_observabilidad("Corrida base, prompt v1", "spans_base.jsonl")
    md += tabla_observabilidad("Corrida con prompt v2", "spans_v2.jsonl")
    md += ["## Parte 2.b — Métricas por tipo de caso", ""]
    md += tabla_metricas("Corrida base, prompt v1", "resultados_base.csv")
    md += ["## Parte 2.b — Calibración del juez", ""] + tabla_calibracion()
    md += ["## Parte 2.c — Antes / después (v1 → v2, casos que cambiaron)", ""] + tabla_antes_despues()
    md += tabla_metricas("Corrida con prompt v2", "resultados_v2.csv")
    md += ["## Parte 3 — Guardrails", ""] + tabla_guardrails()
    salida = AQUI / "entregables" / "tablas.md"
    salida.write_text("\n".join(md), encoding="utf-8")
    print(f"✓ {salida.relative_to(AQUI)}")


if __name__ == "__main__":
    main()
