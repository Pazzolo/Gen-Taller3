"""El diagrama del orquestador, sacado del StateGraph compilado de `solver.py` (no dibujado a mano).

    ../../../solver-v2/.venv/bin/python diagrama_orquestador.py

Escribe orquestador.mmd (Mermaid, tal cual lo da LangGraph) y orquestador.png. Las aristas salen de
`get_graph()`; solo las posiciones de los nodos están puestas aquí. Línea continua: arista fija;
punteada: condicional (la decide una función `r_*` de `solver.py`).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[1]))
import solver  # noqa: E402

g = solver.Solver()._grafo().get_graph()
(AQUI / "orquestador.mmd").write_text(g.draw_mermaid(), encoding="utf-8")

POS = {
    "__start__": (0, 11), "leer": (0, 10), "indexar": (0, 9), "planificar": (0, 8), "validar": (0, 7),
    "elegir": (0, 6), "investigar": (3.4, 5.4), "programar": (2.6, 4.4), "revisar_codigo": (2.6, 3.4),
    "aprobar_red": (5.0, 3.4), "ejecutar": (2.6, 2.4), "revisar": (2.6, 1.4), "fallida": (5.0, 1.4),
    "redactar": (-2.6, 4.0), "comprobar": (-2.6, 2.8), "publicar": (-2.6, 1.4), "__end__": (-2.6, 0.4),
}
COLOR = {"investigar": "#dbeafe", "programar": "#dbeafe", "revisar_codigo": "#fde68a", "aprobar_red": "#fde68a",
         "ejecutar": "#bbf7d0", "revisar": "#fde68a", "fallida": "#fecaca", "validar": "#fde68a",
         "comprobar": "#fde68a"}

fig, ax = plt.subplots(figsize=(9, 9.5))
ax.set_axis_off()
for e in g.edges:
    a, b = e.source, e.target
    (x1, y1), (x2, y2) = POS[a], POS[b]
    if a == b:   # el bucle de programar sobre sí mismo (respuesta vacía del LLM: otro intento)
        ax.annotate("", xy=(x1 + 0.75, y1 + 0.12), xytext=(x1 + 0.75, y1 - 0.12),
                    arrowprops=dict(arrowstyle="->", color="#555", ls="--", connectionstyle="arc3,rad=-1.6"))
        continue
    atras = y2 > y1 or (a, b) in {("fallida", "elegir"), ("revisar", "elegir")}
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.0, ls="--" if e.conditional else "-",
                                shrinkA=16, shrinkB=22, connectionstyle=f"arc3,rad={0.28 if atras else 0.06}"))
for n, (x, y) in POS.items():
    etiqueta = {"__start__": "inicio", "__end__": "fin"}.get(n, n)
    ax.text(x, y, etiqueta, ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", fc=COLOR.get(n, "#f2f0ff"), ec="#444", lw=0.8))
ax.text(3.4, 6.15, "por cada subtarea de cálculo", ha="center", fontsize=8.5, style="italic", color="#333")
ax.text(-2.6, 4.75, "sin subtareas o\nsin presupuesto", ha="center", fontsize=8.5, style="italic", color="#333")
ax.text(-3.6, 10.7, "── fija   - - condicional\namarillo: lo decide código   verde: sandbox (sin LLM)",
        fontsize=8, va="top", color="#333")
ax.set_xlim(-4, 6.2); ax.set_ylim(0, 11.5)
fig.tight_layout()
fig.savefig(AQUI / "orquestador.png", dpi=170)
print(f"{len(g.nodes)} nodos, {len(g.edges)} aristas → orquestador.png, orquestador.mmd")
