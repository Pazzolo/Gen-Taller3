"""Redactor: el entregable en el formato pedido, y la comprobación de cada cifra. REGLA 5.

El LLM escribe; el código comprueba antes de publicar:
    archivo      existe con el nombre que pide el enunciado
    secciones    están todas, en el orden pedido (misma lógica que `evaluar_solver.posicion_seccion`)
    extensión    palabras (sin bloques de código) o páginas dentro del límite
    ejecutado    un notebook tiene todas las celdas ejecutadas y ninguna con error
    procedencia  REGLA 5: toda cifra con dos o más decimales salió de una ejecución (resultados.json,
                 stdout, salidas del notebook) o está en el enunciado. La lógica es la de la
                 comprobación `procedencia` de `evaluar_solver.py`: el solver se revisa con el mismo
                 criterio con que lo van a calificar.
Si algo falla, los problemas vuelven al redactor (el orquestador cuenta las rondas).

Formatos:
    md     el Markdown que escribe el LLM; las figuras se copian a `figuras/` y el código a `codigo/`
    pdf    ese Markdown impreso con PyMuPDF (`fitz.Story`, como `generar_enunciados.py`); si se pasa
           de páginas, el código achica primero la letra y las figuras, y solo después pide recortar
    ipynb  un notebook armado con los scripts aprobados (una celda por subtarea, bajo el título de su
           parte) y las respuestas del LLM en Markdown, ejecutado DENTRO del sandbox (kernel local)
"""
from __future__ import annotations

import json
import re
import shutil
import unicodedata
from pathlib import Path

import pymupdf
from markdown_it import MarkdownIt

from sandbox import ejecutar

NUM = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)(\s?%)?(?![\w])")    # el de evaluar_solver.py
EXCLUIR = re.compile(r"traza|grafo|comunidades|plan|resumen|cache|enunciado|\.mpl|tmp")


# ── Procedencia (REGLA 5) ──────────────────────────────────────────────────────────────────
def numeros(texto: str) -> list[tuple[float, int]]:
    salida = []
    for m in NUM.finditer(texto):
        crudo, pct = m.group(1).replace(",", "."), m.group(2)
        try:
            v = float(crudo)
        except ValueError:
            continue
        dec = len(crudo.split(".")[1]) if "." in crudo else 0
        if pct:
            salida.append((v / 100, dec + 2))
        salida.append((v, dec))
    return salida


def sin_codigo(texto: str) -> str:
    return re.sub(r"```.*?```", "", texto, flags=re.S)


def numeros_de_artefactos(salida: Path, excluir: set[Path]) -> list[tuple[float, int]]:
    nums: list[tuple[float, int]] = []
    for ext in ("json", "csv", "txt", "log", "out"):
        for f in salida.rglob(f"*.{ext}"):
            if f in excluir or f.stat().st_size > 5_000_000 or EXCLUIR.search(str(f.relative_to(salida))):
                continue
            nums += numeros(f.read_text(encoding="utf-8", errors="replace"))
    for nb in salida.rglob("*.ipynb"):
        if EXCLUIR.search(str(nb.relative_to(salida))):
            continue
        for c in json.loads(nb.read_text(encoding="utf-8")).get("cells", []):
            for o in c.get("outputs", []):
                nums += numeros("".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", "")))
    return nums


def texto_entregable(ruta: Path, con_salidas: bool = False) -> str:
    if ruta.suffix == ".pdf":
        return unicodedata.normalize("NFKC", "\n".join(p.get_text() for p in pymupdf.open(ruta)))
    if ruta.suffix == ".ipynb":
        partes = []
        for c in json.loads(ruta.read_text(encoding="utf-8")).get("cells", []):
            if c["cell_type"] == "markdown":
                partes.append("".join(c.get("source", "")))
            elif con_salidas:
                for o in c.get("outputs", []):
                    partes.append("".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", "")))
        return "\n".join(partes)
    return ruta.read_text(encoding="utf-8")


def sin_origen(entregable: Path, salida: Path, texto_enunciado: str) -> list[str]:
    """Las cifras con ≥ 2 decimales del entregable que no salieron de ninguna ejecución."""
    dados = {v for v, _ in numeros(texto_enunciado)}
    propias = [(v, d) for v, d in numeros(sin_codigo(texto_entregable(entregable))) if d >= 2 and v not in dados]
    medidos = numeros_de_artefactos(salida, {entregable})
    huerfanas = [v for v, d in propias if not any(abs(a - v) <= 0.5 * 10 ** -d + 1e-9 for a, _ in medidos)]
    return [f"{v:g}" for v in dict.fromkeys(huerfanas)]


# ── Comprobaciones de forma ────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip()


def posicion_seccion(texto: str, seccion: str) -> int:
    objetivo, pos = _norm(seccion), 0
    for linea in texto.splitlines(keepends=True):
        h = re.sub(r"^\d+ ", "", _norm(linea.lstrip("# ")))
        if h and len(h) <= len(objetivo) + 40 and (h == objetivo or h.startswith(objetivo + " ")):
            return pos
        pos += len(linea)
    return -1


def comprobar(entregable: Path, spec: dict, salida: Path, texto_enunciado: str) -> list[str]:
    problemas: list[str] = []
    if not entregable.exists():
        return [f"archivo: no existe {entregable.name}"]
    texto = texto_entregable(entregable)
    pos = [(s, posicion_seccion(texto, s)) for s in spec.get("secciones") or []]
    faltan = [s for s, p in pos if p < 0]
    if faltan:
        problemas.append(f"secciones: faltan {faltan} (cada una tiene que ser un encabezado con ese nombre exacto)")
    elif [p for _, p in pos] != sorted(p for _, p in pos):
        problemas.append(f"secciones: fuera de orden; el orden pedido es {spec['secciones']}")
    if spec.get("max_palabras"):
        n = len(sin_codigo(texto).split())
        if n > spec["max_palabras"]:
            problemas.append(f"extensión: {n} palabras; el máximo es {spec['max_palabras']} (recorta {n - spec['max_palabras']} o más)")
    if spec.get("max_paginas") and entregable.suffix == ".pdf":
        n = pymupdf.open(entregable).page_count
        if n > spec["max_paginas"]:
            problemas.append(f"extensión: {n} páginas; el máximo es {spec['max_paginas']}")
    if entregable.suffix == ".ipynb":
        celdas = [c for c in json.loads(entregable.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
        sin = sum(c.get("execution_count") is None for c in celdas)
        err = sum(o.get("output_type") == "error" for c in celdas for o in c.get("outputs", []))
        if not celdas or sin or err:
            problemas.append(f"ejecutado: {len(celdas)} celdas de código, {sin} sin ejecutar, {err} con error")
    if huerfanas := sin_origen(entregable, salida, texto_enunciado):
        problemas.append(f"procedencia (regla 5): estas cifras no salieron de ninguna ejecución ni del enunciado: "
                         f"{huerfanas[:15]}. Quítalas o cámbialas por las cifras exactas de los resultados.")
    return problemas


# ── El LLM escribe ─────────────────────────────────────────────────────────────────────────
SISTEMA_REDACTOR = """Eres el redactor de un solver que resuelve tareas de maestría. Escribes el entregable en Markdown,
en español, con voz de estudiante que razona (frases concretas, sin adornos).
Reglas que el código va a comprobar antes de publicar:
- Usa como encabezados (## ) EXACTAMENTE las secciones pedidas, en el orden pedido.
- Toda cifra con decimales sale de RESULTADOS (cópiala; puedes redondear a 4 decimales o expresarla en %).
  No inventes ni estimes ninguna cifra. Si algo no se calculó, dilo en lugar de poner un número.
- Respeta el límite de palabras o de páginas.
- Las figuras ya existen: inclúyelas con ![descripción](nombre.png) usando el nombre exacto que se te da.
- Las tablas en Markdown. No pegues el código: el código va aparte."""


def escribir_markdown(llm, doc_texto: str, spec: dict, resultados: dict, redaccion: list[dict],
                      figuras: list[str], problemas: list[str] | None = None, previo: str | None = None,
                      para_notebook: bool = False) -> str:
    pedido = {k: spec.get(k) for k in ("formato", "archivo", "secciones", "max_palabras", "max_paginas")}
    if para_notebook:
        pedido["instruccion"] = ("Escribe SOLO las respuestas en prosa de las partes conceptuales, una sección ## por "
                                 "parte, con el título exacto de la parte. Las cifras y tablas ya las muestran las celdas de código.")
    mensajes = [{"role": "system", "content": SISTEMA_REDACTOR},
                {"role": "user", "content": f"ENUNCIADO:\n{doc_texto[:9000]}\n\nENTREGABLE PEDIDO:\n{json.dumps(pedido, ensure_ascii=False)}\n\n"
                                            f"RESULTADOS (por subtarea: resultados.json y la salida impresa):\n{json.dumps(resultados, ensure_ascii=False)[:14000]}\n\n"
                                            f"PARTES DE REDACCIÓN:\n{json.dumps(redaccion, ensure_ascii=False)}\n\nFIGURAS DISPONIBLES: {figuras or 'ninguna'}"}]
    if problemas:
        mensajes += [{"role": "assistant", "content": previo or ""},
                     {"role": "user", "content": "La comprobación automática encontró estos problemas. Corrígelos y devuelve el "
                                                 "entregable completo:\n- " + "\n- ".join(problemas)}]
    texto = llm.chat("redactor", mensajes)
    m = re.fullmatch(r"\s*```(?:markdown|md)?\s*\n(.*)```\s*", texto, re.S)
    return (m.group(1) if m else texto).strip() + "\n"


# ── De Markdown a cada formato ─────────────────────────────────────────────────────────────
def md_a_pdf(md: str, destino: Path, carpeta_imagenes: Path, escala: float = 1.0) -> int:
    css = f"""body {{ font-family: sans-serif; font-size: {10 * escala:.1f}pt; line-height: 1.3; }}
h1 {{ font-size: {15 * escala:.1f}pt; }} h2 {{ font-size: {12 * escala:.1f}pt; margin-top: {8 * escala:.0f}pt; }}
table {{ border-collapse: collapse; }} th, td {{ border: 0.5pt solid #888; padding: 1pt 3pt; font-size: {9 * escala:.1f}pt; }}
img {{ width: {300 * escala:.0f}px; }}"""
    html = MarkdownIt("commonmark").enable("table").render(md)
    historia = pymupdf.Story(html=html, user_css=css, archive=str(carpeta_imagenes))
    escritor = pymupdf.DocumentWriter(str(destino))
    a4 = pymupdf.paper_rect("a4")
    caja, mas, paginas = a4 + (42, 42, -42, -42), True, 0
    while mas:
        dispositivo = escritor.begin_page(a4)
        mas, _ = historia.place(caja)
        historia.draw(dispositivo)
        escritor.end_page()
        paginas += 1
    escritor.close()
    return paginas


def armar_notebook(destino: Path, doc, plan: dict, scripts: dict[str, str], prosa_md: str) -> None:
    """Un notebook con una sección por parte: su título, los scripts aprobados que la cubren y la prosa."""
    import nbformat
    prosa = {}
    for bloque in re.split(r"(?m)^## ", prosa_md):
        if bloque.strip():
            titulo, _, cuerpo = bloque.partition("\n")
            prosa[_norm(titulo)] = cuerpo.strip()
    celdas = [nbformat.v4.new_markdown_cell(f"# {doc.secciones[0].titulo}")]
    for s in doc.secciones:
        if s.nivel < 2:
            continue
        subs = [t for t in plan["subtareas"] if s.id in (t.get("secciones") or [])]
        if not subs and not any(_norm(s.titulo).startswith(k) or k.startswith(_norm(s.titulo)) for k in prosa):
            continue
        celdas.append(nbformat.v4.new_markdown_cell(f"## {s.titulo}"))
        for t in subs:
            if t["id"] in scripts:
                celdas.append(nbformat.v4.new_code_cell(f"# Subtarea {t['id']}: {t['descripcion'][:150]}\n{scripts[t['id']]}"))
        texto = next((v for k, v in prosa.items() if k and (_norm(s.titulo).startswith(k) or k.startswith(_norm(s.titulo)))), "")
        if texto:
            celdas.append(nbformat.v4.new_markdown_cell(texto))
    nb = nbformat.v4.new_notebook()
    nb.cells = celdas
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.write(nb, destino)


EJECUTAR_NB = """import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
nb = nbformat.read({nombre!r}, as_version=4)
try:
    ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {{"metadata": {{"path": "."}}}})
finally:
    nbformat.write(nb, {nombre!r})
print("notebook ejecutado")
"""


def ejecutar_notebook(ruta: Path) -> tuple[bool, str]:
    """Ejecuta el notebook dentro del sandbox (sin entorno, sin red salvo localhost para el kernel)."""
    e = ejecutar(EJECUTAR_NB.format(nombre=ruta.name), ruta.parent, timeout=600, nombre="_ejecutar_notebook.py",
                 kernel_local=True)
    for basura in ("_ejecutar_notebook.py", "stdout.txt", "stderr.txt"):
        (ruta.parent / basura).unlink(missing_ok=True)
    shutil.rmtree(ruta.parent / "tmp", ignore_errors=True)
    shutil.rmtree(ruta.parent / ".mpl", ignore_errors=True)
    return e.codigo_salida == 0, e.stderr[-1500:]
