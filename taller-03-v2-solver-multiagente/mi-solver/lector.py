"""Lector: PDF → texto por página, con las tablas como tablas, y las secciones por sus títulos.

Sin LLM. Qué revisa el código (enunciado, Parte 1):
- **texto pegado o partido**: las ligaduras («ﬁ») se normalizan con NFKC, los guiones de corte de
  línea se unen, y las tablas se leen con `find_tables` y se escriben en Markdown. Leídas como texto
  plano, una celda larga se parte en varias líneas y la tabla deja de ser una tabla.
- **PDF escaneado**: una página con imágenes y sin texto se marca. No hay OCR: el solver avisa en
  lugar de inventar un enunciado vacío.

Los títulos se detectan por tamaño de letra: todo renglón con letra mayor que la del cuerpo (la más
frecuente) es un título. Funciona con los PDF del kit (17 pt el título, 13 pt las secciones, 10,5 pt
el cuerpo) y con la mayoría de los PDF de un procesador de texto.
"""
from __future__ import annotations

import collections
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf


@dataclass
class Seccion:
    id: str            # «S0», «S1»…
    titulo: str
    nivel: int         # 1 = título del documento, 2 = sección
    pagina: int        # 1-based, donde empieza
    texto: str = ""

    def cita(self, documento: str) -> str:
        return f"[{documento} §{self.titulo}, p. {self.pagina}]"


@dataclass
class Documento:
    ruta: str
    paginas: list[str]
    secciones: list[Seccion]
    avisos: list[str] = field(default_factory=list)

    @property
    def nombre(self) -> str:
        return Path(self.ruta).stem

    @property
    def texto(self) -> str:
        return "\n\n".join(self.paginas)


def _limpio(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"(\w)-\n(\w)", r"\1\2", t)           # palabra partida por un guion de corte


def _dentro(bbox, cajas) -> bool:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return any(c.x0 <= cx <= c.x1 and c.y0 <= cy <= c.y1 for c in cajas)


def leer(ruta: str | Path) -> Documento:
    doc = pymupdf.open(ruta)
    avisos: list[str] = []
    # 1 · el tamaño del cuerpo: el más frecuente, contando caracteres
    tamanos: collections.Counter = collections.Counter()
    for pagina in doc:
        for b in pagina.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    tamanos[round(s["size"], 1)] += len(s["text"].strip())
    cuerpo = tamanos.most_common(1)[0][0] if tamanos else 10.0

    # 2 · por página: renglones en orden de lectura; las tablas, como Markdown en su lugar
    renglones: list[tuple[int, float, str, bool]] = []   # (página, tamaño, texto, es_título)
    paginas: list[str] = []
    for n, pagina in enumerate(doc, start=1):
        tablas = pagina.find_tables().tables
        cajas = [pymupdf.Rect(t.bbox) for t in tablas]
        piezas: list[tuple[float, str, float, bool]] = []    # (y, texto, tamaño, es_título)
        for t in tablas:
            piezas.append((t.bbox[1], _limpio(t.to_markdown()).replace("<br>", " "), cuerpo, False))
        for b in pagina.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                if _dentro(l["bbox"], cajas):
                    continue
                texto = _limpio("".join(s["text"] for s in l["spans"])).strip()
                if not texto:
                    continue
                tam = max(s["size"] for s in l["spans"])
                piezas.append((l["bbox"][1], texto, tam, tam > cuerpo + 1.0))
        piezas.sort(key=lambda p: p[0])
        if not piezas and pagina.get_images():
            avisos.append(f"página {n}: solo imágenes, sin texto (¿PDF escaneado?): no se puede leer sin OCR")
        paginas.append("\n".join(p[1] for p in piezas))
        renglones += [(n, p[2], p[1], p[3]) for p in piezas]

    # 3 · secciones: un título abre una sección; títulos seguidos del mismo tamaño se unen
    #     (un título largo que el PDF partió en dos renglones)
    secciones: list[Seccion] = []
    actual: Seccion | None = None
    tam_max = max((r[1] for r in renglones if r[3]), default=cuerpo)
    ultimo_titulo = False
    for pag, tam, texto, es_titulo in renglones:
        if es_titulo:
            if ultimo_titulo and actual is not None and not actual.texto:
                actual.titulo += " " + texto
            else:
                actual = Seccion(f"S{len(secciones)}", texto, 1 if tam >= tam_max - 0.5 else 2, pag)
                secciones.append(actual)
            ultimo_titulo = True
            continue
        ultimo_titulo = False
        if actual is None:
            actual = Seccion("S0", "Preámbulo", 1, pag)
            secciones.append(actual)
        actual.texto += ("\n" if actual.texto else "") + texto
    if not secciones:
        avisos.append("no se detectaron títulos: todo el texto queda en una sola sección")
        secciones = [Seccion("S0", "Documento", 1, 1, "\n".join(paginas))]
    sospechosas = [w for p in paginas for w in p.split() if len(w) > 40 and not w.startswith(("|", "http"))]
    if sospechosas:
        avisos.append(f"{len(sospechosas)} palabras de más de 40 letras (¿texto pegado?): {sospechosas[:3]}")
    return Documento(str(ruta), paginas, secciones, avisos)


if __name__ == "__main__":
    import sys
    d = leer(sys.argv[1])
    print(f"{d.nombre}: {len(d.paginas)} páginas, {len(d.secciones)} secciones, avisos={d.avisos}")
    for s in d.secciones:
        print(f"  {s.id} n{s.nivel} p{s.pagina} {s.titulo!r}: {len(s.texto)} car.")
