"""Indexador (GraphRAG): el grafo del enunciado y del material del curso. REGLA 2.

Dos capas, en este orden:

1. **Base fija, sin LLM.** Cada sección del enunciado es un nodo. Cada referencia literal a otra
   parte («Parte 1», «la pregunta anterior», «Sección 2») es una arista `depende_de` que saca un
   regex. Es la conexión que en la 0.b recupera los 3 hechos que el RAG plano pierde: una regla la
   encuentra siempre, gratis y sin alucinar (`solver-v2/parte0/b_rag_plano.py`).
2. **Entidades, con LLM.** Encima, el LLM extrae de cada sección y de cada fragmento de las notas del
   curso sus entidades (datasets, métodos, métricas, conceptos). Una entidad con el mismo nombre
   normalizado en el enunciado y en las notas une los dos lados (`menciona`). Así el investigador
   puede traer la definición vista en clase.

Las notas del curso se indexan una vez, en paralelo, y quedan en `cache/curso.json` (el enunciado
avisa que la primera indexación toma unos 6 minutos). Los embeddings `bge-m3` sirven para la versión
recortada «sin grafo» de la 2.b (top-k por similitud, como la 0.b) y para desempatar.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from lector import Documento, Seccion
from llm import LLM

AQUI = Path(__file__).resolve().parent
NOTAS = AQUI.parent / "notas-teoricas"
CACHE = AQUI / "cache"
TIPOS = ("dataset", "metodo", "metrica", "concepto")

# ── REGLA 2, capa 1: las referencias entre secciones salen de reglas ────────────────────────
_NUMERADA = re.compile(r"\b(Parte|Secci[oó]n|Pregunta|Ejercicio|Problema|Punto)\s+(\d+)\b", re.I)
_ANTERIOR = re.compile(r"\b(la|el)\s+(parte|pregunta|secci[oó]n|ejercicio|punto)\s+anterior\b", re.I)
_MISMO = re.compile(r"\b(la misma|el mismo|las mismas|los mismos)\s+\w+", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return " ".join(w[:-1] if len(w) > 4 and w.endswith("s") else w for w in s.split())   # embeddings = embedding


def referencias(secciones: list[Seccion]) -> list[tuple[str, str, str, str]]:
    """(origen, destino, texto que la delata, relación) para cada referencia entre secciones.

    `depende_de` solo apunta hacia atrás: una sección no puede depender de una que viene después.
    Una mención hacia adelante («se usa al final de la Parte 3») queda como `referencia_adelante`
    y el investigador no la sigue. «La misma división», «las mismas consultas» sin nombrar la parte
    se leen como dependencia de la sección numerada anterior."""
    por_rotulo: dict[tuple[str, str], str] = {}
    for s in secciones:
        m = _NUMERADA.match(s.titulo)
        if m:
            por_rotulo[(norm(m.group(1)), m.group(2))] = s.id
    orden = {s.id: i for i, s in enumerate(secciones)}
    aristas = []
    for i, s in enumerate(secciones):
        explicitas = []
        for m in _NUMERADA.finditer(s.texto):
            destino = por_rotulo.get((norm(m.group(1)), m.group(2)))
            if destino and destino != s.id:
                rel = "depende_de" if orden[destino] < i else "referencia_adelante"
                aristas.append((s.id, destino, " ".join(m.group(0).split()), rel))
                explicitas.append(destino)
        previa = next((secciones[j] for j in range(i - 1, -1, -1) if _NUMERADA.match(secciones[j].titulo)), None)
        if previa is not None and _NUMERADA.match(s.titulo):
            for patron in (_ANTERIOR, _MISMO):
                m = patron.search(s.texto)
                if m and not any(orden[d] < i for d in explicitas):
                    aristas.append((s.id, previa.id, " ".join(m.group(0).split()), "depende_de"))
                    break
    vistas, unicas = set(), []
    for a in aristas:
        if a[:2] not in vistas:
            vistas.add(a[:2]); unicas.append(a)
    return unicas


def es_seccion_de_entrega(s: Seccion) -> bool:
    return bool(re.search(r"\b(reporte|formato|entrega|entregable|informe)\b", norm(s.titulo)))


# ── Notas del curso: fragmentos por encabezado ──────────────────────────────────────────────
@dataclass
class Fragmento:
    id: str
    archivo: str
    titulo: str
    texto: str

    def cita(self) -> str:
        return f"[notas {self.archivo} §{self.titulo}]"


def fragmentos_curso(carpeta: Path = NOTAS, max_car: int = 1800) -> list[Fragmento]:
    salida = []
    for md in sorted(carpeta.glob("*.md")):
        actual_titulo, buf = md.stem, []
        def cerrar():
            texto = "\n".join(buf).strip()
            for k in range(0, len(texto), max_car):
                if texto[k:k + max_car].strip():
                    salida.append(Fragmento(f"N{len(salida)}", md.name, actual_titulo, texto[k:k + max_car]))
        for linea in md.read_text(encoding="utf-8").splitlines():
            if re.match(r"^#{2,3}\s", linea):
                cerrar(); buf = []; actual_titulo = linea.lstrip("# ").strip()
            else:
                buf.append(linea)
        cerrar()
    return salida


SISTEMA_ENTIDADES = """Extraes entidades de un fragmento de un enunciado o de unas notas de un curso de IA generativa.
Responde solo JSON: {"entidades": [{"nombre": "...", "tipo": "dataset|metodo|metrica|concepto"}]}
- dataset: conjuntos de datos con nombre (p. ej. Breast Cancer Wisconsin, Diabetes, Wine).
- metodo: algoritmos, modelos y técnicas (p. ej. Naive Bayes, regresión logística, TF-IDF, LSA, k-means, top-p).
- metrica: medidas de desempeño (p. ej. exactitud, F1 macro, MRR, R², silueta, entropía).
- concepto: ideas teóricas (p. ej. sesgo y varianza, curva de aprendizaje, temperatura).
Como mucho 12 entidades, con el nombre en su forma más común y en español si es un término en español."""


def entidades(llm: LLM, texto: str) -> list[dict]:
    r = llm.json("indexador", [{"role": "system", "content": SISTEMA_ENTIDADES},
                               {"role": "user", "content": texto[:4000]}])
    salida = []
    for e in r.get("entidades", []) if isinstance(r, dict) else []:
        if isinstance(e, dict) and e.get("nombre") and norm(str(e["nombre"])):
            tipo = e.get("tipo") if e.get("tipo") in TIPOS else "concepto"
            salida.append({"nombre": str(e["nombre"])[:80], "tipo": tipo})
    return salida


def _clave(texto: str) -> str:
    return hashlib.sha256(texto.encode()).hexdigest()[:16]


def indice_curso(llm: LLM, carpeta: Path = NOTAS, hilos: int = 8) -> dict:
    """Fragmentos, entidades y embeddings de las notas, con caché por contenido."""
    CACHE.mkdir(exist_ok=True)
    ruta = CACHE / "curso.json"
    cache = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    frags = fragmentos_curso(carpeta)
    faltan = [f for f in frags if _clave(f.texto) not in cache]
    if faltan:
        llm.traza("nodo", "indexador", nodo="indice_curso", fragmentos=len(frags), nuevos=len(faltan))
        with ThreadPoolExecutor(hilos) as ex:
            ents = list(ex.map(lambda f: entidades(llm, f"{f.titulo}\n{f.texto}"), faltan))
        vecs = llm.embed([f"{f.titulo}\n{f.texto}" for f in faltan])
        for f, e, v in zip(faltan, ents, vecs):
            cache[_clave(f.texto)] = {"entidades": e, "embedding": v}
        ruta.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return {"fragmentos": frags, "datos": {f.id: cache[_clave(f.texto)] for f in frags}}


# ── El grafo ────────────────────────────────────────────────────────────────────────────────
class Grafo:
    def __init__(self, doc: Documento, llm: LLM, con_curso: bool = True, hilos: int = 8):
        self.doc, self.llm = doc, llm
        self.g = nx.DiGraph()
        self.secciones = {s.id: s for s in doc.secciones}
        for s in doc.secciones:
            self.g.add_node(s.id, clase="seccion", titulo=s.titulo, pagina=s.pagina,
                            entrega=es_seccion_de_entrega(s))
        # capa 1 (REGLA 2): depende_de por regla
        self.aristas_regla = referencias(doc.secciones)
        for o, d, texto, rel in self.aristas_regla:
            self.g.add_edge(o, d, rel=rel, evidencia=texto)
        # capa 2: entidades del LLM, del enunciado (en paralelo) y del curso (con caché)
        with ThreadPoolExecutor(hilos) as ex:
            ents = list(ex.map(lambda s: entidades(self.llm, f"{s.titulo}\n{s.texto}"), doc.secciones))
        for s, es in zip(doc.secciones, ents):
            for e in es:
                self._entidad(s.id, e)
        self.curso = {"fragmentos": [], "datos": {}}
        if con_curso:
            self.curso = indice_curso(llm, hilos=hilos)
            for f in self.curso["fragmentos"]:
                self.g.add_node(f.id, clase="nota", titulo=f.titulo, archivo=f.archivo)
                for e in self.curso["datos"][f.id]["entidades"]:
                    self._entidad(f.id, e)
        self.emb_secciones = dict(zip(self.secciones, llm.embed([f"{s.titulo}\n{s.texto}" for s in doc.secciones])))

    def _entidad(self, nodo: str, e: dict) -> None:
        eid = "E:" + norm(e["nombre"])
        if eid not in self.g:
            self.g.add_node(eid, clase="entidad", nombre=e["nombre"], tipo=e["tipo"])
        self.g.add_edge(nodo, eid, rel="menciona")

    # ── consultas del investigador ──
    def dependencias(self, sid: str) -> list[str]:
        """Las secciones de las que depende `sid`, siguiendo depende_de hasta el final (sin repetir)."""
        visto, cola = [], [sid]
        while cola:
            actual = cola.pop(0)
            for _, d, datos in self.g.out_edges(actual, data=True):
                if datos.get("rel") == "depende_de" and d not in visto and d != sid:
                    visto.append(d); cola.append(d)
        return visto

    def seccion_de_entrega(self) -> Seccion | None:
        return next((self.secciones[n] for n, d in self.g.nodes(data=True) if d.get("entrega")), None)

    def _mismas(self, a: str, b: str) -> bool:
        """Dos entidades son la misma si una contiene a la otra palabra por palabra («naive bayes» ⊂
        «naive bayes gaussiano»). La más corta necesita dos palabras o más: «modelo» no basta."""
        pa, pb = a[2:].split(), b[2:].split()
        corta, larga = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
        return a == b or (len(corta) >= 2 and " ".join(corta) in " ".join(larga))

    def notas_relacionadas(self, sid: str, k: int = 2, max_df: float = 0.06,
                           min_cos: float = 0.495) -> list[tuple[Fragmento, list[str]]]:
        """Fragmentos del curso que comparten una entidad ESPECÍFICA con la sección.

        Una entidad que aparece en más de `max_df` de los fragmentos del curso («LLM», «prompt»)
        no une nada. El puntaje es la suma de log(N/df) de las entidades en común, y el coseno de
        bge-m3 solo desempata: sus valores quedan muy juntos (0,52–0,63 en la Tarea A) y por sí
        solo trae fragmentos que no tienen que ver. Si no hay ninguna entidad específica en común,
        no se trae nada: es mejor un contexto sin notas que uno con notas irrelevantes.
        `min_cos` es un piso adicional, calibrado con la Tarea A (2026-10-05): los fragmentos útiles
        quedaron entre 0,499 y 0,567 y los irrelevantes que compartían «exactitud» entre 0,458 y
        0,492. Es un umbral de una sola tarea; hay que revisarlo con las tareas reales."""
        import math
        frags = self.curso["fragmentos"]
        if not frags:
            return []
        de_nota = {f.id: {e for _, e, d in self.g.out_edges(f.id, data=True) if d.get("rel") == "menciona"} for f in frags}
        df: dict[str, int] = {}
        for ents in de_nota.values():
            for e in ents:
                df[e] = df.get(e, 0) + 1
        mias = [e for _, e, d in self.g.out_edges(sid, data=True) if d.get("rel") == "menciona"]
        puntaje: dict[str, tuple[float, float, list[str]]] = {}
        for f in frags:
            comunes = sorted({c for m in mias for c in de_nota[f.id]
                              if self._mismas(m, c) and df[c] / len(frags) <= max_df})
            cos = sum(a * b for a, b in zip(self.emb_secciones[sid], self.curso["datos"][f.id]["embedding"])) if comunes else 0.0
            if comunes and cos >= min_cos:
                puntaje[f.id] = (sum(math.log(len(frags) / df[c]) for c in comunes), cos,
                                 [self.g.nodes[c]["nombre"] for c in comunes])
        mejores = sorted(puntaje, key=lambda i: puntaje[i][:2], reverse=True)[:k]
        por_id = {f.id: f for f in frags}
        return [(por_id[i], puntaje[i][2]) for i in mejores]

    def top_k(self, consulta: str, k: int = 3) -> list[tuple[str, str, str]]:
        """Versión recortada «sin grafo» (2.b): los k fragmentos más parecidos, y nada más."""
        q = self.llm.embed([consulta])[0]
        candidatos = [(s.cita(self.doc.nombre), f"{s.titulo}\n{s.texto}", self.emb_secciones[s.id]) for s in self.doc.secciones]
        candidatos += [(f.cita(), f.texto, self.curso["datos"][f.id]["embedding"]) for f in self.curso["fragmentos"]]
        orden = sorted(candidatos, key=lambda c: -sum(a * b for a, b in zip(q, c[2])))
        return [(c[0], c[1], "similitud") for c in orden[:k]]

    # ── persistencia y dibujo ──
    def guardar(self, ruta: Path) -> None:
        datos = nx.node_link_data(self.g, edges="aristas")
        ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    def resumen(self) -> dict:
        clases = [d.get("clase") for _, d in self.g.nodes(data=True)]
        rels = [d.get("rel") for *_, d in self.g.edges(data=True)]
        return {"secciones": clases.count("seccion"), "notas": clases.count("nota"),
                "entidades": clases.count("entidad"), "depende_de": rels.count("depende_de"),
                "menciona": rels.count("menciona"),
                "aristas_regla": [f"{self.secciones[o].titulo} —{r}→ {self.secciones[d].titulo} («{t}»)"
                                  for o, d, t, r in self.aristas_regla]}

    def dibujar(self, ruta: Path) -> None:
        """Secciones, sus aristas depende_de (rojas, gruesas) y las entidades que comparten con el curso."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        secs = [n for n, d in self.g.nodes(data=True) if d["clase"] == "seccion"]
        ents = {e for s in secs for _, e, d in self.g.out_edges(s, data=True) if d["rel"] == "menciona"}
        con_curso = {e for e in ents if any(self.g.nodes[o]["clase"] == "nota" for o, _ in self.g.in_edges(e))}
        sub = self.g.subgraph(secs + sorted(ents))
        pos = {s: (0.0, -i * 1.6) for i, s in enumerate(secs)}
        for j, e in enumerate(sorted(ents)):
            pos[e] = (3.2 + 2.4 * (j % 2), -j * (1.6 * max(len(secs) - 1, 1) / max(len(ents) - 1, 1)))
        fig, ax = plt.subplots(figsize=(13, max(7, len(ents) * 0.42)))
        menciona = [(o, d) for o, d, x in sub.edges(data=True) if x["rel"] == "menciona"]
        depende = [(o, d) for o, d, x in sub.edges(data=True) if x["rel"] == "depende_de"]
        nx.draw_networkx_edges(sub, pos, edgelist=menciona, edge_color="#c9c9c9", width=0.7, arrows=False, ax=ax)
        nx.draw_networkx_edges(sub, pos, edgelist=depende, edge_color="#c0392b", width=2.6, arrowsize=22,
                               connectionstyle="arc3,rad=-0.55", ax=ax)
        nx.draw_networkx_nodes(sub, pos, nodelist=secs, node_color="#2e86c1", node_size=900, node_shape="s", ax=ax)
        nx.draw_networkx_nodes(sub, pos, nodelist=sorted(ents - con_curso), node_color="#f5b041", node_size=160, ax=ax)
        nx.draw_networkx_nodes(sub, pos, nodelist=sorted(ents & con_curso), node_color="#27ae60", node_size=160, ax=ax)
        for s in secs:
            ax.text(pos[s][0] - 0.35, pos[s][1], self.g.nodes[s]["titulo"][:38], ha="right", va="center", fontsize=9)
        for e in ents:
            ax.text(pos[e][0] + 0.15, pos[e][1], self.g.nodes[e]["nombre"][:30], ha="left", va="center", fontsize=7.5)
        ax.set_title(f"Grafo de «{self.doc.nombre}»: secciones (azul), depende_de por regla (rojo), "
                     f"entidades del LLM (verde = también en las notas del curso)", fontsize=10)
        ax.set_xlim(-4.5, 7.5); ax.axis("off"); fig.tight_layout(); fig.savefig(ruta, dpi=130); plt.close(fig)
