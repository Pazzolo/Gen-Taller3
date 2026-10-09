```python
# -*- coding: utf-8 -*-
"""
T1 - Parte 1: linea base lexica de recuperacion (TF-IDF + similitud coseno).

Datos: el corpus de 10 documentos (d01-d10) y las 6 consultas (q1-q6) EXACTOS
del enunciado de la tarea. El script no inventa ningun texto: localiza el
archivo del enunciado en la carpeta actual (p. ej. "enunciado.md",
"tarea-c-recuperacion-lexica-lsa.md" o similar) y extrae verbatim las
entradas d01..d10 y q1..q6. Si no logra reunir el corpus completo, aborta
con un error explicito en lugar de sustituir los datos.

Juicios de relevancia (fijados por el enunciado):
    q1 -> d04 | q2 -> d03 | q3 -> d08 | q4 -> d05 | q5 -> d09 | q6 -> d02

Pipeline:
  1) TfidfVectorizer con parametros por defecto; fit SOLO con los documentos,
     las consultas solo se transforman.
  2) Similitud coseno consulta-documento (matriz 6 x 10).
  3) Ranking de los 10 documentos por consulta (similitud decreciente; los
     empates se resuelven por el orden d01..d10).
  4) Metricas sobre las 6 consultas: Hit@1, Hit@3 y MRR.

Salidas:
  - resultados.json: fuente de datos, textos cargados, rankings, similitudes,
    metricas por consulta y agregadas.
  - stdout: cifras principales con 4 decimales.
Figura: esta subtarea no pide figura.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. Constantes de la subtarea
# ---------------------------------------------------------------------------
DOC_IDS = ["d%02d" % i for i in range(1, 11)]
QUERY_IDS = ["q%d" % i for i in range(1, 7)]
RELEVANT = {"q1": "d04", "q2": "d03", "q3": "d08",
            "q4": "d05", "q5": "d09", "q6": "d02"}
MIN_LEN_DOC = 15   # longitud minima plausible de un texto de documento
MIN_LEN_Q = 8      # longitud minima plausible de una consulta

# ---------------------------------------------------------------------------
# 2. Parser del enunciado: entradas tipo "d01: texto", "- **d01** texto",
#    "| d01 | texto |", "Documento 1: texto", con continuacion de parrafo.
# ---------------------------------------------------------------------------
RE_MARKER = re.compile(
    r'^\s*(?:[#>*+|\-{\[\(]+\s*)*(?:\d+\s*[.)]\s+)?'
    r'(?:\*\*|__)?\s*([dqDQ])\s*(\d{1,2})\s*(?:\*\*|__)?'
    r'(?:\s*\([^)]*\))?'
    r'\s*(?:\]\s*\()?'
    r'(?:[:.\-–—=)]|\)\s*)?\s*'
    r'(.*)$'
)
RE_MARKER_WORD = re.compile(
    r'^\s*(?:[#>*+|\-{\[\(]+\s*)*(?:\*\*|__)?\s*'
    r'(documentos?|docs?|consultas?|query|queries|preguntas?)\s+'
    r'(\d{1,2})(?:\s*\([^)]*\))?(?:\*\*|__)?\s*[:.\-–—=)]\s*(.*)$',
    re.IGNORECASE,
)
RE_ARROWS = re.compile(r'→|->|=>')
RE_BODY_ID_START = re.compile(r'^[\(\[\{\-–—,;]?\s*[dqDQ]\s*\d')
RE_ID_TOKEN = re.compile(r'\b[dqDQ]\d{1,2}\b')
RE_CONT_SKIP = re.compile(
    r'→|->|=>|\b[dqDQ]\d{1,2}\b|^\s*\(?\s*relevan\w*\s*[:\-–—=]',
    re.IGNORECASE,
)
RE_TABLE_SEP = re.compile(r'^[\s|:\-+=]+$')
RE_KV_DQ = re.compile(r'"(d\d{1,2}|q\d{1,2})"\s*[:=]\s*"([^"]*)"', re.IGNORECASE)
RE_KV_SQ = re.compile(r"'(d\d{1,2}|q\d{1,2})'\s*[:=]\s*'([^']*)'", re.IGNORECASE)
QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘
