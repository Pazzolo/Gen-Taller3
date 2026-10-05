"""Convierte informe.md en informe.html (con estilos de impresión) para imprimirlo a PDF con Chrome.

    python construir_pdf.py   # requiere el paquete `markdown`; luego Chrome headless --print-to-pdf
"""
from pathlib import Path
import markdown

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 10.2pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 19pt; margin: 0 0 4pt; } h2 { font-size: 14pt; margin-top: 16pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 12pt; break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.8pt; break-inside: auto; }
th, td { border: 1px solid #c8c8c8; padding: 3pt 5pt; vertical-align: top; text-align: left; } th { background: #f0f0f0; }
tr { break-inside: avoid; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.4pt; background: #f4f4f4; padding: 0 2px; border-radius: 2px; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 6pt; font-size: 7.6pt; line-height: 1.3; white-space: pre-wrap; word-break: break-word; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
img { max-width: 92%; display: block; margin: 6pt auto; } hr { border: 0; border-top: 1px solid #ddd; margin: 14pt 0; }
"""
aqui = Path(__file__).parent
html = markdown.markdown((aqui / "informe.md").read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
(aqui / "informe.html").write_text(f'<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Taller 03 — Informe</title><style>{CSS}</style></head><body>{html}</body></html>', encoding="utf-8")
print("✓ informe.html")
