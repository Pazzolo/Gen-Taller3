"""
Taller 03 — Parte 3: cada freno de MiAgente, activado, con su traza.

    python frenos_demo.py            # con el modelo de guion (sin red, determinista)
    python frenos_demo.py --real     # además, pasos y presupuesto con el modelo real (H200)

«Un freno que nunca saltó no está probado.» Cada demostración corre `MiAgente` sin tocar su
bucle (el guion solo sustituye `_llamar_modelo`, como `guion.py` en la Parte 0), comprueba que
el freno saltó y copia la carpeta de la traza a entregables/parte3/<demo>/.

    1_max_pasos         el modelo pide consultas distintas sin parar → max_steps_reached
    2_presupuesto       cada llamada cuesta más que la anterior → budget_exceeded ANTES de llamar
    3_repeticion        la misma consulta (con otros espacios y un ';') tres veces → repetition_detected
    4_bucle_no_detectado  la misma pregunta escrita de 5 formas: el freno 3 no la ve; la para el 1
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from mi_agente import MiAgente
from modelo_guion import AgenteDeGuion, herramienta

load_dotenv()
DESTINO = Path("entregables/parte3")
lineas: list[str] = []


def registrar(nombre: str, r: dict, esperado: str) -> None:
    carpeta = Path(r["trace_path"]).parent
    shutil.copytree(carpeta, DESTINO / nombre, dirs_exist_ok=True)
    ok = r["status"] == esperado
    freno = next((p["motivo"] for p in r["trace"] if p.get("tipo") == "freno"), "—")
    entradas = [p["tokens_entrada"] for p in r["trace"] if p.get("tokens_entrada")]
    acumulado = [sum(entradas[: i + 1]) for i in range(len(entradas))]
    texto = (f"{'✓' if ok else '✗'} {nombre}: status={r['status']} (esperado {esperado})\n"
             f"   freno: {freno}\n"
             f"   modelo={r['model']} llamadas_llm={r['totales']['llamadas_llm']} "
             f"herramientas={r['totales']['llamadas_herramienta']} "
             f"tokens={r['usage']['tokens_entrada']}→{r['usage']['tokens_salida']}\n"
             f"   tokens de entrada por llamada: {entradas}  acumulado: {acumulado}\n"
             f"   respuesta: {r['answer'][:220]!r}\n"
             f"   traza: {DESTINO / nombre / 'traza.json'}")
    print(texto)
    lineas.append(texto)
    if not ok:
        raise SystemExit(f"El freno de {nombre} no saltó como se esperaba.")


def guion() -> None:
    # 1 · Máximo de pasos: consultas siempre distintas, ninguna repetida; solo el freno 1 la para.
    r = AgenteDeGuion(lambda n: herramienta(
        "consultar_sql", consulta=f"SELECT COUNT(*) FROM ventas WHERE fecha LIKE '2025-{n:02d}%'"),
        max_pasos=4).run("Cuenta las ventas de cada mes de 2025, una consulta por mes.")
    registrar("1_max_pasos", r, "max_steps_reached")

    # 2 · Presupuesto: el historial crece y cada llamada cuesta más; se corta ANTES de pasarse.
    r = AgenteDeGuion(lambda n: herramienta("describir_base") if n == 1 else herramienta(
        "consultar_sql", consulta=f"SELECT * FROM productos WHERE producto_id <= {n}"),
        presupuesto_tokens=12_000, tokens_por_llamada=(2_000, 300)).run("Revisa los productos.")
    registrar("2_presupuesto", r, "budget_exceeded")
    gastado = r["usage"]["tokens_entrada"] + r["usage"]["tokens_salida"]
    assert gastado <= 12_000, f"el presupuesto se superó: {gastado}"

    # 3 · Repetición: la misma consulta con espacios, mayúsculas y ';' distintos → misma clave.
    variantes = ["SELECT COUNT(*) FROM ventas", "select  count(*)  from ventas;", "SELECT COUNT(*)\nFROM ventas"]
    r = AgenteDeGuion(lambda n: herramienta("consultar_sql", consulta=variantes[(n - 1) % 3]),
                      max_repeticiones=2).run("¿Cuántas ventas hay?")
    registrar("3_repeticion", r, "repetition_detected")

    # 4 · Lo que el freno 3 NO detecta: la misma pregunta con SQL distinto (alias, LIKE vs
    #     BETWEEN, un LIMIT de más). Para el detector son llamadas diferentes; la para el freno 1.
    equivalentes = [
        "SELECT COUNT(*) FROM ventas WHERE fecha LIKE '2026-03%'",
        "SELECT COUNT(*) AS n FROM ventas WHERE fecha LIKE '2026-03%'",
        "SELECT COUNT(*) FROM ventas WHERE fecha BETWEEN '2026-03-01' AND '2026-03-31'",
        "SELECT COUNT(venta_id) FROM ventas WHERE substr(fecha, 1, 7) = '2026-03'",
        "SELECT COUNT(*) FROM ventas WHERE fecha LIKE '2026-03%' LIMIT 1",
    ]
    r = AgenteDeGuion(lambda n: herramienta("consultar_sql", consulta=equivalentes[(n - 1) % 5]),
                      max_pasos=5).run("¿Cuántas ventas hubo en marzo de 2026?")
    registrar("4_bucle_no_detectado", r, "max_steps_reached")


def real() -> None:
    # Pasos: G13 necesita 3 herramientas (consulta → estadísticas → respuesta); con 2 pasos no llega.
    r = MiAgente(max_pasos=2).run("¿Cuál fue la mediana del importe facturado por mes en 2026?")
    registrar("real_1_max_pasos", r, "max_steps_reached")
    # Presupuesto: la primera llamada cuesta ~1 200 tokens y la segunda ~1 800; 3 000 no alcanzan.
    r = MiAgente(presupuesto_tokens=3_000).run("¿Qué producto vendió menos unidades en 2025, y cuántas?")
    registrar("real_2_presupuesto", r, "budget_exceeded")


if __name__ == "__main__":
    DESTINO.mkdir(parents=True, exist_ok=True)
    print("── con el modelo de guion ──")
    guion()
    if "--real" in sys.argv:
        print("\n── con el modelo real ──")
        real()
    (DESTINO / "resumen.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"\n✓ {DESTINO}/resumen.txt")
