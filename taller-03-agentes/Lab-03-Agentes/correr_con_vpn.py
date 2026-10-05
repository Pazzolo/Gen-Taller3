"""
Taller 03 — Todo lo que necesita la H200, en orden. Conecta GlobalProtect y corre:

    python correr_con_vpn.py todo       # los cuatro pasos de abajo, en orden
    python correr_con_vpn.py modelo     # lee el id de /v1/models y lo escribe en .env (AGENT_MODEL)
    python correr_con_vpn.py humo       # una llamada con tools=: ¿el servidor acepta herramientas?
    python correr_con_vpn.py parte0a    # 0.a: el agente de ejemplo sobre P1 y P3 → entregables/parte0/
    python correr_con_vpn.py parte1     # dos corridas de MiAgente para el informe → entregables/parte1/

El id del modelo no se escribe a mano: se le pregunta al servidor, como `mini-opencode/h200.py`.
Lo que se use queda anotado en `entregables/modelo.json` (ruta, modelo y fecha, para el informe).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

AQUI = Path(__file__).resolve().parent
ENV = AQUI / ".env"
ENTREGABLES = AQUI / "entregables"
PREGUNTAS_PARTE1 = [
    # Una que usa las tres herramientas (consulta → estadísticas → gráfico), como pide la Parte 1.
    "Calcula el importe vendido por mes en 2025, dame la media y la mediana mensual, "
    "y guarda un gráfico de barras del importe por mes.",
    # Una simple, la del ejemplo del enunciado.
    "¿Cuánto se vendió en marzo de 2026?",
]


def _base_url() -> str:
    load_dotenv(ENV, override=True)
    return os.getenv("OPENAI_BASE_URL", "").rstrip("/")


def modelo() -> str:
    base = _base_url()
    if not base:
        sys.exit("OPENAI_BASE_URL no está en .env")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=8) as r:
            ids = [m["id"] for m in json.load(r)["data"]]
    except Exception as err:
        sys.exit(f"Sin respuesta de {base} ({err}). ¿Está GlobalProtect conectada?")
    elegido = ids[0]
    texto = ENV.read_text(encoding="utf-8")
    texto = (re.sub(r"(?m)^AGENT_MODEL=.*$", f"AGENT_MODEL={elegido}", texto)
             if re.search(r"(?m)^AGENT_MODEL=", texto) else texto + f"\nAGENT_MODEL={elegido}\n")
    ENV.write_text(texto, encoding="utf-8")
    ENTREGABLES.mkdir(exist_ok=True)
    (ENTREGABLES / "modelo.json").write_text(json.dumps({
        "ruta": "H200 de la USFQ (vLLM, GlobalProtect)" if "172.28." in base else base,
        "base_url": base, "modelo": elegido, "modelos_servidos": ids,
        "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ AGENT_MODEL={elegido}  (servidos: {ids}) → .env y entregables/modelo.json")
    return elegido


def humo() -> None:
    load_dotenv(ENV, override=True)
    from mi_agente import MiAgente
    agente = MiAgente()
    r = agente._llamar_modelo(
        [{"role": "user", "content": "Usa la herramienta describir_base para ver las tablas."}])
    pidio = [tc["name"] for tc in r["tool_calls"]]
    print(f"modelo={r['modelo_servido']} tokens={r['tokens_entrada']}→{r['tokens_salida']} "
          f"razonamiento={len(r['razonamiento'])} car. fin={r['fin']} herramientas pedidas={pidio}")
    if not pidio:
        print("⚠ El modelo no pidió ninguna herramienta. Si el contenido trae el JSON de la llamada "
              "como texto, el servidor no tiene activado el tool calling: hay que pasar a esquemas "
              f"en el prompt. Contenido: {r['contenido'][:300]!r}")
        sys.exit(1)
    print("✓ el servidor acepta tools= y el modelo las usa")


def _correr(*args: str) -> None:
    print("$", " ".join(args))
    subprocess.run([sys.executable, *args], cwd=AQUI, check=True)


def parte0a() -> None:
    """0.a: el andamiaje sin tocar, sobre P1 y P3. Copia sus trazas y resume status/pasos/acciones."""
    trazas = AQUI / "traces"
    antes = set(trazas.glob("trace-*.json"))
    _correr("evaluar.py", "--solo", "P1", "P3", "--salida", "parte0.csv")
    nuevas = sorted(set(trazas.glob("trace-*.json")) - antes)
    destino = ENTREGABLES / "parte0"
    destino.mkdir(parents=True, exist_ok=True)
    shutil.copy(AQUI / "parte0.csv", destino / "0a_parte0.csv")
    lineas = []
    for i, ruta in enumerate(nuevas, start=1):
        shutil.copy(ruta, destino / f"0a_trace_{i}.json")
        t = json.loads(ruta.read_text(encoding="utf-8"))
        acciones = [p.get("action") for p in t["trace"]]
        lineas.append(f"pregunta: {t['question']}\n  status={t['status']} pasos={len(t['trace'])} "
                      f"acciones={json.dumps(acciones, ensure_ascii=False)}\n  respuesta: {t['answer'][:300]}")
    if len(nuevas) < 2:
        lineas.append(f"⚠ se esperaban 2 trazas y hay {len(nuevas)}: una corrida murió antes de "
                      "escribirla (ver la columna excepcion de 0a_parte0.csv)")
    texto = "\n".join(lineas)
    (destino / "0a_resumen.txt").write_text(texto + "\n", encoding="utf-8")
    print(texto)


def parte1() -> None:
    """Dos corridas completas de MiAgente para el informe; copia su carpeta de traza."""
    load_dotenv(ENV, override=True)
    from mi_agente import MiAgente
    destino = ENTREGABLES / "parte1"
    destino.mkdir(parents=True, exist_ok=True)
    for i, pregunta in enumerate(PREGUNTAS_PARTE1, start=1):
        r = MiAgente().run(pregunta)
        carpeta = Path(r["trace_path"]).parent
        shutil.copytree(carpeta, destino / f"corrida_{i}", dirs_exist_ok=True)
        usadas = sorted({p["action"]["name"] for p in r["trace"] if p.get("action")})
        print(f"corrida {i}: status={r['status']} herramientas={usadas} totales={r['totales']}\n"
              f"  respuesta: {r['answer'][:300]}")


PASOS = {"modelo": modelo, "humo": humo, "parte0a": parte0a, "parte1": parte1}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in (*PASOS, "todo"):
        sys.exit(__doc__)
    for nombre in (PASOS if sys.argv[1] == "todo" else [sys.argv[1]]):
        print(f"\n── {nombre} ──")
        PASOS[nombre]()
