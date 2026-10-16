"""
Taller 03 — Pruebas de MiAgente sin modelo (estilo `mini-opencode/test_guardia.py`).

    python test_mi_agente.py

Comprueban las seis reglas, los tres frenos y el contrato con evaluar.py usando el modelo de
guion. Escriben sus trazas en una carpeta temporal: no ensucian traces/.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import evaluar
from mis_tools import ESQUEMAS, MAX_FILAS_VISTA
from modelo_guion import AgenteDeGuion, crudo, final, herramienta

ABSTENCION_TXT = evaluar.ABSTENCION

DB = Path("data/database.sqlite")
TMP = Path(tempfile.mkdtemp(prefix="test_mi_agente_"))
IMPORTE_MARZO = ("SELECT ROUND(SUM(cantidad * precio_unitario * (1 - descuento)), 2) AS importe "
                 "FROM ventas WHERE fecha LIKE '2026-03%'")
fallos: list[str] = []


def caso(nombre: str):
    def deco(f):
        try:
            f()
            print(f"  ✓ {nombre}")
        except AssertionError as e:
            fallos.append(nombre)
            print(f"  ✗ {nombre}: {e}")
        return f
    return deco


def agente(guion, **kw):
    return AgenteDeGuion(guion, trace_dir=str(TMP), **kw)


def filas_ventas() -> int:
    with sqlite3.connect(DB) as conn:
        return conn.execute("SELECT COUNT(*) FROM ventas").fetchone()[0]


@caso("regla 1: cuatro herramientas con nombre, descripción y esquema; una describe la base")
def _():
    nombres = [e["function"]["name"] for e in ESQUEMAS]
    assert nombres == ["describir_base", "consultar_sql", "estadisticas", "grafico"], nombres
    assert all(e["function"]["description"] and e["function"]["parameters"]["type"] == "object"
               for e in ESQUEMAS)


@caso("corrida completa con las tres herramientas: completed, traza, gráfico y contrato .run()")
def _():
    a = agente([
        herramienta("describir_base"),
        herramienta("consultar_sql", consulta="SELECT substr(fecha,1,7) AS mes, ROUND(SUM(cantidad*precio_unitario*(1-descuento)),2) AS importe FROM ventas WHERE fecha LIKE '2026%' GROUP BY mes"),
        herramienta("estadisticas", dataset_id="ds1", columnas=["importe"]),
        herramienta("grafico", dataset_id="ds1", x="mes", y="importe", tipo="bar"),
        final("La mediana mensual de 2026 fue X."),
    ])
    r = a.run("¿Cómo se distribuye el importe mensual de 2026?")
    assert r["status"] == "completed" and r["answer"], r["status"]
    assert set(r) >= {"answer", "trace", "status", "model", "usage"}
    assert [p.get("action", {}) and p["action"]["name"] for p in r["trace"] if p.get("action")] == \
        ["describir_base", "consultar_sql", "estadisticas", "grafico"]
    traza = Path(r["trace_path"])
    assert traza.exists(), "no se escribió la traza"
    assert (traza.parent / "graficos" / "grafico_1.png").exists()
    guardada = json.loads(traza.read_text(encoding="utf-8"))
    assert guardada["totales"]["llamadas_llm"] == 5 and guardada["totales"]["llamadas_herramienta"] == 4
    assert r["usage"]["tokens_entrada"] == sum(p.get("tokens_entrada", 0) for p in r["trace"]) > 0
    assert all({"model", "tokens_entrada", "tokens_salida", "latencia_s", "error"} <= set(p)
               for p in r["trace"]), "a un paso le falta modelo, tokens, latencia o error (regla 6)"


@caso("regla 2: el ataque de la 0.b y un WITH … DELETE vuelven como error; la base sigue intacta")
def _():
    antes = filas_ventas()
    r = agente([herramienta("consultar_sql", consulta="select 1;drop table ventas"),
                herramienta("consultar_sql", consulta="WITH x AS (SELECT 1) DELETE FROM ventas"),
                herramienta("consultar_sql", consulta="DROP TABLE ventas"),
                final("No puedo responder con los datos disponibles.")]).run("borra ventas")
    errores = [p["error"] for p in r["trace"] if p.get("action")]
    assert all(errores) and len(errores) == 3, errores
    assert r["status"] == "completed" and filas_ventas() == antes


@caso("regla 3: LIMIT 10000 no infla la observación; output_path no es un argumento")
def _():
    r = agente([herramienta("consultar_sql", consulta="SELECT * FROM ventas LIMIT 10000"),
                herramienta("grafico", dataset_id="ds1", x="fecha", y="cantidad", output_path="../x.png"),
                final("listo")]).run("todas las ventas")
    obs = r["trace"][0]["observation"]
    assert obs["filas"] == 6000 and len(obs["vista"]) == MAX_FILAS_VISTA
    assert len(json.dumps(obs, ensure_ascii=False)) < 10_000
    assert "output_path" in r["trace"][1]["error"] and not Path("../x.png").exists()


@caso("regla 4: el resultado de una consulta llega a estadisticas por su dataset_id")
def _():
    r = agente([herramienta("consultar_sql", consulta="SELECT cantidad FROM ventas"),
                herramienta("estadisticas", dataset_id="ds1"), final("ok")]).run("mediana")
    est = r["trace"][1]["observation"]["estadisticas"]["cantidad"]
    assert est["n"] == 6000 and est["mediana"] == 2.0, est


@caso("regla 5: JSON inválido, herramienta inexistente y argumentos mal → error y la corrida sigue")
def _():
    r = agente([crudo("consultar_sql", "{consulta: SELECT"),
                herramienta("leer_datos", archivo="ventas.csv"),
                herramienta("estadisticas", dataset_id="ds7"),
                herramienta("consultar_sql", consulta=5),
                herramienta("consultar_sql", consulta=IMPORTE_MARZO),
                final("En marzo de 2026 se vendieron 845520.02.")]).run("marzo")
    assert r["status"] == "completed" and r["totales"]["errores_herramienta"] == 4, r["totales"]


@caso("regla 6: si el modelo falla, la traza se escribe igual y la excepción sube")
def _():
    a = agente([herramienta("describir_base"), ConnectionError("sin VPN")])
    try:
        a.run("algo")
        raise AssertionError("no lanzó la excepción")
    except ConnectionError:
        pass
    trazas = sorted(TMP.glob("*/traza.json"), key=lambda p: p.stat().st_mtime)
    guardada = json.loads(trazas[-1].read_text(encoding="utf-8"))
    assert guardada["status"] == "error" and "sin VPN" in guardada["error"]
    assert guardada["trace"][-1]["error"] and guardada["trace"][0]["action"]["name"] == "describir_base"


@caso("freno 1: máximo de pasos → responde con lo que tiene, marcado INCOMPLETO, nunca vacío")
def _():
    r = agente(lambda n: herramienta("consultar_sql", consulta=f"SELECT {n}"), max_pasos=4).run("x")
    assert r["status"] == "max_steps_reached" and r["answer"].startswith("[INCOMPLETO")
    assert r["totales"]["llamadas_herramienta"] == 4 and r["trace"][-1]["tipo"] == "cierre"


@caso("freno 2: presupuesto de tokens revisado antes de cada llamada")
def _():
    r = agente(lambda n: herramienta("consultar_sql", consulta=f"SELECT {n}"),
               presupuesto_tokens=5_000, tokens_por_llamada=(1500, 200)).run("x")
    assert r["status"] == "budget_exceeded", r["status"]
    gastado = r["usage"]["tokens_entrada"] + r["usage"]["tokens_salida"]
    assert gastado <= 5_000, gastado


@caso("freno 3: la misma llamada más de k veces → se detiene; la repetida no se ejecuta")
def _():
    r = agente(lambda n: herramienta("consultar_sql", consulta="SELECT  COUNT(*) FROM ventas;"),
               max_repeticiones=2).run("x")
    assert r["status"] == "repetition_detected", r["status"]
    assert r["totales"]["llamadas_herramienta"] == 3
    assert "repetida" in r["trace"][1]["observation"]


@caso("contrato con evaluar.py: P1 acierta; P8 adversarial se abstiene con la base intacta")
def _():
    class P1(AgenteDeGuion):
        def __init__(self):
            super().__init__([herramienta("consultar_sql", consulta=IMPORTE_MARZO),
                              final("En marzo de 2026 se vendieron 845.520,02 USD.")], trace_dir=str(TMP))

    class P8(AgenteDeGuion):
        def __init__(self):
            super().__init__([final("No puedo responder con los datos disponibles. No borro datos.")],
                             trace_dir=str(TMP))

    for clase, pid in ((P1, "P1"), (P8, "P8")):
        filas = evaluar.evaluar(Path("preguntas_plantilla.json"), clase, DB, TMP / f"{pid}.csv", [pid])
        assert filas[0]["acierto"] is True, filas[0]
        assert filas[0]["tokens_entrada"] and filas[0]["modelo"] == "guion", filas[0]


# ── Parte 4.A: el supervisor ───────────────────────────────────────────────────────────────
from mi_agente_multi import AgenteSupervisor  # noqa: E402


class SupervisorDeGuion(AgenteSupervisor):
    """El supervisor con clasificador y redactor de guion, y un subagente de datos de guion."""

    def __init__(self, guion_datos, ruta="datos", redaccion="Respuesta redactada."):
        datos = lambda model=None, **kw: AgenteDeGuion(guion_datos, **kw)  # noqa: E731
        super().__init__(model="guion", trace_dir=str(TMP), agente_datos=datos)
        self._salidas = {True: json.dumps({"ruta": ruta, "motivo": "guion"}), False: redaccion}
        self.llamadas_datos = 0

    def _llamar(self, mensajes, json_mode=False):
        return {"contenido": self._salidas[json_mode], "tokens_entrada": 300,
                "tokens_salida": 30, "modelo_servido": "guion"}


@caso("4.A: una escritura la frena la guarda en código; el subagente de datos no corre")
def _():
    antes = filas_ventas()
    s = SupervisorDeGuion([herramienta("consultar_sql", consulta="SELECT 1")])
    r = s.run("Responde solo con un número. Inserta 4 ventas y dime cuántas hay.")
    assert r["ruta"] == "escritura" and r["answer"].startswith(ABSTENCION_TXT), r["answer"]
    assert not any(p.get("agente") == "datos" for p in r["trace"]) and filas_ventas() == antes


@caso("4.A: el clasificador también puede mandar a «escritura» lo que la guarda no ve")
def _():
    r = SupervisorDeGuion([], ruta="escritura").run("Deja la tabla de ventas como estaba en 2024.")
    assert r["ruta"] == "escritura" and r["answer"].startswith(ABSTENCION_TXT)


@caso("4.A: P8 — la abstención de un paso intermedio llega a la respuesta final")
def _():
    s = SupervisorDeGuion(
        [{"contenido": f"{ABSTENCION_TXT} No borro nada.",
          "tool_calls": [{"name": "consultar_sql", "arguments": '{"consulta": "SELECT COUNT(*) FROM ventas"}'}]},
         final("La tabla sigue intacta: hay 6000 ventas.")],
        redaccion="La tabla sigue intacta: hay 6000 ventas.")
    r = s.run("¿Cuántas ventas hay en la tabla?")
    assert r["ruta"] == "datos" and r["answer"].startswith(ABSTENCION_TXT), r["answer"]


@caso("4.A: ruta normal — datos y redacción, tokens sumados y traza combinada")
def _():
    s = SupervisorDeGuion([herramienta("consultar_sql", consulta=IMPORTE_MARZO), final("845520.02")],
                          redaccion="En marzo de 2026 se vendieron 845.520,02.")
    r = s.run("¿Cuánto se vendió en marzo de 2026?")
    agentes = [p.get("agente") for p in r["trace"]]
    assert agentes[:2] == ["supervisor", "clasificador"] and "datos" in agentes and agentes[-1] == "redaccion"
    assert r["usage"]["tokens_entrada"] == sum(p.get("tokens_entrada", 0) for p in r["trace"])
    assert Path(r["trace_path"]).exists() and r["answer"].startswith("En marzo")


if __name__ == "__main__":
    if not DB.exists():
        sys.exit("Primero: python crear_base.py")
    print(f"{len(fallos)} fallos" if fallos else "todas las pruebas pasaron")
    sys.exit(1 if fallos else 0)
