"""
Taller 03 — Las herramientas de MiAgente.

Cuatro herramientas, cada una con nombre, descripción y esquema JSON (REGLA 1):

    describir_base   tablas, columnas, valores posibles y la fórmula del importe
    consultar_sql    SQL de solo lectura → guarda el resultado como dataset y devuelve una vista
    estadisticas     media, mediana, dispersión y cuartiles de un dataset
    grafico          guarda un gráfico de un dataset como PNG

Las diferencias con `tools.py` del andamiaje (Parte 0) están marcadas con `# REGLA n`:

- REGLA 2: la base se abre en modo solo lectura y con `PRAGMA query_only`; se acepta una sola
  sentencia y solo `SELECT` o `WITH … SELECT`. El filtro de texto es la primera capa, no la
  única: aunque se lo saltaran, SQLite rechaza la escritura.
- REGLA 3: el máximo de filas, el tiempo máximo de una consulta y la carpeta de los gráficos
  son constantes de este archivo. Ninguno es un argumento que el modelo pueda cambiar.
- REGLA 4: `consultar_sql` guarda el resultado completo en un CSV de la corrida y devuelve un
  `dataset_id` («ds1», «ds2»…). `estadisticas` y `grafico` reciben ese id, nunca una ruta.
  Así la consulta llega a las otras dos herramientas, el modelo no ve ni elige rutas, y al
  historial solo vuelve una vista de pocas filas (en la 0.c volvían 842 321 caracteres).
- REGLA 5: `ejecutar()` valida los argumentos contra el esquema antes de llamar a la función,
  y cualquier fallo vuelve como `{"error": "…"}`. Nada aquí lanza una excepción hacia el bucle.

Patrones tomados de la referencia del curso: el formato de esquema de `s3-lun-profesor.ipynb`
(`a_formato_openai`) y la clase `Herramienta` de `mini-opencode/agentes.py`; la idea de una
guarda determinística en código, de `mini-opencode/guardia.py`.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # sin ventana: el agente corre en consola y en evaluar.py
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# ── Límites: los pone el código, no el modelo (REGLA 3) ───────────────────────────────────
MAX_FILAS_DATASET = 10_000      # filas que se leen y se guardan en el CSV del dataset
MAX_FILAS_VISTA = 20            # filas que vuelven al modelo en la observación
MAX_SEGUNDOS_CONSULTA = 5.0     # una consulta más lenta se aborta (p. ej. un producto cartesiano)
MAX_CARACTERES_CONSULTA = 4_000
CARPETA_DATOS = "datos"         # dentro de la carpeta de la corrida
CARPETA_GRAFICOS = "graficos"   # dentro de la carpeta de la corrida

# Lo que el esquema de la base no dice y el modelo no puede adivinar (Parte 0.a).
NOTAS_DOMINIO = [
    "El importe de una venta es cantidad * precio_unitario * (1 - descuento). "
    "No existe una columna 'importe', 'total' ni 'monto': hay que calcularlo.",
    "descuento es una fracción entre 0 y 1 (0.15 = 15 %).",
    "Las fechas son texto ISO 'AAAA-MM-DD': para un mes usa fecha LIKE '2026-03%'; "
    "para un rango, fecha BETWEEN '2026-01-01' AND '2026-06-30'.",
    "ventas.cliente_id → clientes.cliente_id y ventas.producto_id → productos.producto_id.",
    "No hay datos de costos, márgenes, ganancias, inventario ni satisfacción de clientes.",
]


# ── Los datos que comparten las herramientas dentro de UNA corrida (REGLA 4) ───────────────
class Espacio:
    """La carpeta de una corrida: sus datasets y sus gráficos. La crea MiAgente.run()."""

    def __init__(self, db_path: Path, carpeta_corrida: Path):
        self.db_path = Path(db_path)
        self.carpeta = Path(carpeta_corrida)
        self.datasets: dict[str, Path] = {}
        self.n_graficos = 0

    def nuevo_dataset(self, frame: pd.DataFrame) -> str:
        dataset_id = f"ds{len(self.datasets) + 1}"
        ruta = self.carpeta / CARPETA_DATOS / f"{dataset_id}.csv"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(ruta, index=False)
        self.datasets[dataset_id] = ruta
        return dataset_id

    def leer_dataset(self, dataset_id: str) -> pd.DataFrame:
        if dataset_id not in self.datasets:
            disponibles = list(self.datasets) or "ninguno: primero usa consultar_sql"
            raise ErrorHerramienta(f"No existe el dataset '{dataset_id}'. Disponibles: {disponibles}")
        return pd.read_csv(self.datasets[dataset_id])

    def ruta_grafico(self) -> Path:
        # REGLA 3: nombre generado por el código; dos gráficos nunca se pisan (Parte 0.c).
        self.n_graficos += 1
        ruta = self.carpeta / CARPETA_GRAFICOS / f"grafico_{self.n_graficos}.png"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        return ruta


class ErrorHerramienta(Exception):
    """Un error que se le devuelve al modelo como {"error": ...}."""


# ── La guarda de SQL (REGLA 2) ─────────────────────────────────────────────────────────────
_COMENTARIO_LINEA = re.compile(r"--[^\n]*")
_COMENTARIO_BLOQUE = re.compile(r"/\*.*?\*/", re.S)


def validar_sql(consulta: str) -> str:
    """Devuelve la consulta lista para ejecutar, o lanza ErrorHerramienta con el motivo."""
    if len(consulta) > MAX_CARACTERES_CONSULTA:
        raise ErrorHerramienta(f"Consulta demasiado larga (máximo {MAX_CARACTERES_CONSULTA} caracteres).")
    limpia = _COMENTARIO_BLOQUE.sub(" ", _COMENTARIO_LINEA.sub(" ", consulta)).strip()
    limpia = limpia.rstrip().rstrip(";").rstrip()
    if not limpia:
        raise ErrorHerramienta("La consulta está vacía.")
    # Una sola sentencia: cualquier ';' que quede separa dos. Se rechaza también dentro de un
    # literal de texto: es raro en una pregunta analítica y no vale el riesgo de interpretarlo.
    if ";" in limpia:
        raise ErrorHerramienta("Solo se acepta una sentencia SQL por llamada (no uses ';').")
    primera = limpia.split(None, 1)[0].lower()
    if primera not in ("select", "with"):
        raise ErrorHerramienta("Solo se permiten consultas de lectura: SELECT o WITH … SELECT.")
    return limpia


def _conectar_solo_lectura(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise ErrorHerramienta(f"No existe la base de datos: {db_path.name}")
    # REGLA 2: modo solo lectura del archivo y, además, de la conexión.
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


# ── Las herramientas ───────────────────────────────────────────────────────────────────────
def describir_base(espacio: Espacio) -> dict:
    with _conectar_solo_lectura(espacio.db_path) as conn:
        tablas = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        salida = {}
        for tabla in tablas:
            columnas = conn.execute(f'PRAGMA table_info("{tabla}")').fetchall()
            n = conn.execute(f'SELECT COUNT(*) FROM "{tabla}"').fetchone()[0]
            info_columnas = {}
            for _, nombre, tipo, *_ in columnas:
                distintos = conn.execute(
                    f'SELECT COUNT(DISTINCT "{nombre}") FROM "{tabla}"').fetchone()[0]
                info = {"tipo": tipo}
                if tipo == "TEXT" and distintos <= 15:
                    info["valores"] = [r[0] for r in conn.execute(
                        f'SELECT DISTINCT "{nombre}" FROM "{tabla}" ORDER BY 1')]
                elif tipo == "TEXT" and nombre.startswith("fecha"):
                    info["rango"] = list(conn.execute(
                        f'SELECT MIN("{nombre}"), MAX("{nombre}") FROM "{tabla}"').fetchone())
                elif tipo in ("REAL", "INTEGER") and not nombre.endswith("_id"):
                    info["rango"] = list(conn.execute(
                        f'SELECT MIN("{nombre}"), MAX("{nombre}") FROM "{tabla}"').fetchone())
                info_columnas[nombre] = info
            salida[tabla] = {"filas": n, "columnas": info_columnas}
    return {"tablas": salida, "notas": NOTAS_DOMINIO}


def consultar_sql(espacio: Espacio, consulta: str) -> dict:
    sql = validar_sql(consulta)
    limite_tiempo = time.monotonic() + MAX_SEGUNDOS_CONSULTA
    with _conectar_solo_lectura(espacio.db_path) as conn:
        # REGLA 3: el tiempo máximo lo pone el código; devolver 1 aborta la consulta.
        conn.set_progress_handler(lambda: int(time.monotonic() > limite_tiempo), 10_000)
        try:
            cursor = conn.execute(sql)
        except sqlite3.OperationalError as err:
            if "interrupted" in str(err):
                raise ErrorHerramienta(f"La consulta superó {MAX_SEGUNDOS_CONSULTA:.0f} s y se canceló.") from err
            raise ErrorHerramienta(f"Error de SQL: {err}") from err
        except sqlite3.Error as err:
            raise ErrorHerramienta(f"Error de SQL: {err}") from err
        columnas = [d[0] for d in cursor.description or []]
        # REGLA 3: el tope de filas no depende de un LIMIT escrito en la consulta (Parte 0.c).
        filas = cursor.fetchmany(MAX_FILAS_DATASET + 1)
    truncado = len(filas) > MAX_FILAS_DATASET
    frame = pd.DataFrame(filas[:MAX_FILAS_DATASET], columns=columnas)
    dataset_id = espacio.nuevo_dataset(frame)
    resultado = {
        "dataset_id": dataset_id,
        "columnas": columnas,
        "filas": len(frame),
        "vista": json.loads(frame.head(MAX_FILAS_VISTA).to_json(orient="records", force_ascii=False)),
    }
    if len(frame) > MAX_FILAS_VISTA:
        resultado["aviso"] = (f"Solo se muestran {MAX_FILAS_VISTA} de {len(frame)} filas. "
                              f"Agrega en SQL o usa estadisticas/grafico con '{dataset_id}'.")
    if truncado:
        resultado["truncado"] = f"El resultado tenía más de {MAX_FILAS_DATASET} filas y se cortó."
    return resultado


def estadisticas(espacio: Espacio, dataset_id: str, columnas: list[str] | None = None) -> dict:
    frame = espacio.leer_dataset(dataset_id)
    numericas = frame.select_dtypes(include="number")
    if columnas:
        faltan = [c for c in columnas if c not in numericas.columns]
        if faltan:
            raise ErrorHerramienta(f"Columnas numéricas no encontradas: {faltan}. "
                                   f"Numéricas en {dataset_id}: {list(numericas.columns)}")
        numericas = numericas[columnas]
    if numericas.empty:
        raise ErrorHerramienta(f"El dataset {dataset_id} no tiene columnas numéricas.")

    def _r(x: float) -> float | None:
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)

    return {
        "dataset_id": dataset_id,
        "estadisticas": {
            col: {"n": int(s.count()), "media": _r(s.mean()), "mediana": _r(s.median()),
                  "desviacion_estandar": _r(s.std()), "minimo": _r(s.min()),
                  "q1": _r(s.quantile(0.25)), "q3": _r(s.quantile(0.75)), "maximo": _r(s.max()),
                  "suma": _r(s.sum())}
            for col, s in numericas.items()
        },
    }


def grafico(espacio: Espacio, dataset_id: str, x: str, y: str, tipo: str = "line",
            titulo: str | None = None) -> dict:
    frame = espacio.leer_dataset(dataset_id)
    for col in (x, y):
        if col not in frame.columns:
            raise ErrorHerramienta(f"La columna '{col}' no está en {dataset_id}. "
                                   f"Disponibles: {list(frame.columns)}")
    ruta = espacio.ruta_grafico()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    try:
        if tipo == "bar":
            ax.bar(frame[x].astype(str), frame[y])
        elif tipo == "scatter":
            ax.scatter(frame[x], frame[y])
        else:
            ax.plot(frame[x].astype(str), frame[y], marker="o")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if titulo:
            ax.set_title(titulo)
        if len(frame) > 8:
            ax.tick_params(axis="x", labelrotation=60)
        fig.tight_layout()
        fig.savefig(ruta, dpi=110)
    finally:
        plt.close(fig)
    return {"grafico": ruta.name, "tipo": tipo, "x": x, "y": y, "puntos": len(frame)}


# ── Registro: descripción + esquema + función (REGLA 1) ────────────────────────────────────
@dataclass
class Herramienta:
    nombre: str
    descripcion: str
    parametros: dict
    funcion: Callable[..., dict]

    def esquema(self) -> dict:
        """Formato `tools` de la API de OpenAI (el que entiende el vLLM de la H200)."""
        return {"type": "function", "function": {
            "name": self.nombre, "description": self.descripcion, "parameters": self.parametros}}


HERRAMIENTAS = [
    Herramienta(
        "describir_base",
        "Describe la base de datos: tablas, columnas con su tipo, valores posibles de las "
        "columnas categóricas, rangos de fechas y números, y notas del dominio (por ejemplo, "
        "cómo se calcula el importe de una venta). Úsala antes de escribir tu primera consulta.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        describir_base,
    ),
    Herramienta(
        "consultar_sql",
        "Ejecuta UNA consulta SQLite de solo lectura (SELECT o WITH … SELECT) sobre la base. "
        "Guarda el resultado completo como un dataset y devuelve su dataset_id, las columnas, "
        f"el número de filas y una vista de hasta {MAX_FILAS_VISTA} filas. Haz los cálculos "
        "(SUM, AVG, GROUP BY, JOIN) en SQL siempre que puedas. No escribe ni borra datos.",
        {"type": "object",
         "properties": {"consulta": {"type": "string", "description": "Una sola sentencia SELECT, sin ';' intermedios."}},
         "required": ["consulta"], "additionalProperties": False},
        consultar_sql,
    ),
    Herramienta(
        "estadisticas",
        "Calcula n, media, mediana, desviación estándar, mínimo, cuartiles, máximo y suma de "
        "las columnas numéricas de un dataset creado por consultar_sql. Úsala para medianas y "
        "dispersión, que SQLite no calcula.",
        {"type": "object",
         "properties": {
             "dataset_id": {"type": "string", "description": "El id que devolvió consultar_sql, p. ej. 'ds1'."},
             "columnas": {"type": "array", "items": {"type": "string"},
                          "description": "Columnas numéricas a resumir; si se omite, todas."}},
         "required": ["dataset_id"], "additionalProperties": False},
        estadisticas,
    ),
    Herramienta(
        "grafico",
        "Guarda un gráfico (línea, barras o dispersión) de dos columnas de un dataset creado por "
        "consultar_sql y devuelve el nombre del archivo. El código decide dónde se guarda.",
        {"type": "object",
         "properties": {
             "dataset_id": {"type": "string", "description": "El id que devolvió consultar_sql."},
             "x": {"type": "string", "description": "Columna del eje x."},
             "y": {"type": "string", "description": "Columna numérica del eje y."},
             "tipo": {"type": "string", "enum": ["line", "bar", "scatter"]},
             "titulo": {"type": "string"}},
         "required": ["dataset_id", "x", "y"], "additionalProperties": False},
        grafico,
    ),
]
POR_NOMBRE = {h.nombre: h for h in HERRAMIENTAS}
ESQUEMAS = [h.esquema() for h in HERRAMIENTAS]


# ── Validación y ejecución: un error nunca sale de aquí como excepción (REGLA 5) ────────────
_TIPOS = {"string": str, "array": list, "object": dict, "integer": int, "number": (int, float),
          "boolean": bool}


def validar_argumentos(args: Any, esquema: dict) -> None:
    if not isinstance(args, dict):
        raise ErrorHerramienta("Los argumentos deben ser un objeto JSON.")
    propiedades = esquema.get("properties", {})
    faltan = [r for r in esquema.get("required", []) if r not in args]
    if faltan:
        raise ErrorHerramienta(f"Faltan argumentos obligatorios: {faltan}")
    sobran = [k for k in args if k not in propiedades]
    if sobran and esquema.get("additionalProperties") is False:
        raise ErrorHerramienta(f"Argumentos no permitidos: {sobran}. Permitidos: {list(propiedades)}")
    for nombre, valor in args.items():
        regla = propiedades[nombre]
        if valor is None and nombre not in esquema.get("required", []):
            continue
        if not isinstance(valor, _TIPOS[regla["type"]]):
            raise ErrorHerramienta(f"'{nombre}' debe ser de tipo {regla['type']}.")
        if "enum" in regla and valor not in regla["enum"]:
            raise ErrorHerramienta(f"'{nombre}' debe ser uno de {regla['enum']}.")
        if regla["type"] == "array" and not all(isinstance(v, _TIPOS[regla["items"]["type"]]) for v in valor):
            raise ErrorHerramienta(f"Cada elemento de '{nombre}' debe ser {regla['items']['type']}.")


def ejecutar(espacio: Espacio, nombre: str, args: Any) -> dict:
    """Valida y ejecuta una herramienta. Siempre devuelve un dict; si algo falla, {"error": ...}."""
    herramienta = POR_NOMBRE.get(nombre)
    if herramienta is None:
        return {"error": f"La herramienta '{nombre}' no existe. Disponibles: {list(POR_NOMBRE)}"}
    try:
        validar_argumentos(args, herramienta.parametros)
        return herramienta.funcion(espacio, **args)
    except ErrorHerramienta as err:
        return {"error": str(err)}
    except Exception as err:  # un fallo inesperado también vuelve al modelo como dato
        return {"error": f"{type(err).__name__}: {err}"}
