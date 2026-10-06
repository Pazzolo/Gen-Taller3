"""Sandbox: revisar el script antes de ejecutarlo y ejecutarlo encerrado. REGLA 3. Sin LLM.

Tres capas independientes, porque ninguna basta sola:

1. **Revisión estática (AST), antes de que exista un proceso.** Rechaza red, procesos, borrado,
   `eval`/`exec`/`__import__` y rutas fuera de la carpeta. Lista negra de capacidades, con
   rechazo ante la duda («fail closed», como `mini-opencode/guardia.py`): un falso positivo cuesta un
   reintento del programador; un falso negativo cuesta un archivo o una clave.
   Aparte, marca lo que **quiere la red para descargar datos** (`fetch_openml`, `load_dataset`,
   `download=True`…): eso no se rechaza, se para y se le pregunta a una persona (freno 4).
2. **Ejecución encerrada.** Entorno vacío salvo lo mínimo para que Python arranque: ninguna variable
   del usuario (y por lo tanto ninguna clave) llega al script. Carpeta propia, `TMPDIR` dentro de
   ella, sesión propia (`start_new_session`) y tiempo máximo: al vencer se mata **el grupo de procesos
   entero** (`os.killpg`), así que un hijo que el script lanzara tampoco sobrevive (freno 3).
   `mini-opencode/herramientas.py` mata solo el proceso directo.
3. **`sandbox-exec` de macOS, si existe.** Un perfil del sistema operativo niega la red y la escritura
   fuera de la carpeta. Atrapa lo que la revisión estática no vea (p. ej. `getattr(os, "sys" + "tem")`).
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PYTHON = sys.executable                      # el del venv: trae numpy, sklearn, matplotlib
TIMEOUT_S = 180
SANDBOX_EXEC = shutil.which("sandbox-exec")
RESULTADOS = "resultados.json"               # REGLA 4: nombre fijo

MODULOS_PROHIBIDOS = {
    "socket", "ssl", "http", "urllib", "urllib3", "requests", "httpx", "aiohttp", "ftplib", "smtplib",
    "telnetlib", "paramiko", "subprocess", "multiprocessing", "ctypes", "cffi", "importlib", "pty",
    "signal", "shutil", "webbrowser", "pickle", "dill", "joblib.externals", "asyncio",
}
LLAMADAS_PROHIBIDAS = {"eval", "exec", "compile", "__import__", "breakpoint", "input", "globals", "vars"}
ATRIBUTOS_PROHIBIDOS = {
    "system", "popen", "spawnl", "spawnv", "spawnve", "execv", "execve", "execl", "fork", "forkpty",
    "kill", "killpg", "remove", "unlink", "rmdir", "removedirs", "rmtree", "rename", "replace",
    "chmod", "chown", "chdir", "putenv", "unsetenv", "environ", "getenv", "symlink", "link",
}
DUNDER_PROHIBIDOS = {"__builtins__", "__dict__", "__subclasses__", "__globals__", "__code__",
                     "__getattribute__", "__loader__", "__spec__", "__import__", "__bases__", "__mro__"}
MARCAS_RED = {"fetch_openml", "load_dataset", "urlretrieve", "hf_hub_download", "snapshot_download",
              "download_url", "get_file"}


@dataclass
class Revision:
    rechazos: list[str] = field(default_factory=list)
    requiere_red: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rechazos


def _ruta_fuera(s: str) -> bool:
    return (s.startswith("/") and len(s) > 1 and " " not in s) or s.startswith("~") or "../" in s or s == ".."


def revisar(codigo: str) -> Revision:
    rev = Revision()
    try:
        arbol = ast.parse(codigo)
    except SyntaxError as err:
        rev.rechazos.append(f"no compila: {err.msg} (línea {err.lineno})")
        return rev
    for n in ast.walk(arbol):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            nombres = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            for nombre in nombres:
                raiz = nombre.split(".")[0]
                if raiz in MODULOS_PROHIBIDOS or nombre in MODULOS_PROHIBIDOS:
                    rev.rechazos.append(f"línea {n.lineno}: importa «{nombre}» (red, procesos o borrado)")
            if isinstance(n, ast.ImportFrom) and n.module == "os":
                for a in n.names:
                    if a.name in ATRIBUTOS_PROHIBIDOS:
                        rev.rechazos.append(f"línea {n.lineno}: importa os.{a.name}")
        elif isinstance(n, ast.Call):
            f = n.func
            nombre = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if isinstance(f, ast.Name) and nombre in LLAMADAS_PROHIBIDAS:
                rev.rechazos.append(f"línea {n.lineno}: llama a {nombre}()")
            if nombre in MARCAS_RED or nombre.startswith("fetch_"):
                rev.requiere_red.append(f"línea {n.lineno}: {nombre}()")
            for kw in n.keywords:
                if kw.arg == "download" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    rev.requiere_red.append(f"línea {n.lineno}: {nombre}(download=True)")
            if nombre in ("unlink", "rmdir") and isinstance(f, ast.Attribute):
                rev.rechazos.append(f"línea {n.lineno}: borra archivos ({nombre})")
        elif isinstance(n, ast.Name) and n.id in DUNDER_PROHIBIDOS:
            rev.rechazos.append(f"línea {n.lineno}: usa {n.id} (así se esconde un import)")
        elif isinstance(n, ast.Attribute) and n.attr in DUNDER_PROHIBIDOS:
            rev.rechazos.append(f"línea {n.lineno}: usa .{n.attr} (así se esconde un import)")
        elif isinstance(n, ast.Attribute) and n.attr in ATRIBUTOS_PROHIBIDOS:
            if isinstance(n.value, ast.Name) and n.value.id in ("os", "shutil", "posix", "nt"):
                rev.rechazos.append(f"línea {n.lineno}: usa {n.value.id}.{n.attr}")
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            if n.value.startswith(("http://", "https://", "ftp://")):
                rev.requiere_red.append(f"línea {n.lineno}: URL «{n.value[:60]}»")
            elif _ruta_fuera(n.value):
                rev.rechazos.append(f"línea {n.lineno}: ruta fuera de la carpeta «{n.value[:60]}»")
    rev.rechazos = list(dict.fromkeys(rev.rechazos))
    rev.requiere_red = list(dict.fromkeys(rev.requiere_red))
    return rev


@dataclass
class Ejecucion:
    codigo_salida: int | None
    duracion_s: float
    stdout: str
    stderr: str
    archivos_creados: list[str]
    timeout: bool = False
    con_sandbox_exec: bool = False

    def resumen(self) -> dict:
        d = asdict(self)
        d["stdout"], d["stderr"] = self.stdout[-1500:], self.stderr[-1500:]
        return d


def _perfil(carpeta: Path, con_red: bool) -> str:
    c = str(carpeta.resolve()).replace('"', "")
    reglas = ["(version 1)", "(allow default)", "(deny file-write*)",
              f'(allow file-write* (subpath "{c}") (subpath "/private{c}") (literal "/dev/null") (literal "/dev/tty"))']
    if not con_red:
        reglas.append("(deny network*)")
    return " ".join(reglas)


def _foto(carpeta: Path) -> dict[str, float]:
    fotos = {}
    for p in carpeta.rglob("*"):
        rel = p.relative_to(carpeta)
        if p.is_file() and rel.parts[0] not in (".mpl", "tmp"):
            fotos[str(rel)] = p.stat().st_mtime
    return fotos


def ejecutar(codigo: str, carpeta: Path, timeout: float = TIMEOUT_S, con_red: bool = False,
             nombre: str = "script.py") -> Ejecucion:
    """Corre un script YA revisado. No vuelve a revisarlo: eso lo decide quien llama (el orquestador)."""
    carpeta = Path(carpeta).resolve()
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "tmp").mkdir(exist_ok=True)
    (carpeta / nombre).write_text(codigo, encoding="utf-8")
    antes = _foto(carpeta)
    # REGLA 3: sin variables de entorno del usuario. Solo lo que Python y matplotlib necesitan.
    entorno = {"PATH": f"{Path(PYTHON).parent}:/usr/bin:/bin", "HOME": str(carpeta),
               "TMPDIR": str(carpeta / "tmp"), "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(carpeta / ".mpl"),
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1", "LANG": "en_US.UTF-8",
               "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    orden = [PYTHON, nombre]
    if SANDBOX_EXEC:
        orden = [SANDBOX_EXEC, "-p", _perfil(carpeta, con_red), *orden]
    t0 = time.perf_counter()
    p = subprocess.Popen(orden, cwd=carpeta, env=entorno, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    vencio = False
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        vencio = True
        os.killpg(p.pid, signal.SIGKILL)          # el grupo entero: el script y todos sus hijos
        out, err = p.communicate()
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)      # hijos que hayan quedado vivos tras una salida normal
        except (ProcessLookupError, PermissionError):
            pass
    dur = round(time.perf_counter() - t0, 2)
    stdout, stderr = out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
    (carpeta / "stdout.txt").write_text(stdout, encoding="utf-8")    # artefacto: respalda cifras (regla 5)
    if stderr.strip():
        (carpeta / "stderr.txt").write_text(stderr, encoding="utf-8")
    despues = _foto(carpeta)
    creados = sorted(k for k, v in despues.items() if k not in antes or antes[k] != v
                     if k not in (nombre, "stdout.txt", "stderr.txt"))
    if vencio:
        stderr += f"\n[sandbox] superó {timeout:.0f} s: se mató el grupo de procesos {p.pid}"
    return Ejecucion(None if vencio else p.returncode, dur, stdout, stderr, creados, vencio, bool(SANDBOX_EXEC))


def leer_resultados(carpeta: Path) -> tuple[dict | None, str | None]:
    ruta = Path(carpeta) / RESULTADOS
    if not ruta.exists():
        return None, f"no escribió {RESULTADOS}"
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        return None, f"{RESULTADOS} no es JSON válido: {err}"
    if not isinstance(datos, dict) or not datos:
        return None, f"{RESULTADOS} está vacío o no es un objeto"
    return datos, None
