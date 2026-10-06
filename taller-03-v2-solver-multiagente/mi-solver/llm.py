"""Cliente de la H200 para todos los agentes: chat, embeddings, presupuesto y traza.

Patrón de `solver-v2/h200.py` (biblioteca estándar, el modelo se lee de `/v1/models`), con tres
cosas más que el solver necesita:

- **`finish_reason=length` con respuesta vacía:** el modelo razona siempre y lo cobra del mismo
  cupo de `max_tokens`. Si se queda sin cupo razonando, `content` llega vacío. Se reintenta con el
  doble de cupo, hasta `MAX_TOKENS_TOPE` (enunciado, «Qué modelo usar»).
- **Freno de presupuesto (Parte 3):** se revisa ANTES de cada llamada. Una parte del presupuesto se
  reserva para el redactor: los demás agentes no pueden gastarla, así que el solver siempre puede
  entregar lo que tiene.
- **Traza (REGLA 6):** cada llamada deja un evento `llm`, también si falla.

`LLMGuion` sustituye las respuestas por una lista fija por agente: sirve para probar el
orquestador y forzar los frenos sin gastar (enunciado, Parte 3).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field

from traza import Traza

HOST = os.environ.get("H200_HOST", "172.28.230.10")
PUERTOS_LLM = [int(p) for p in os.environ.get("H200_PUERTOS", "12555,12559").split(",")]
PUERTO_EMB = int(os.environ.get("H200_PUERTO_EMB", "11434"))
MODELO_EMB = os.environ.get("H200_MODELO_EMB", "bge-m3:latest")
MAX_TOKENS_INICIAL = 16_384
MAX_TOKENS_TOPE = 49_152


class PresupuestoAgotado(Exception):
    """El freno de presupuesto saltó: el orquestador pasa a redactar con lo que tiene."""


@dataclass
class Presupuesto:
    max_tokens: int = 600_000
    reserva_redactor: int = 60_000
    gastado: int = 0
    _candado: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def disponible(self, agente: str) -> int:
        tope = self.max_tokens if agente == "redactor" else self.max_tokens - self.reserva_redactor
        return tope - self.gastado

    def revisar(self, agente: str, estimado: int) -> None:
        if estimado > self.disponible(agente):
            raise PresupuestoAgotado(
                f"{agente}: {self.gastado:,} gastados + ~{estimado:,} estimados > "
                f"{'presupuesto' if agente == 'redactor' else 'presupuesto menos la reserva del redactor'} "
                f"({self.max_tokens if agente == 'redactor' else self.max_tokens - self.reserva_redactor:,})")

    def cargar(self, tokens: int) -> None:
        with self._candado:
            self.gastado += tokens


def _pedir(puerto: int, ruta: str, cuerpo: dict | None = None, timeout: float = 900) -> dict:
    pet = urllib.request.Request(
        f"http://{HOST}:{puerto}/{ruta}",
        data=None if cuerpo is None else json.dumps(cuerpo).encode(),
        # vLLM no valida la clave; «local» no es una credencial
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    try:
        with urllib.request.urlopen(pet, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"La H200 respondió {err.code}: {err.read()[:300]!r}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Sin respuesta de {HOST}:{puerto} ({err.reason}). ¿Está GlobalProtect conectada?") from err


def extraer_json(texto: str) -> dict | list:
    """El JSON de una respuesta, aunque venga dentro de ```json … ``` o con texto alrededor."""
    m = re.search(r"```(?:json)?\s*(.*?)```", texto, re.S)
    candidato = m.group(1) if m else texto
    try:
        return json.loads(candidato)
    except json.JSONDecodeError:
        i, j = candidato.find("{"), candidato.rfind("}")
        if i >= 0 and j > i:
            return json.loads(candidato[i:j + 1])
        raise


class LLM:
    def __init__(self, traza: Traza, presupuesto: Presupuesto | None = None, puerto: int | None = None):
        self.traza = traza
        self.presupuesto = presupuesto or Presupuesto()
        self.puerto = puerto or PUERTOS_LLM[0]
        self.modelo = self._leer_modelo()

    def _leer_modelo(self) -> str:
        return _pedir(self.puerto, "v1/models", timeout=8)["data"][0]["id"]

    def _llamar(self, mensajes: list[dict], json_mode: bool, max_tokens: int) -> dict:
        cuerpo = {"model": self.modelo, "messages": mensajes, "temperature": 0,
                  "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": True}}
        if json_mode:
            cuerpo["response_format"] = {"type": "json_object"}
        r = _pedir(self.puerto, "v1/chat/completions", cuerpo)
        m = r["choices"][0]["message"]
        uso = r.get("usage", {})
        return {"contenido": m.get("content") or "", "fin": r["choices"][0].get("finish_reason"),
                "razonamiento": len(m.get("reasoning") or m.get("reasoning_content") or ""),
                "tokens_entrada": uso.get("prompt_tokens", 0), "tokens_salida": uso.get("completion_tokens", 0)}

    def chat(self, agente: str, mensajes: list[dict], json_mode: bool = False,
             max_tokens: int = MAX_TOKENS_INICIAL) -> str:
        """Una respuesta no vacía, o una excepción. Reintenta si el razonamiento agotó el cupo."""
        while True:
            estimado = len(json.dumps(mensajes, ensure_ascii=False)) // 3
            try:
                self.presupuesto.revisar(agente, estimado)
            except PresupuestoAgotado as err:
                self.traza("freno", agente, freno="presupuesto", motivo=str(err))
                raise
            t0 = time.perf_counter()
            try:
                r = self._llamar(mensajes, json_mode, max_tokens)
            except Exception as err:
                self.traza("llm", agente, modelo=self.modelo, tokens_entrada=0, tokens_salida=0,
                           latencia_s=round(time.perf_counter() - t0, 2), fin=None,
                           error=f"{type(err).__name__}: {err}"[:400])
                raise
            self.presupuesto.cargar(r["tokens_entrada"] + r["tokens_salida"])
            vacia = not r["contenido"].strip()
            self.traza("llm", agente, modelo=self.modelo, tokens_entrada=r["tokens_entrada"],
                       tokens_salida=r["tokens_salida"], latencia_s=round(time.perf_counter() - t0, 2),
                       fin=r["fin"], razonamiento_car=r["razonamiento"], max_tokens=max_tokens,
                       error="respuesta vacía por fin=length" if vacia and r["fin"] == "length" else
                       ("respuesta vacía" if vacia else None))
            if not vacia:
                return r["contenido"]
            if r["fin"] == "length" and max_tokens < MAX_TOKENS_TOPE:
                max_tokens = min(max_tokens * 2, MAX_TOKENS_TOPE)   # reintento con más cupo
                continue
            raise RuntimeError(f"{agente}: el modelo devolvió una respuesta vacía (fin={r['fin']}, max_tokens={max_tokens})")

    def json(self, agente: str, mensajes: list[dict], intentos: int = 2) -> dict:
        """Una respuesta JSON; si no se puede leer, se pide de nuevo diciendo por qué."""
        ultimo = ""
        for _ in range(intentos):
            texto = self.chat(agente, mensajes, json_mode=True)
            try:
                return extraer_json(texto)
            except (json.JSONDecodeError, ValueError) as err:
                ultimo = f"{err}"
                mensajes = mensajes + [{"role": "assistant", "content": texto[:2000]},
                                       {"role": "user", "content": f"Eso no es un JSON válido ({err}). Responde solo el JSON."}]
        raise RuntimeError(f"{agente}: el modelo no devolvió un JSON válido ({ultimo})")

    def embed(self, textos: list[str], lote: int = 32) -> list[list[float]]:
        """Embeddings bge-m3 normalizados (Ollama de la H200, puerto 11434)."""
        salida: list[list[float]] = []
        for i in range(0, len(textos), lote):
            t0 = time.perf_counter()
            r = _pedir(PUERTO_EMB, "v1/embeddings", {"model": MODELO_EMB, "input": textos[i:i + lote]})
            for d in sorted(r["data"], key=lambda d: d["index"]):
                v = d["embedding"]
                n = sum(x * x for x in v) ** 0.5 or 1.0
                salida.append([x / n for x in v])
            self.traza("embedding", "indexador", modelo=MODELO_EMB, textos=len(textos[i:i + lote]),
                       latencia_s=round(time.perf_counter() - t0, 2))
        return salida


class LLMGuion(LLM):
    """Mismo contrato, respuestas fijas por agente. `guion[agente]` es una lista o una función(mensajes)."""

    def __init__(self, traza: Traza, guion: dict, presupuesto: Presupuesto | None = None,
                 tokens: tuple[int, int] = (1000, 200)):
        self.traza, self.presupuesto, self.puerto = traza, presupuesto or Presupuesto(), 0
        self.modelo, self._guion, self._tokens = "guion", {k: (v if callable(v) else list(v)) for k, v in guion.items()}, tokens
        self._agente = ""

    def chat(self, agente: str, mensajes: list[dict], json_mode: bool = False,
             max_tokens: int = MAX_TOKENS_INICIAL) -> str:
        self._agente = agente
        return super().chat(agente, mensajes, json_mode, max_tokens)

    def _llamar(self, mensajes, json_mode, max_tokens):
        g = self._guion.get(self._agente)
        if g is None:
            raise RuntimeError(f"el guion no tiene respuestas para el agente «{self._agente}»")
        salida = g(mensajes) if callable(g) else (g.pop(0) if g else "")
        if not isinstance(salida, str):
            salida = json.dumps(salida, ensure_ascii=False)
        return {"contenido": salida, "fin": "stop", "razonamiento": 0,
                "tokens_entrada": self._tokens[0], "tokens_salida": self._tokens[1]}

    def embed(self, textos, lote=32):
        # Embeddings deterministas y baratos para pruebas: bolsa de palabras con hash.
        salida = []
        for t in textos:
            v = [0.0] * 64
            for w in re.findall(r"\w+", t.lower()):
                v[zlib.crc32(w.encode()) % 64] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            salida.append([x / n for x in v])
        return salida
