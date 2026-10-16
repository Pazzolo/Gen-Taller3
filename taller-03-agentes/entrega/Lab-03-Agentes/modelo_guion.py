"""
Taller 03 — Un modelo de guion para MiAgente: sin LLM, sin VPN, sin costo.

La misma idea que `guion.py` (Parte 0) y `LLMSimulado` de `s3-lun-profesor.ipynb`: se sustituye
SOLO `_llamar_modelo` por una lista fija de salidas con la forma que devuelve el modelo real.
El bucle, las herramientas, los frenos y la traza son los de `mi_agente.py` sin tocar.
Lo usan `test_mi_agente.py` y la demostración de los frenos (Parte 3).
"""

from __future__ import annotations

import json
from typing import Callable

from mi_agente import MiAgente


def herramienta(nombre: str, **args) -> dict:
    """Una salida del «modelo» que pide una herramienta."""
    return {"tool_calls": [{"name": nombre, "arguments": json.dumps(args, ensure_ascii=False)}]}


def crudo(nombre: str, argumentos: str) -> dict:
    """Una petición con los argumentos tal cual (para probar JSON inválido)."""
    return {"tool_calls": [{"name": nombre, "arguments": argumentos}]}


def final(texto: str) -> dict:
    return {"contenido": texto}


class AgenteDeGuion(MiAgente):
    """MiAgente con el modelo sustituido por un guion. `guion` es una lista o una función(paso)."""

    def __init__(self, guion: list[dict] | Callable[[int], dict], tokens_por_llamada=(900, 120), **kw):
        super().__init__(model="guion", **kw)
        self._guion = guion if callable(guion) else list(guion)
        self._tokens = tokens_por_llamada
        self._n = 0

    def _llamar_modelo(self, mensajes, con_herramientas=True):
        self._n += 1
        if callable(self._guion):
            salida = self._guion(self._n)
        else:
            salida = self._guion.pop(0) if self._guion else final("fin del guion")
        if not con_herramientas:  # el cierre de un freno no ofrece herramientas
            salida = final("Con lo que alcancé a consultar no tengo la respuesta completa.")
        if isinstance(salida, Exception):
            raise salida
        entrada = self._tokens[0] + 40 * len(mensajes)  # crece con el historial, como uno real
        return {
            "contenido": salida.get("contenido", ""),
            "tool_calls": [{"id": f"call_{self._n}_{i}", **tc}
                           for i, tc in enumerate(salida.get("tool_calls", []))],
            "razonamiento": "", "fin": "tool_calls" if salida.get("tool_calls") else "stop",
            "tokens_entrada": entrada, "tokens_salida": self._tokens[1], "modelo_servido": "guion",
        }
