"""Traza JSONL (REGLA 6): un evento por línea, escrito y vaciado al momento.

Por eso se guarda también si algo falla: lo que pasó hasta el error ya está en el disco. Patrón de
`mini-opencode/traza.py`. Cada evento lleva el instante, el tipo y el agente, y según el tipo:

    llm        agente, modelo, tokens_entrada, tokens_salida, latencia_s, fin, error
    ejecucion  subtarea, intento, codigo_salida, duracion_s, archivos_creados, error
    revision   subtarea, intento, aprobado, motivos (de código o del LLM)
    freno      qué freno saltó y por qué
    nodo       entrada/salida de un nodo del orquestador

La traza no guarda variables de entorno ni claves: el LLM local no usa ninguna, y el sandbox
corre los scripts sin entorno (REGLA 3).
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class Traza:
    def __init__(self, ruta: Path | str | None):
        self.ruta = Path(ruta) if ruta else None
        if self.ruta:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.eventos: list[dict] = []
        self._t0 = time.perf_counter()
        self._candado = threading.Lock()      # el indexador llama al LLM en paralelo

    def __call__(self, tipo: str, agente: str = "orquestador", **datos) -> dict:
        ev = {"t": round(time.perf_counter() - self._t0, 3),
              "hora": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "tipo": tipo, "agente": agente, **datos}
        with self._candado:
            self.eventos.append(ev)
            if self.ruta:
                with self.ruta.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        return ev

    def uso_por_agente(self) -> dict[str, dict]:
        """Tokens, llamadas y latencia por agente (para la 2.b y la pregunta 2 de la Parte 5)."""
        salida: dict[str, dict] = {}
        for ev in self.eventos:
            if ev["tipo"] != "llm":
                continue
            a = salida.setdefault(ev["agente"], {"llamadas": 0, "tokens_entrada": 0,
                                                 "tokens_salida": 0, "latencia_s": 0.0})
            a["llamadas"] += 1
            a["tokens_entrada"] += ev.get("tokens_entrada", 0)
            a["tokens_salida"] += ev.get("tokens_salida", 0)
            a["latencia_s"] = round(a["latencia_s"] + ev.get("latencia_s", 0.0), 2)
        return salida
