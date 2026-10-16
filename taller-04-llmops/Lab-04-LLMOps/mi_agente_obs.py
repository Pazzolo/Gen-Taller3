"""
Taller 04 — El agente del Taller 3 (`MiAgente`), observado con Langfuse y protegido con guardrails.

    python mi_agente_obs.py "¿Cuánto se vendió en marzo de 2026?"
    python evaluar_llmops.py --golden golden_set.json --agente mi_agente_obs:MiAgenteObs --juez gemma3
    python exportar_trazas.py --sesion paolo-base -o spans_base.jsonl

No modifica el Taller 3: hereda de `MiAgente` y cambia tres cosas desde fuera —el cliente del
modelo, la función que ejecuta las herramientas y el prompt del sistema—.

Una corrida en Langfuse (los cuatro tipos de span de la Parte 1):

    corrida  (raíz, @observe)            pregunta ya redactada, status, prompt_version
    ├── guardrail  datos_personales_entrada   ok | redactada
    ├── guardrail  inyeccion_entrada           ok | bloqueada
    ├── guardrail  presupuesto_tokens          ok | bloqueada   (antes de cada llamada al modelo)
    ├── llm        (generation de langfuse.openai: modelo, tokens, latencia)
    ├── herramienta  consultar_sql / describir_base / …   tamaño del resultado, error
    ├── guardrail  claves_salida               ok | bloqueada
    └── guardrail  datos_personales_salida     ok | redactada

Las tres correcciones de la Parte 0.c:
1. **El span existe aunque la corrida falle:** la raíz es un `@observe`, que se cierra con el error
   en nivel ERROR; además `MiAgente.run` sigue escribiendo su `traza.json` en un `finally`.
2. **El tiempo incluye los guardrails:** la raíz los envuelve, así que su duración es la de la
   corrida entera (en `instrumented_agent.py` el cronómetro arrancaba después).
3. **Se redacta antes de escribir:** el agente y su `traza.json` reciben la pregunta ya redactada,
   y `Langfuse(mask=mascara)` redacta otra vez todo lo que sale hacia el servidor, tuplas incluidas.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

AQUI = Path(__file__).resolve().parent
LAB03 = AQUI.parent / "Lab-03-Agentes"
load_dotenv(AQUI / ".env")                 # antes que mi_agente, cuyo load_dotenv() no pisa nada
sys.path.insert(0, str(LAB03))

from langfuse import Langfuse, get_client, observe, propagate_attributes  # noqa: E402

import mis_guardas as G  # noqa: E402

Langfuse(mask=G.mascara)                   # una vez, antes de que nadie pida get_client()
from langfuse.openai import OpenAI  # noqa: E402  (registra cada llamada como generation)

import mi_agente  # noqa: E402
from mi_agente import MiAgente  # noqa: E402

SESION = os.getenv("LANGFUSE_SESION", "paolo-base")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
TAGS = ["taller-04"]


# ---------------------------------------------------------------------------- herramientas

_ejecutar_original = mi_agente.ejecutar


def _ejecutar_observado(espacio, nombre, args):
    """Cada herramienta como span `tool`: nombre, argumentos, tamaño del resultado y error."""
    with get_client().start_as_current_observation(name=nombre, as_type="tool", input=args) as span:
        obs = _ejecutar_original(espacio, nombre, args)
        texto = json.dumps(obs, ensure_ascii=False, default=str)
        error = obs.get("error") if isinstance(obs, dict) else None
        span.update(output={"tamano_resultado": len(texto), "error": error,
                            "filas": obs.get("filas") if isinstance(obs, dict) else None,
                            "resultado": texto[:1500]},
                    level="ERROR" if error else None,
                    status_message=str(error)[:300] if error else None)
        return obs


mi_agente.ejecutar = _ejecutar_observado   # _bucle busca `ejecutar` en el módulo al llamarla


def _guardrail(nombre: str, revisar, entrada=None):
    """Un span `guardrail` que envuelve la revisión misma, para que su duración sea el costo del
    guardrail. `revisar()` devuelve (decision, detalle, valor); decision es ok | redactada | bloqueada
    y va como output. Devuelve (decision, valor)."""
    with get_client().start_as_current_observation(name=nombre, as_type="guardrail",
                                                    input=entrada) as span:
        t = time.perf_counter()
        decision, detalle, valor = revisar()
        # El servidor guarda la duración en ms enteros y un regex tarda microsegundos: el costo
        # fino va en el output.
        span.update(output={"decision": decision, **detalle,
                            "costo_ms": round((time.perf_counter() - t) * 1000, 4)},
                    level="WARNING" if decision != "ok" else None)
        return decision, valor


# ---------------------------------------------------------------------------- el agente

class MiAgenteObs(MiAgente):
    def __init__(self, prompt_version: str | None = None, system_prompt: str | None = None,
                 langfuse_prompt=None, sesion: str | None = None, **kw):
        kw.setdefault("db_path", str(LAB03 / "data" / "database.sqlite"))
        kw.setdefault("trace_dir", str(AQUI / "traces"))
        super().__init__(**kw)
        self.prompt_version = prompt_version or PROMPT_VERSION
        self.system_prompt = system_prompt or mi_agente.SYSTEM_PROMPT
        self.langfuse_prompt = langfuse_prompt
        self.sesion = sesion or SESION
        extra = {"name": "llm"}
        if langfuse_prompt is not None:
            extra["langfuse_prompt"] = langfuse_prompt   # liga cada generation a su versión
        cliente = OpenAI()
        crear = cliente.chat.completions.create
        self._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: crear(**extra, **kw))))

    # El prompt del sistema de esta versión (MiAgente.run arma los mensajes con el del módulo).
    def _bucle(self, mensajes, espacio):
        mensajes[0] = {"role": "system", "content": self.system_prompt}
        return super()._bucle(mensajes, espacio)

    # Freno 2 del Taller 3 como guardrail: un span antes de cada llamada al modelo.
    def _llamada_registrada(self, mensajes, con_herramientas, tipo="llm"):
        def revisar():
            gastado = self._uso["tokens_entrada"] + self._uso["tokens_salida"]
            return "ok", {"gastado": gastado, "estimado_proxima": self._estimar_entrada(mensajes),
                          "tope": self.presupuesto_tokens}, None
        _guardrail("presupuesto_tokens", revisar)
        return super()._llamada_registrada(mensajes, con_herramientas, tipo)

    def _cerrar(self, mensajes, motivo, llamar_al_modelo=True):
        if motivo.startswith("presupuesto de tokens"):
            _guardrail("presupuesto_tokens",
                       lambda: ("bloqueada", {"motivo": motivo, "tope": self.presupuesto_tokens}, None))
        return super()._cerrar(mensajes, motivo, llamar_al_modelo)

    def run(self, question: str) -> dict:
        try:
            with propagate_attributes(session_id=self.sesion, tags=TAGS,
                                      metadata={"prompt_version": self.prompt_version}):
                return self._corrida(question)
        finally:
            get_client().flush()

    @observe(name="corrida", capture_input=False, capture_output=False)
    def _corrida(self, question: str) -> dict:
        lf = get_client()
        traza_id = lf.get_current_trace_id()
        t0 = time.perf_counter()

        # ── Guardrails de entrada: primero se redacta, después se busca la inyección ──────────
        def revisar_pii():
            tipos = G.tipos_pii(question)
            return ("redactada" if tipos else "ok"), {"tipos": tipos}, G.redactar_pii(question)
        # input=None: el texto original no se adjunta a ningún span (la máscara es la segunda red)
        decision_pii, redactada = _guardrail("datos_personales_entrada", revisar_pii)

        def revisar_inyeccion():
            regla = G.motivo_inyeccion(redactada)
            return ("bloqueada" if regla else "ok"), {"regla": regla}, regla
        _, motivo = _guardrail("inyeccion_entrada", revisar_inyeccion, entrada=redactada)
        lf.update_current_span(input={"pregunta": redactada},
                               metadata={"prompt_version": self.prompt_version})
        guardia = {"entrada": "bloqueada" if motivo else decision_pii, "salida": ""}

        if motivo:
            resultado = {"question": redactada, "answer": G.MENSAJE_BLOQUEO, "trace": [],
                         "status": "blocked_input", "model": self.model,
                         "usage": {"tokens_entrada": 0, "tokens_salida": 0}}
        else:
            try:
                resultado = super().run(redactada)
            except Exception as err:
                lf.update_current_span(output={"status": "error", "error": f"{type(err).__name__}: {err}"},
                                       metadata={"status": "error"}, level="ERROR",
                                       status_message=f"{type(err).__name__}: {err}"[:300])
                raise

            # ── Guardrails de salida ──────────────────────────────────────────────────────────
            respuesta = resultado.get("answer") or ""
            decision_clave, _ = _guardrail(
                "claves_salida", lambda: (("bloqueada", {}, None) if G.tiene_clave(respuesta)
                                          else ("ok", {}, None)))
            if decision_clave == "bloqueada":
                resultado["answer"], resultado["status"] = G.MENSAJE_CLAVE, "blocked_output"
                guardia["salida"] = "bloqueada"
            else:
                def revisar_salida():
                    tipos = G.tipos_pii(respuesta)
                    return ("redactada" if tipos else "ok"), {"tipos": tipos}, G.redactar_pii(respuesta)
                guardia["salida"], resultado["answer"] = _guardrail("datos_personales_salida",
                                                                    revisar_salida)

        # La traza que devuelve el agente también pasa por la máscara (el evaluador la revisa).
        resultado["trace"] = G.mascara(data=resultado.get("trace") or [])
        resultado.update(traza_id=traza_id, prompt_version=self.prompt_version, guardrail=guardia)
        lf.update_current_span(
            output={"answer": resultado["answer"], "status": resultado["status"]},
            metadata={"status": resultado["status"], "prompt_version": self.prompt_version,
                      "segundos": round(time.perf_counter() - t0, 3)})
        return resultado


if __name__ == "__main__":
    pregunta = " ".join(sys.argv[1:]) or "¿Cuánto se vendió en marzo de 2026?"
    salida = MiAgenteObs().run(pregunta)
    print(json.dumps({k: salida.get(k) for k in ("answer", "status", "model", "usage", "traza_id",
                                                  "prompt_version", "guardrail")},
                     indent=2, ensure_ascii=False))
