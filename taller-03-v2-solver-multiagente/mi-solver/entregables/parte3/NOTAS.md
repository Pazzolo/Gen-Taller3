# Parte 3 — Los cuatro frenos, activados (v2)

`frenos_demo.py`, con el modelo de guion (`llm.LLMGuion`): solo cambian las respuestas del LLM; el
orquestador, el sandbox, el revisor y la traza son los de `solver.py`. Cada carpeta tiene la
`traza.jsonl` de la corrida que activó el freno. Resumen: [resumen.json](resumen.json).

| Freno | Dónde | Corrida | Qué pasó |
|---|---|---|---|
| 1. Presupuesto con reserva para el redactor | `llm.Presupuesto.revisar`, antes de cada llamada | [1_presupuesto](1_presupuesto/traza.jsonl): 14 000 tokens, 4 000 de reserva, 1 200 por llamada | indexar y planificar gastan 8 400; el programador hace un intento y el segundo no cabe: «programador: 9 600 gastados + ~1 204 estimados > presupuesto menos la reserva del redactor (10 000)». El solver pasa a redactar, el redactor entra en la reserva y entrega (status `parcial`; T1 y T2 quedan `sin_presupuesto`) |
| 2. Máximo de intentos + repetición | `solver.n_revisar_codigo` (hash de los scripts rechazados), `r_revisar` | [2_intentos](2_intentos/traza.jsonl): el programador entrega siempre el script con fuga para T1 | el script se ejecuta **una vez** y el revisor lo rechaza (`fuga_fit`); los intentos 2 y 3 son idénticos y **no se vuelven a ejecutar** (freno «repeticion»); al tercero, T1 queda `fallida` y el solver sigue con T2, que se aprueba |
| 3. Tiempo máximo del sandbox | `sandbox.ejecutar` (`start_new_session` + `killpg`) | [3_tiempo](3_tiempo/traza.jsonl): un script con un hilo dormido y `sleep(600)`, tiempo máximo de 4 s | `timeout=True`, código de salida `None`, 4,01 s; el revisor lo rechaza y el segundo intento se aprueba. Que mueren también los **procesos hijos** lo prueba `test_solver.py` («freno 3»): un `sleep 300` lanzado por el script ya no existe tras el corte |
| 4. Confirmación humana antes de usar la red | `sandbox.revisar` marca `fetch_*`, `load_dataset`, `download=True` y URLs; `solver.n_aprobar_red` llama a `aprobar_red` | [4_red_negada](4_red_negada/traza.jsonl) · [4_red_aprobada](4_red_aprobada/traza.jsonl) | sin aprobación (`negar_red`, el valor por defecto) el script **no corre** y vuelve al programador con el motivo; con una persona que aprueba, corre con la red habilitada en el perfil del sandbox. En consola se usa `preguntar_en_consola` |

Detalles:
- El presupuesto se estima antes de cada llamada como el tamaño del historial entre 3 caracteres por
  token. La reserva solo la puede gastar el agente `redactor`.
- El freno 4 no rechaza: separa «peligroso» (rechazo automático) de «necesita una persona» (pausa).
  Que la red esté bloqueada de verdad cuando no hay aprobación lo garantiza `sandbox-exec`, no la
  revisión estática (`test_solver.py`: una conexión a 1.1.1.1 da `PermissionError`).
- Un freno más que apareció en las corridas reales: `llm_sin_respuesta`. Una respuesta vacía del LLM
  (el modelo agota `max_tokens` razonando) cuenta como un intento fallido de la subtarea, no como el
  fin de la corrida.
