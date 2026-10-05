# Parte 3 — los tres frenos, activados

Generado con [`frenos_demo.py`](../../frenos_demo.py) (`python frenos_demo.py --real`) el 2026-10-05.
Salida completa: [resumen.txt](resumen.txt). Cada carpeta tiene la `traza.json` de la corrida que
activó el freno. Las demostraciones con guion solo sustituyen `_llamar_modelo`
([`modelo_guion.py`](../../modelo_guion.py)), igual que `guion.py` en la Parte 0. El bucle y los
frenos son los de `mi_agente.py` sin cambios. Las dos `real_*` usan `zai-org/GLM-5.3-Flash` en la H200.

Referencia del curso: los tres frenos dentro del bucle, como `Agente.correr` en
`mini-opencode/agentes.py`; el presupuesto antes de gastar, como `ejecutar_con_presupuesto`, y la
normalización de acciones, como `normalizar_accion` / `detectar_loop` en `s3-mar-profesor.ipynb`.

| Freno | Código | Demostración | Resultado |
|---|---|---|---|
| 1. Máximo de pasos (8 por defecto) | `mi_agente.py` «Freno 1», `_cerrar()` | [1_max_pasos](1_max_pasos/traza.json): consultas siempre distintas, `max_pasos=4` · [real_1_max_pasos](real_1_max_pasos/traza.json): G13 con `max_pasos=2` | `max_steps_reached`. Una última llamada **sin herramientas** responde con lo que hay y lleva `[INCOMPLETO — límite de N pasos alcanzado]`. En la real, el modelo ya tenía la mediana y la dio (650 732,53), marcada como incompleta. |
| 2. Presupuesto de tokens (120 000 por corrida, entrada + salida) | «Freno 2», `_estimar_entrada()` | [2_presupuesto](2_presupuesto/traza.json): 12 000 · [real_2_presupuesto](real_2_presupuesto/traza.json): 3 000 | `budget_exceeded` **antes** de llamar: «1 186 gastados + ~1 927 de la próxima llamada > 3 000». Gasto final ≤ presupuesto (10 000 ≤ 12 000 con guion). Sin presupuesto no se hace la llamada de cierre: la respuesta lo dice y se abstiene. |
| 3. Detector de repetición (k = 2) | «Freno 3», `_clave_llamada()` | [3_repeticion](3_repeticion/traza.json): `SELECT COUNT(*) FROM ventas` escrita 3 veces con distintos espacios, mayúsculas y `;` | `repetition_detected` a la 3.ª. La 2.ª no se ejecuta: el modelo recibe `{"repetida": …, "resultado_anterior": …}`. |

## Cómo se estima la próxima llamada (freno 2)
Entrada estimada = `prompt_tokens` reales de la última llamada + (caracteres de los mensajes
añadidos desde entonces) / 3. La primera llamada se estima con todo el prompt más los esquemas. El
freno corta si `gastado + estimado > presupuesto`. Se revisa antes de llamar porque el historial se
reenvía completo en cada paso: el paso *n* paga de nuevo todo lo anterior, así que el gasto
acumulado crece con el cuadrado de los pasos. En la corrida real: 1 139 → 1 789 tokens de entrada
por llamada, 2 928 acumulados en dos pasos. El cierre cuesta menos (1 445) porque se manda sin los
esquemas de las herramientas.

## Qué bucles NO detecta el freno 3
La clave es (herramienta, argumentos en forma canónica): mismo orden de claves, y en el SQL se
quitan los espacios repetidos, las mayúsculas y el `;` final. Cuenta apariciones totales, no
consecutivas, así que **sí** detecta A, B, A, B, A. **No** detecta:
- **La misma pregunta escrita con SQL distinto.** Demostración
  [4_bucle_no_detectado](4_bucle_no_detectado/traza.json): cinco formas de contar las ventas de marzo
  de 2026 (alias, `LIKE` contra `BETWEEN`, `substr`, `COUNT(venta_id)`, un `LIMIT 1` de más). Para el
  detector son cinco llamadas diferentes. La paró el freno 1, no el 3.
- **El bucle sin progreso con argumentos que cambian a propósito:** «consulta el mes 1, el 2, el 3…»
  (demo 1). Cada llamada es nueva y legítima por separado; solo el máximo de pasos o el presupuesto
  lo cortan.
- **El bucle sin herramientas:** el modelo que razona o pide aclaraciones sin llamar a nada. No
  hay llamadas que comparar. (Las respuestas vacías tienen su propio tope: dos y se cierra.)

Por eso los tres frenos se complementan: el 3 es barato y preciso pero solo ve copias exactas; el
1 y el 2 no saben qué está pasando pero siempre terminan la corrida.

## Cuarto freno
No aplica: el agente no puede escribir (regla 2: `mode=ro` + `query_only`), así que no hay ninguna
acción irreversible que confirmar.
