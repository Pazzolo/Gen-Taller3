# Parte 1 — arquitectura y notas para el informe

Código: [`mi_agente.py`](../../mi_agente.py) (bucle, frenos, traza) · [`mis_tools.py`](../../mis_tools.py)
(herramientas, guarda, validación) · [`modelo_guion.py`](../../modelo_guion.py) (modelo sin LLM) ·
[`test_mi_agente.py`](../../test_mi_agente.py) (11 pruebas sin red, todas pasan).

Base de referencia del curso: el bucle sigue `Agente.correr` de `mini-opencode/agentes.py` (los
frenos dentro del bucle, el razonamiento fuera del historial) y el ejercicio 3.1 de
`s3-lun-profesor.ipynb`. El formato de los esquemas viene de `a_formato_openai` (`s3-lun`), y la
guarda en código, de `mini-opencode/guardia.py`.

## Herramientas (REGLA 1)

| Nombre | Qué hace | Esquema de entrada |
|---|---|---|
| `describir_base` | tablas, columnas, valores de las categóricas, rangos y notas del dominio (fórmula del importe, qué datos NO hay) | `{}` |
| `consultar_sql` | una consulta de solo lectura → guarda el dataset, devuelve `dataset_id` + vista de 20 filas | `{consulta: string}` (obligatorio) |
| `estadisticas` | n, media, mediana, desviación estándar, mín., Q1, Q3, máx. y suma de un dataset | `{dataset_id: string, columnas?: string[]}` |
| `grafico` | PNG de línea, barras o dispersión de un dataset | `{dataset_id, x, y: string, tipo?: line\|bar\|scatter, titulo?: string}` |

Todos los esquemas llevan `additionalProperties: false` y se pasan con el parámetro `tools` de la
API (`mi_agente.py:108`).

## Dónde está cada regla

| Regla | Dónde | Cómo |
|---|---|---|
| 1. El modelo conoce sus herramientas | `mis_tools.py:253` y siguientes, `mi_agente.py:108` | nombre + descripción + esquema JSON por `tools=`; `describir_base` añade lo que el esquema de la base no dice |
| 2. Solo lectura de verdad | `mis_tools.py:103-131` | `mode=ro` + `PRAGMA query_only`; se quitan los comentarios y se rechaza cualquier `;` restante; solo `SELECT`/`WITH` |
| 3. Los límites los pone el código | `mis_tools.py:47-53`, `:92`, `:167`, `:178` | 10 000 filas por dataset, 20 en la vista y 5 s por consulta; carpeta y nombre de gráfico generados |
| 4. Las herramientas se pasan datos | `mis_tools.py:67-97` | `consultar_sql` guarda `datos/dsN.csv` en la carpeta de la corrida; las otras dos reciben el `dataset_id` |
| 5. Un error no termina la corrida | `mis_tools.py:319` y siguientes, `mi_agente.py:202` | validación contra el esquema antes de ejecutar; JSON inválido, herramienta inexistente o excepción → `{"error": …}` |
| 6. Siempre se guarda la traza | `mi_agente.py:158` (`finally`), `:248` y siguientes | por paso: modelo, tokens de entrada y salida, latencia y error; totales en `traza.json` |

Abstención: el prompt exige la frase de `evaluar.ABSTENCION`, importada (no copiada) en
`mi_agente.py`. Así, si cambia la frase, cambia en los dos lados.

## Justificación de la regla 4 (borrador)
Se eligió un dataset por id en lugar de pasar rutas o filas:
- el modelo nunca ve ni elige una ruta (cierra el `output_path` de la 0.c);
- al historial solo vuelve una vista de 20 filas, y el dataset completo queda en disco (en la 0.c
  volvían 842 321 caracteres en cada paso; con `LIMIT 10000` ahora vuelven ~3 000);
- `estadisticas` y `grafico` trabajan sobre el resultado exacto de la consulta, no sobre un CSV
  aparte que podría no coincidir (el desajuste SQLite/CSV del andamiaje);
- cada corrida tiene su carpeta (`traces/<run_id>/datos/`), así que los datasets quedan como
  evidencia junto a la traza.

Costo: un CSV por consulta en disco, y el id solo vale dentro de la corrida.

## Diagrama del bucle y condición de parada

```
pregunta ─► [system + user]
   ┌───────────────────────────────────────────────────────────────────────┐
   │ ¿gastado + entrada estimada > presupuesto? ── sí ─► cierre sin modelo  │ freno 2 (budget_exceeded)
   │        │ no                                                            │
   │        ▼                                                               │
   │   modelo (con tools=)  ── sin tool_calls ─► respuesta final            │ parada normal (completed)
   │        │ tool_calls                                                    │
   │        ▼                                                               │
   │   ¿misma llamada > k veces? ── sí ─► cierre: responde con lo que tiene │ freno 3 (repetition_detected)
   │        │ no                                                            │
   │        ▼                                                               │
   │   validar + ejecutar ─► observación o {"error"} ─► mensaje role=tool   │
   └──────────── paso < max_pasos ◄─────────────────────────────────────────┘
          paso = max_pasos ─► cierre: responde con lo que tiene           freno 1 (max_steps_reached)
```

Un cierre por freno le pide al modelo, sin herramientas, que responda con lo que sabe, y antepone
`[INCOMPLETO — motivo]`. La respuesta nunca queda vacía.

## Nota para la 2.b
En la traza de MiAgente, un paso es una llamada al modelo, incluida la de la respuesta final. La
traza del andamiaje solo registra los pasos con herramienta. Por eso `pasos` del CSV cuenta una
llamada más en MiAgente que en el andamiaje para el mismo trabajo, y hay que decirlo al comparar.

## Pendiente con VPN
`python correr_con_vpn.py parte1` hace dos corridas reales (una usa las tres herramientas) y las
copia en `entregables/parte1/corrida_N/`.
