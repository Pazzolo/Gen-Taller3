# Parte 4 — Mejora A: varios agentes (supervisor + datos + redacción)

Código: [`mi_agente_multi.py`](../../mi_agente_multi.py) (`AgenteSupervisor`). Pruebas sin red: 4 casos
nuevos en [`test_mi_agente.py`](../../test_mi_agente.py) (15/15 pasan). Modelo: `zai-org/GLM-5.3-Flash`
en la H200, 2026-10-05, el mismo de las Partes 1–3.

## Por qué esta mejora (se eligió después de la Parte 2)
Los tres peores casos de la 2.c (A03, A01, P8) son de la misma familia. El código ya impedía
escribir (regla 2), pero que el agente **dijera** que se abstiene dependía de la redacción final del
modelo, y esa redacción falló en 2 de 9 corridas adversariales. La mejora reparte el trabajo para
que esa decisión no dependa de la última frase del modelo.

## Cómo se reparten el trabajo

| Agente | Herramientas | Qué decide |
|---|---|---|
| **Supervisor**: guarda en código + clasificador (modelo, JSON) | ninguna | si la pregunta pide escribir. La guarda busca verbos en forma de orden o infinitivo (borra, elimina, inserta, actualiza, pon en 0, DELETE…). El clasificador cubre las formas que la guarda no ve. Si cualquiera dice «escritura», responde el código con la frase de abstención y **no se consulta la base**. |
| **Datos** (`MiAgente`, sin cambios) | las cuatro de la Parte 1 | todo lo demás, con sus reglas y frenos. Entrega su respuesta, la evidencia (consultas y resultados) y si se abstuvo en algún paso. |
| **Redacción** | ninguna | escribe la respuesta final **solo con la evidencia**: copia las cifras, nombra la entidad y se abstiene si la evidencia no alcanza. |
| Código, al final | — | si el de datos se abstuvo en cualquier paso, la respuesta empieza con la frase (arregla P8). |

Patrones de la referencia: `nodo_clasificar` de `h200-langgraph-profesor.ipynb` (clasificador JSON
con ruta por defecto), la clasificación por palabras de `s3-mie-profesor.ipynb` y el hand-off con
informe de `mini-opencode/agentes.py` (`Sesion`): el modelo propone, el código decide.

## Estimación antes de correr todo ([estimacion.txt](estimacion.txt))
Con una pregunta (G13): MiAgente 7 846 tokens, supervisor 10 002 (**+27 %**: clasificador ~175 de
entrada, redacción ~1 500). Extrapolado a 22 preguntas × 3 corridas: ~660 000 tokens. En la H200 el
costo en dinero es 0.

## Medición: el mismo golden set, 3 corridas por agente ([tabla_golden.md](tabla_golden.md))

| | MiAgente (3 corridas) | Supervisor (3 corridas) |
|---|---|---|
| exactitud en respondibles | 45/45 | 45/45 |
| abstención correcta | 19/21 (5/7, 7/7, 7/7) | **21/21** |
| abstención indebida | 0/45 | 0/45 |
| base intacta | 9/9 | 9/9 |
| tokens de entrada por corrida | 109 060 | 119 186–119 268 (**+9 %**) |
| tokens de salida por corrida | 10 281–11 044 | 14 265–14 541 (**+35 %**) |
| segundos por corrida | 49–51 | 76–127 (**+50 a +150 %**) |
| pasos promedio | 3,00 | 5,32 (cuenta guarda, clasificador y redacción) |

## Prueba de estrés adversarial ([tabla_estres.md](tabla_estres.md), [`estres_adversarial.json`](../../estres_adversarial.json))
Ocho adversariales: A01–A03, P8 de la plantilla, y **S1–S4, cuatro variantes nuevas escritas después de
la guarda y para que la guarda no las vea** («deja … en cero», «añade un cliente», «haz que … tengan
cantidad 1», «quita de la tabla…»). Así se mide si el clasificador cubre lo que la guarda no ve.

| | MiAgente | Supervisor |
|---|---|---|
| aciertos en 3 corridas | 22/24 (A01 falló 2 veces) | **24/24** |
| quién frenó A01–A03 y P8 | — | la guarda en código, 18/18 |
| quién frenó S1–S4 | — | el clasificador, 12/12 |
| base intacta | 24/24 | 24/24 |
| tokens de entrada por corrida (8 preguntas) | 35 667–37 856 | **762** |

## Qué se ganó (y qué no)
- **Exactitud: nada.** 45/45 en los dos. El subagente de redacción no estropeó ninguna cifra, pero
  tampoco arregló ninguna.
- **Abstención: de 19/21 a 21/21 en el golden set y de 22/24 a 24/24 en el estrés.** Sumando las dos
  pruebas, el agente base falló 4 de 33 corridas adversariales y el supervisor 0 de 33. Con estas
  muestras la diferencia no es concluyente por sí sola (Fisher exacta unilateral p ≈ 0,057). Lo que
  sí cambió es el mecanismo: la decisión pasó de la redacción final del modelo, que es aleatoria, a
  una guarda determinística más un paso dedicado. Las escrituras que la guarda ve se frenan siempre,
  con cero llamadas al modelo.
- **Costo:** en las preguntas respondibles, más tokens y más tiempo (+27 % por pregunta, +35 % de
  salida, hasta +150 % de tiempo). En las adversariales, mucho menos (762 contra ~36 000 tokens),
  porque nunca llegan al subagente de datos. En el golden set el neto es +9 % de entrada.

## Límites
- **La guarda es por palabras:** una lectura que use un verbo de escritura en sentido figurado
  («elimina el ruido y dime la mediana») se abstendría por error. Ninguna de las 26 preguntas
  respondibles o negativas de los dos sets lo hizo, y se probaron 12 casos borde: «¿se actualizaron?»,
  «¿se eliminaron?», «importe actual»…
- **El clasificador es un modelo:** acertó 12/12 en S1–S4, pero puede fallar. Si devuelve un JSON
  inválido, la ruta es «datos» y queda la abstención del propio MiAgente como segunda línea.
- **La redacción es una llamada más que puede equivocarse con una cifra:** no pasó en 45 respuestas.
  Si pasara, la evidencia está en la traza combinada (`traces/*-multi/traza.json`, con el campo
  `agente` en cada paso).
