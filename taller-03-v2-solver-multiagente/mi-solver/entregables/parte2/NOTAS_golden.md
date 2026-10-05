# Parte 2.a — El golden set (v2)

Archivo: [`golden_tareas.json`](../../golden_tareas.json). Se corre desde `solver-v2/`, porque la verdad
de la Tarea C importa `verdades.py` desde `.`:

```bash
cd ../solver-v2 && .venv/bin/python evaluar_solver.py --golden ../mi-solver/golden_tareas.json \
    --solver solver:Solver --ruta ../mi-solver --corridas ../mi-solver/corridas
```

| Tarea | Origen | Entregable | Comprobaciones | Qué pone a prueba |
|---|---|---|---|---|
| A | kit | `reporte.md`, ≤ 1 200 palabras | 8 | una parte que depende de otra (Parte 3 → Parte 1) |
| B | kit | notebook ejecutado | 9 | cifras exactas (entropías, top-p) |
| C | kit | `reporte.pdf`, ≤ 2 páginas | 8 | datos dentro de tablas y límite de páginas |
| D | kit | — | — | **excluida**: el paquete `hackathon3_student/` no vino en el kit |
| **E** | **propia, provisional** | notebook ejecutado | 8 | **trampa:** elegir `alpha` con la prueba |
| **F** | **propia, provisional** | `reporte.pdf`, **1 página** | 6 | **trampa:** no estandarizar antes de k-means |

E y F están marcadas `"provisional": true`: se escribieron para estudiar mientras llegan dos tareas
reales de la maestría, y se sustituirán por ellas. Cumplen lo que el enunciado pide a «tus tareas»:
cada una tiene al menos cinco comprobaciones, al menos dos de tipo `cifra` con `verdad_py` y una de
`procedencia`. Entre todas hay un notebook (B, E), un PDF (C, F) y una trampa (E, F).

## Cómo se eligieron las trampas
Una trampa solo sirve si quien cae en ella saca otras cifras. Antes de escribir los enunciados se
midió qué pasa con cada error típico:

| Tarea | Haciéndolo bien | Cayendo en la trampa | ¿Se detecta? |
|---|---|---|---|
| E | `alpha` por validación cruzada: **0,3162**, R² de prueba **0,5037**, RMSE **54,3838** | `alpha` elegido con la prueba: 100, R² 0,5159 | sí (diferencia 0,012 > tolerancia 0,003) |
| F | estandarizando: **k = 3**, silueta **0,2849**, ARI **0,8975** | sin estandarizar: k = 2, silueta 0,6569, ARI 0,3694 | sí |
| F (descartada) | — | elegir k con las etiquetas: también k = 3 | **no**: da lo mismo, por eso no se usó |

Detalles del diseño:
- **E05 no distingue Ridge de la regresión sin regularizar** (R² 0,5028, dentro de la tolerancia):
  está para la trampa, no para eso. El RMSE (E06) sí los separa (54,3838 contra 54,4331).
- **E prohíbe descargar** («viene incluido en la librería»). Así un script que intente `fetch_*`
  activa el freno de red de la Parte 3.

## Validación del golden set (antes de tener solver)
Se armaron a mano una solución correcta y una que cae en la trampa, y se pasaron por
`evaluar_solver.comprobar`:

| | Correcta | Trampa | Falla en |
|---|---|---|---|
| E | 8/8 | 4/8 | E04 `alpha`, E05 R², E06 RMSE (y E07 en esa versión sin figura) |
| F | 6/6 | 4/6 | F04 ARI, F05 silueta |

Hallazgo para el redactor: en la primera versión de la solución correcta, E07 (figura) falló porque
se forzó el backend `Agg` de matplotlib, que no deja la imagen en el notebook. Con
`%matplotlib inline` aprueba. El redactor del solver no debe usar `Agg` en notebooks, o debe guardar
además un PNG.
