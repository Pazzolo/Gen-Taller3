#!/bin/zsh
# 2.b: completo y sin_grafo, una tarea por invocación: cada tarea deja su CSV en parciales/,
# así un corte de la VPN no borra las que ya terminaron. Volver a correrlo salta las hechas.
#   zsh correr_2b.sh   (desde mi-solver/)
set -u
BASE=${0:A:h:h}
PY=$BASE/solver-v2/.venv/bin/python
cd $BASE/solver-v2
for v in completo:Solver sin_grafo:SolverSinGrafo; do
  nombre=${v%%:*}; clase=${v##*:}
  out=$BASE/mi-solver/resultados/$nombre
  mkdir -p $out/parciales
  for t in A B C E F; do
    if [[ -f $out/parciales/resultados_solver_$t.csv ]]; then echo "== $nombre $t ya hecha"; continue; fi
    echo "== $nombre $t $(date -u +%H:%M:%S)"
    $PY evaluar_solver.py --golden ../mi-solver/golden_tareas.json --solver solver:$clase \
      --ruta ../mi-solver --corridas $out/corridas --solo $t \
      --salida $out/parciales/resultados_solver_$t.csv
  done
done
echo "== FIN $(date -u +%H:%M:%S)"
