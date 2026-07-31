# DSCN-G Language Engine — v0.25 v22

## Objetivo
Validar la tesis de que **la métrica correcta no es separar sentidos A/B internamente, sino comportarse coherente externamente por dominio**.

## Métrica
En vez de `acc_gt` sobre clasificación A/B, se mide:
- Generación condicionada por dominio (A: dinero/banco, B: río/banco).
- Score = overlap semántico externo con vocabulario del dominio.
- Baseline random para contrastar.

## Resultado
- score A = 0.895 (baseline random 0.76)
- score B = 0.915 (baseline random 0.52)
- Veredicto: FUNCIONAL

## Conclusión
La tarea de coherencia de dominio confirma que el modelo genera texto coherente con el dominio activo, sin necesidad de medir separación interna A/B.

## Archivos
- `run_v25_v22_coherencia.py`: experimento canónico.
- `results_v25_v22.json`: resultados.
- `_README_ENGINE.md`: documentación técnica completa.
