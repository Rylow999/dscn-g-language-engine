# DSCN-G Language Engine — Estado honesto

> Motor de lenguaje sobre DSCN-G: grafo de conceptos + transformer de contexto, Python puro (sin numpy/torch). Experimental, no producto.

## Estado consolidado (2026-07-30)

### Confirmado (señal real del dato)
- Contexto: transformer separa ~4x el grafo solo (v0.14d).
- Categorización: geometría omega separa SUST/VERB (v0.9b, pureza 0.73).
- Dolor: error next-token real guía aprendizaje (v0.9c limpio).
- Memoria: preservar/reintegrar omega mantiene rendimiento (v0.3b v2).
- Polisemia: 6/150 palabras con 2 sentidos separables (v0.17).
- Evasión: dolor aleja de lo que lastima (v0.19 v3).
- Atención selectiva: separa A/B en bloques largos (v0.25 v6).
- Transición explícita: top1=0.850 (v0.25 v13).
- Sentido A/B: pureza=1.000 (v0.25 v14).
- Loop por sentido: acc_sense=0.938 (v0.25 v15).
- Memoria competitiva: coherencia 0.750 (v0.25 v16).
- Loop clasificador: 1.000 sobre corpus sintético (v0.25 v21).
- Coherencia de dominio: FUNCIONAL sobre corpus sintético (v0.25 v22, score 0.895-0.915).

### Parcial / débil
- v18 DQ "cabo": FUNCIONAL PARCIAL (k=3, tamaños 61/22/7, cohesión 0.42-0.62).
- v0.24 vitalidad: foco real (60% dominancia) pero no mejora next-token.

### No funcional / gap abierto
- Loop cerrado sobre corpus real: colapsa (v0.25 v8/v8b/v10).
- Decoder embeddings: top1=0.020 (v0.25 v12).
- Hebb 3-body: 0.036 vs 0.026 (v0.23 v3).
- Coherencia sobre corpus real Don Quijote: NO FUNCIONAL (v0.25 v22b, score=0.0).
- v3-v6, v12 sobre core: NO FUNCIONAL.
- v7/v7b/v7c: scripts no recuperados (solo JSON).

## Arquitectura
- `dscng_core.py`: core reutilizable (SimpleTransformer, RootMemory, LinearSenseClassifier, SkipGram, MetricLogger, build_polysemy_corpus).
- `test_dscng_core.py`: tests unitarios.
- `run_v25_v2_core.py`: script canónico mínimo.
- `run_v25_v*_core.py`: experimentos revalidados sobre core.

## Métrica canónica
acc_pred, acc_gt, dolor, foco_acc, W_actual. La coherencia de dominio (v22) supera a acc_gt para validar comportamiento externo.

## Próximo
- Corpus real polisémico etiquetable (no Don Quijote).
- W(t) dinámica por dolor (Ec.8).
- Decodificador generativo con fase phi real (von Mises).
- Integrar coherencia como feedback en loop cerrado.
