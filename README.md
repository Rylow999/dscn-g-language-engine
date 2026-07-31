# DSCN-G Language Engine

Motor de lenguaje experimental sobre DSCN-G: grafo de conceptos + transformer de contexto, en Python puro (sin numpy/torch), orientado a Android.

⚠️ Este proyecto está en fase de investigación. No es un producto terminado.

## Estado honesto (2026-07-30)

### Confirmado
- Contexto genuino: transformer head aprende ~4x el grafo solo (v0.14d, baseline correcto).
- Categorización: geometría omega separa SUST/VERB (v0.9b v2, pureza 0.73 > 0.50).
- Dolor genuino: error next-token real guía aprendizaje (v0.9c limpio).
- Memoria: preservar omega mantiene representación; reintégralo recupera rendimiento (v0.3b v2).
- Polisemia: WSD no supervisado descubre 6/150 palabras con 2 sentidos separables (v0.17).
- Evasión: dolor de consecuencia aleja de lo que lastima (v0.19 v3).
- Atención selectiva: separa A/B en bloques largos (v0.25 v6, acc 0.890).
- Transición explícita: bigramas generan estructura next-token (v0.25 v13, top1 0.850).
- Sentido condicional: modelo por sentido A/B genera coherente (v0.25 v14, pureza 1.000).
- Loop por sentido: clasificador + generador condicional mantiene sentido (v0.25 v15, acc_sense 0.938).
- Memoria competitiva: foco dominante en slot recién disparado (v0.25 v16, coherencia 0.750).
- v21: clasificador lineal + loop sobre embeddings alcanza 1.000 en régimen controlado.

### Parcial / débil
- Memoria de trabajo: foco real (60% dominancia), pero no mejora next-token (v0.24).
- k-means offline sobre Don Quijote señala estructura, pero online no la refine aún.
- v18 “cabo” en Don Quijote: k=3 con clusters diferenciables, desbalanceado.

### Abierto / no funcional
- Loop cerrado total sobre corpus real con generalización robusta.
- Decodificador generativo denso desde embeddings: top1 0.020 (v0.25 v12).
- Composición relacional 3-body con solidez (>0.7): señal <0.04 en corpus real (v0.23 v3).
- Meta/autoobservación y dolor de decisión como disparador de búsqueda.
- Corpus polisémico real etiquetable y estable para entrenamiento supervisado.

## Estructura del repo
- `_README_ENGINE.md`: documentación técnica completa y honesta, con secciones por experimento.
- `dscng_core.py`: clases reutilizables (`Engine`, `Transformer`, `Root`, `Grafo`, `MetricLogger`).
- `test_dscng_core.py`: tests unitarios básicos para `dot`, `cos`, `softmax`, `train_transformer`, `root_refuerza`.
- `run_v25_v2_core.py`: script canónico mínimo importando el core.
- `run_v25_v*.py`: experimentos v0.25 inline.
- `engine_export/`: artifacts listos para sincronización externa.
- `vault/`: `nexus-vault` cuando está accesible.

## Ejecutar
```bash
python3 run_v25_v2_core.py
python3 test_dscng_core.py
```

## Nota metodológica
Los experimentos usan métricas honestas: `acc_pred`, `acc_gt`, `dolor`, `foco_acc`, `W_actual`. No se basa en rewards fijos ni diccionarios en train. En corpus real sin ground truth, se indica explícitamente “no funcional” cuando la señal es artefacto o ruido.
