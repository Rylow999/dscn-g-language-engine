# CHANGELOG — DSCN-G Language Engine

## Fase 1 — Sustrato cognitivo + contexto + DB semántica (COMPLETA)
Estado de cierre: todo lo medible en Python puro (sin numpy/torch) está hecho y
documentado con datos empíricos por experimento.

### Confirmado (empírico)
- v0.3 REAL: memoria masiva persistente (100% retenida, hibernada).
- v0.6a: next-token sobre Don Quijote (10.11%, vocab 150).
- v0.9b: categorización emergente (92.67% sustantivo/verbo sin supervisión).
- v0.9c: dolor interno autopreservación (vitalidad G 0.0 → 1.0).
- v0.10: memoria viva por relevancia (SynapticCache 2.1/2.4/2.5).
- v0.14d: contexto resuelto con backprop manual (10.55% > 10.11% baseline).
- v0.16-bis: DB semántica compositiva (boda={flores,vestido,blanco,beso} jaccard
  1.0; poda por incoherencia desenlaza sin borrar el nodo externo).

### Descartado (honesto, con razón)
- v0.4: β contextual Pandora no aporta (ρ no se activa en D=8).
- v0.7/0.8/0.12/0.13: contexto de 1 capa / atención rústica no desambigua.
- v0.11: abstracción por dimensión se aplana con next-token.
- v0.14/0.14b/0.14c: Hebbiano y backprop sin head aprendido no convergen.
- v0.15: sense nodes no miden polisemia (next-token aplasta sentidos); idea válida
  pero requiere transformer (v0.14d) para resolverse.

### Arquitectura
GRAFO (memoria/categoría/dolor, probado) + TRANSFORMER (contexto, backprop manual)
como capas complementarias. Nodo = ω (geométrico) + referencias (simbólico).

### Tooling
numpy/PyTorch no entran en el entorno (py3.13 aarch64, sin wheels/toolchain).
Backprop manual en Python puro resuelve el contexto.

### Pendiente (fase 2)
- Polisemia estructural (v0.15) sobre transformer completo (v0.14d).
- Abstracción como tamaño del conjunto de referencias (v0.16 extendido).
- v0.15+ (entorno / dolor de consecuencia, gap original #2).
- Repositorio en GitHub: público en Rylow999/dscn-g-language-engine.
