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

## Fase 2 — AUDITORÍA HONESTA + v0.17 → v0.25 (2026-07-28)
ESTADO REVISADO: los "✓ confirmados" de Fase 1 eran en parte ARTEFACTOS DE
DISEÑO (señal circular), no validación real. Se re-ejecutó con señal del dato
(SIN reward fijo / SIN dict en train / SIN corpus armado). Ver README.md (tabla
de CORRECCIÓN) para el detalle de cada experimento circular.

### Correcciones de auditoría (señal real del dato)
- v0.9c: reward FIJO empujaba omega_ideal -> G=1.0 por construcción (CIRCULAR).
  Limpio: dolor = error next-token real; B aprende 0.9927->0.933. DOLOR GENUINO.
- v0.9b: dict SUST/VERB en train (CIRCULAR). v2 con vocab 50/50: pureza 0.73.
- v0.16-bis: corpus sintético para jaccard=1.0 (CIRCULAR). v0.3b v2: hibernar
  reintegra ~0.98, borrar mata 0.0. MEMORIA REAL (no identidad matemática).
- v0.14d: comparaba corpus distinto (INVÁLIDO). Audit: base 0.0237, híbrido 0.0958.

### Nuevos experimentos (v0.17 → v0.25) — datos empíricos
- v0.17 polisemia: 6/150 palabras con 2 sentidos separables (cos<0.5). GENUINA.
- v0.19 dolor de consecuencia: aff(A,B) 0.94 -> -0.47 tras dolor. EVASION GENUINA.
- v0.21 v8 grafo fractal ANCLA+REPULSIÓN (fix oversmoothing): Don Quijote 39/40
  sentidos separados SIN transformer. La regla de update (no el sustrato) sostiene.
- v0.22 ROOT DIRECTOR + proyección Hebb: ruteo FASE A = 1.0. DUDA no emerge
  (grafo separa sentidos tan bien que siempre hay claro ganador; no es bug).
- v0.23 composición relacional Hebb 3-body: v2 = 0.312 (azar 0.25); v3 datos
  reales Don Quijote = 0.042 (azar 0.011). GAP ABIERTO (señal débil, 89 rels).
- v0.24 memoria trabajo VITALIDAD competitiva: foco dominado 0.601 (60%). La
  vitalidad NO ayuda next-token (0.038 < 0.095 sin ella). Foco = atención real.
- v0.25 harness INTEGRACIÓN (ciclo 12 pasos NOUS v4): bloques SE COMPONEN en ciclo
  cerrado (banco->dinero/rio resuelto en ambas frases, corpus mini). 1er intento.

### Mapa de gaps hacia pseudoAGI
CONFIRMADO: polisemia (v0.21 v8), ruteo (v0.22 v3), memoria (v0.3b v2), memoria
de trabajo/foco (v0.24), ajuste por dolor (v0.9c/v0.19).
DÉBIL/ABIERTO: composición relacional (v0.23, señal 4x azar pero ruidosa).
NO INTEGRADO (muro real): loop cerrado a escala (v0.25 solo mini), decodificador
generativo, decisión sobre foco, duda de decisión.
PRÓXIMO: v0.25 v2 = integrar sobre Don Quijote real (grafo fractal v0.21 v8), fase
φ real para von Mises, decodificador generativo, y forzar dolor para ver ventana
contrarse (Ec.8 NOUS v4).
