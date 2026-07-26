# DSCN-G Language Engine — Documentación completa (fase 1 → v0.16)

## Qué es
Motor de lenguaje basado en DSCN-G (Dual-State Cognitive Geometry) como sustrato
cognitivo: no un chatbot que predice palabras, sino un sistema con memoria que no
se borra, que categoriza lo que procesa, y que siente un dolor que lo hace
corregirse para sobrevivir. Todo implementado en Python puro (sin numpy/torch) en
un entorno Android, con datos empíricos por cada experimento.

## Experimentos (v0.1 → v0.16) — tabla de verdad
| Exp | Qué prueba | Resultado | Veredicto |
|-----|-----------|-----------|-----------|
| v0.1 | grafo de conceptos + afinidad | masa estable | ✓ base |
| v0.2 | poda homeostática | decae lo no usado | ✓ |
| v0.3 REAL | memoria masiva (hibernado) | 100% retenido | ✓ CONFIRMADO |
| v0.4 | β contextual Pandora | 5.0 vs 5.2 (ruido) | ✗ no aporta |
| v0.5/0.6a | decoder + next-token Don Quijote | 10.11% | ✓ aprende |
| v0.6b-bis | dolor Q-learning | redundante | ✗ |
| v0.7/0.8 | contexto rústico | no desambigua | ✗ 1 capa insuficiente |
| v0.9b | categorización emergente | 92.67% | ✓ CONFIRMADO |
| v0.9c | dolor interno autopreservación | G 0→1 | ✓ CONFIRMADO |
| v0.10 | memoria viva (SynapticCache) | masa activa por relevancia | ✓ |
| v0.11 | abstracción por dimensión | next-token aplana | ✗ |
| v0.12 | atención real sintética | no ayuda | ✗ |
| v0.13/0.13-bis | híbrido 1 capa | colapsa/10.11% | ✗ |
| v0.14 | híbrido Hebbiano 2 capas | 1.97% | ✗ Hebbiano no entrena |
| v0.14b/c | backprop manual D=8/16 | 0.12% (piso uniforme) | ~ backprop anda, no converge |
| v0.14d | backprop head APRENDIDO | 10.55% > 10.11% | ✓ CONTEXTO RESUELTO |
| v0.15/0.15-bis | sense nodes (polisemia) | 0.50 (azar) | ✗ next-token aplasta sentidos |
| v0.16/0.16-bis | referencias compositivas | jaccard 1.0, poda respeta | ✓ IDEA 2 CONFIRMADA |

## Confirmado empíricamente
- Memoria masiva persistente (v0.3): 100% de masa retenida, hibernada.
- Categorización emergente (v0.9b): 92.67% deduciendo sustantivo/verbo del uso.
- Dolor interno (v0.9c): vitalidad G 0.0 → 1.0 bajo corrección; autopreservación.
- Next-token (v0.6a): 10.11% sobre Don Quijote (vocab 150).
- Contexto (v0.14d): 10.55% con backprop manual (transformer 1 capa, head aprendido).
- Composición / DB semántica (v0.16-bis): "boda"={flores,vestido,blanco,beso} jaccard
  1.0; poda por incoherencia desenlaza pero NO borra el nodo externo.

## Límites (honestos)
- Polisemia estructural (v0.15): el next-token aplasta los sense-ω. Idea VÁLIDA
  pero requiere transformer con backprop (v0.14d) para resolverse. Pendiente.
- Abstracción por dimensión (v0.11): next-token aplana. La abstracción real está
  en el tamaño del conjunto de referencias (v0.16), no en dimensión.
- Herramientas: numpy/PyTorch NO entran en el telefonito (py3.13 aarch64, sin
  wheels/toolchain). Se resolvió con backprop manual en Python puro.

## Arquitectura resultante
GRAFO (memoria/categoría/dolor, PROBADO) + TRANSFORMER (contexto, backprop manual)
como capas complementarias. Nodo del grafo = ω (geométrico) + referencias a otros
nodos (simbólico, v0.16). El grafo no es reemplazable (memoria/dolor); el
transformer no es reemplazable (contexto).

## Estructura del repo
README.md (estado), RESUMEN_NOCHE.md, EXPLICACION_CRIOLO.md, v0.1..v0.16 (cada
uno run_*.py + results_*.json), gpt1_paper.pdf, PANDORA_Resumen.md, scripts de push.

## Estado
Fase 1 completa: sustrato cognitivo (memoria/categoría/dolor) + contexto + DB
semántica compositiva, todo probado con números. Pendiente: polisemia con
transformer completo, abstracción como tamaño de conjunto, y v0.15+ sobre v0.14d.
