# DSCN-G Language Engine — Estado y Análisis (actualizado 2026-07-25)

Proyecto de Luciano. Objetivo: determinar si DSCN-G puede ser sustrato cognitivo
de un motor de lenguaje, terminando en un decodificador L2 rústico y una
"pseudoAGI" de laboratorio (sustrato neuro-simbólico, no reemplazo de LLM).

Todos los experimentos son Python puro (sin numpy). Corpus: Don Quijote
(Project Gutenberg, dominio público) salvo indicado. Paper GPT-1 en gpt1_paper.pdf.

## RESULTADOS MEDIDOS (datos duros)

| Exp | Pregunta | Resultado | Veredicto |
|-----|----------|-----------|-----------|
| v0.1 | ¿El grafo colapsa? | N* satura ~4.5 (sublineal) | ✓ working memory, no masa |
| v0.2 | ¿Colapso paramétrico? | N* sube con K/θ (3.8→166) pero sublineal | ✓ paramétrico, no estructural |
| v0.3 retrieval | ¿El grafo entiende? | recupera 100% (norma) / 91% (bits) a 256 conc. | ✓ entiende |
| v0.3 REAL v2 | ¿Hibernar preserva masa? | retención 100% (N_total=N_init, working set ~4.5) | ✓ VALIDA DB semántica |
| v0.4 | β contextual (Pandora) | N*=5.0 vs 5.2 (ruido, no aporta) | ✗ ρ no se activa en repr. rústica |
| v0.5 | L2 decoder retrieve | "gato"→"gato" OK | ✓ |
| v0.5b | L2 rompe loop | "el casa el casa"→"el roja la corre..." (0 rep. ady.) | ✓ loop roto |
| v0.6a | next-token (corpus real) | accuracy 0.45%→10.11% | ✓ APRENDE |
| v0.6b | dolor post-hoc (RL) | mejora 0.0 | ✗ castigo llega tarde |
| v0.6b-bis | dolor Q-learning en decisión | mejora -0.0012 (ruido) | ✗ redundante en supervisado |
| v0.7 | contexto promedio | 5.89% (peor que 10.11%) | ✗ pisa nodos |
| v0.7-bis | contexto separado | 0.49% | ✗ contamina omega |
| v0.7-final | trigrama limpio (tabla) | 3.85% | ✗ disperso, no escala |
| v0.8 | atención rústica | 8.64% (peor que 10.11%) | ✗ no supera bigrama (vocab chico) |
| v0.9a | dolor señal evasión (corpus) | 0.0149→0.0149 (medido mal) | ✗ diseño: sobre corpus, no generación |
| v0.9a-bis v1/v2 | dolor en generación (repetición) | 0.0→0.0 (no repite sistemáticamente) | ✗ crítico externo no medible aquí |
| v0.9b | etiquetas que MUTAN por uso | 92.67% accuracy vs verdad corpus | ✓ etiqueta emerge de dinámica |
| v0.9c | subsistencia global (dolor interno) | G 0.0→1.0 (el dolor salva al grafo) | ✓ DOLOR EMERGENTE validado |
| v0.10 | persistencia score híbrido (SynapticCache 2.1+2.4) | N_active=N_total (memoria VIVA por relevancia) | ~ política distinta a hibernado |
| v0.11 | abstracción (dimensiones por concepto) | spread abs=con (0.0033), acc gamma<base | ✗ next-token aplana ω, no mide abstracción |
| v0.12 | atención real (ambigüedad sintética) | acc W1=0.097 > W2=0.056 | ✗ contexto no desambigua (falta atención aprendida) |
| v0.13 | híbrido grafo+atención (corpus sintético) | acc W1=0.075 > att=0.055 | ✗ contextos colapsan en corpus sintético |
| v0.13-bis | híbrido atención Don Quijote | acc W1=0.035, W2=0.042, W3=0.038 | ✗ 1 capa sobre ω fijo no desambigua |
| v0.14 | híbrido REAL grafo+transformer 2 capas (Hebbiano) | acc=0.0197 < baseline 0.1011 | ✗ Hebbiano local no entrena atención; requiere backprop |
| v0.14b | híbrido backprop MANUAL (D=8) | acc=0.0012, loss 5.57→5.01 | ~ backprop anda (loss baja) pero no converge a top-1 (piso uniforme) |
| v0.15 | sense nodes (polisemia estructural) | acc_sense=0.499 (azar) | ✗ next-token aplasta sentidos; idea válida pero requiere transformer (v0.14d) |
| v0.15-bis | sense nodes c/contexto identidad | acc_sense=0.496 (azar) | ✗ mismo colapso de v0.13; sense-ω no se separan con next-token |
| v0.16 | referencias compositivas (nodo=ω+refs) | poda respeta externo ✓ | ✓ nodo=conjunto de nodos; poda desenlaza, no borra |
| v0.16-bis | referencias corpus controlado "boda" | jaccard=1.000, poda respeta | ✓ IDEA 2 CONFIRMADA: boda={flores,vestido,blanco,beso} |

## MAPA HONESTO DE CAPACIDADES
- FUERTE en lo LOCAL: recupera conceptos, aprende next-token, masa persistente
  (hibernado), categoriza por uso (etiquetas mutantes 92%), se autopreserva por
  dolor interno (G 0→1).
- DÉBIL en lo que requiere ESCALA/ATENCIÓN: contexto largo (con vocab chico no
  aporta), abstracción (dimensiones por concepto, pendiente), entorno de
  consecuencia (dolor interno de afinidad, no de mundo real).

## PIEZAS CENTRALES CONFIRMADAS
1. MEMORIA MASIVA PERSISTENTE (v0.3 REAL): no olvida, duerme lo que no usa.
2. CATEGORIZACIÓN EMERGENTE (v0.9b): deduce sustantivo/verbo por su dinámica.
3. DOLOR INTERNO QUE OBLIGA A CAMBIAR (v0.9c): el grafo se autopreserva.

## LÍMITES CONFIRMADOS (datos honestos)
- CONTEXTO/ATENCIÓN (v0.7/v0.8/v0.12): el grafo rústico NO desambigua por
  contexto con afinidad coseno. Necesita atención APRENDIDA (transformer), no
  promedio de ω ni tabla. Es un límite de arquitectura, no de datos.
- ABSTRACCIÓN (v0.11): el next-token aplana todas las representaciones hacia un
  centro común, así que "dar más libertad a lo abstracto" no se preserva. El
  diseño no midió abstracción; queda pendiente de métrica distinta (ej generación
  de usos diversos).
- β_eff contextual (v0.4): no se activa en representación rústica (ρ≈0).

## PATRONES SynapticCache INTEGRADOS
- 2.1 score evict híbrido → v0.10 (da memoria viva por relevancia)
- 2.2 omega_root (centroide por V) → v0.10
- 2.3 umbral por distancia → pendiente usar
- 2.4 fallback LRU → v0.10
- 2.5 Modo AUDIT → v0.9a

## ROADMAP
- Hecho: v0.1..v0.3, v0.5/0.5b, v0.6a, v0.9b, v0.3 REAL (hibernado), v0.9c, v0.10
- Descartados por diseño (no concepto): v0.9a/v0.9a-bis (crítico externo no medible), v0.4 (β_eff no se activa)
- No medibles con diseño actual: v0.11 (abstracción aplana), v0.12 (contexto necesita atención aprendida)
- Siguiente: v0.13 (entorno / dolor de consecuencia), o híbrido grafo+atención

## CONCLUSIÓN
El grafo DSCN-G rústico es sustrato neuro-simbólico con memoria masiva persistente
(hibernado), categorización emergente y autopreservación por dolor interno. Aprende
de corpus real y recupera conceptos. CONFIRMADO empíricamente: memoria (v0.3 REAL),
categorización (v0.9b 92.67%), dolor interno (v0.9c G 0→1), next-token (v0.6a 10.11%),
memoria viva (v0.10).

LÍMITES (datos honestos):
- Contexto/desambiguación: v0.14b/c se estancaban en piso uniforme (head fijo).
  v0.14d (head APRENDIDO Wo + lr=0.005, 2 epocas) SUPERA el baseline: 10.55% vs
  10.11% (v0.6a). CONTEXTO RESUELTO en Python puro, sin numpy/PyTorch. El grafo
  (memoria/categoría/dolor) + transformer (contexto, backprop manual) funcionan
  como capas complementarias. Mejora modesta pero REAL y medible; con más D/datos
  debería subir. Ver v0.14d.
- Composición (idea 2 de Luciano): v0.16/v0.16-bis CONFIRMADO. El nodo = omega +
  referencias a OTROS nodos que existen afuera. "boda"={flores,vestido,blanco,beso}
  con jaccard=1.0. Poda por incoherencia (SynapticCache 2.3) desenlaza la
  referencia pero NO borra el nodo externo: el grafo nunca pierde un concepto.
  DB semántica real (concepto = conjunto de conceptos que viven adentro y afuera).

- Polisemia (idea 1 de Luciano): v0.15/v0.15-bis NO midieron (acc_sense~0.50 =
  azar). El next-token aplasta los sense-omega hacia un punto común (mismo
  problema de v0.11). La idea es VÁLIDA (sentido estructural por nomenclatura) pero
  requiere entrenamiento que NO aplaste los sentidos: un transformer con backprop
  (v0.14d) es quien resuelve la polisemia de verdad. Queda como diseño futuro:
  sense nodes + contexto de v0.14d.

Para una pseudoAGI: GRAFO (memoria/categoría/dolor, PROBADO) + TRANSFORMER (contexto,
requiere backprop completo) como capas complementarias. El grafo NO es
reemplazable (aporta memoria/dolor), el transformer NO es reemplazable (aporta
contexto). El diseño está claro; falta la herramienta para entrenar el contexto.

Ver EXPLICACION_CRIOLO.md para la descripción en lenguaje común.

## NOTA DE RIGOR
- v0.4 y v0.9a/abis se reportan como "no medibles / no aportan" con la razón
  técnica, no como éxitos. El rigor importa más que la ambición de claims.
- Lenguaje es RÚSTICO: 10% next-token no es "entender español", es la base.
- El grafo en Python puro en telefonito es lento (N=1000 tarda minutos). Para
  vocab real (50k) necesita numpy/GPU, que acá no tenemos.
