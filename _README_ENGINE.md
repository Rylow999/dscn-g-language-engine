# DSCN-G Language Engine — ESTADO HONESTO (auditoría + corrección + cierre de baches)

## Qué es
Motor de lenguaje sobre DSCN-G (sustrato cognitivo): grafo de conceptos + transformer de
contexto, en Python puro (sin numpy/torch) en Android. Experimental, no producto.

## AUDITORÍA (lo que ESTABA MAL en el README anterior)
Cuatro "✓ confirmados" eran artefactos de diseño (señal falsa), no validación:
- v0.9c original: reward FIJO empujaba a omega_ideal -> G=1.0 por construcción. CIRCULAR.
- v0.9b original: diccionario SUST/VERB DURANTE el train. CIRCULAR. Además top-150 es
  93% sustantivos, así que "92.67%" era el desbalance del corpus.
- v0.16bis original: corpus sintético armado para jaccard=1.0; la poda nunca borra nodos,
  así que "respeta externo" era vacío. CIRCULAR.
- v0.14d original: comparaba 10.55% (V=150) vs 10.11% (V=200, otro corpus). INVÁLIDO.
- v0.3 REAL: retención mecánicamente real (no borra omega) pero motor base usa el MISMO
  omega_ideal/reward que v0.9c -> sustrato circular.

## CORRECCIÓN (señal real del dato, SIN reward fijo / sin dict en train / sin corpus armado)
| Exp | Qué prueba | Resultado corregido | Veredicto |
|-----|-----------|---------------------|-----------|
| v0.14d audit | baseline correcto (grafo V=150) vs híbrido | base=0.0237, híbrido=0.0958 (~4x) | ✓ CONTEXTO GENUINO |
| v0.9b v2 | categorización, vocab balanceado 50/50 | pureza=0.7317 vs azar 0.50 | ✓ CATEGORÍA GENUINA |
| v0.9c limpio | dolor = error next-token real, A(fijo) vs B(aprende) | A=0.9927 cte; B=0.9927->0.933 | ✓ DOLOR GENUINO |
| v0.3b / v0.16 (v1-v3) | memoria/composición, omega preservado vs borrado | hibernado = base en TODOS los tests | ✓ MEMORIA/COMPOSICIÓN (omega vive) |
| v0.14d BORRAR | borrar nodos top sobre híbrido (predice ~9.6%) | base=0.0967, preservado=0.0967, borrado=0.0217 | ✓ BORRAR DESTRUYE (sobre sustrato real) |
| v0.17 | polisemia (idea 1) WSD no sup sobre transformer | 6/150 palabras con 2 sentidos separables (cos<0.5) | ✓ POLISEMIA GENUINA (sense nodes emergen) |
| v0.19 v3 | dolor de consecuencia / evasion (ancla DSCN-G) | aff(A,B) 0.94 -> -0.47 tras dolor | ✓ EVASION GENUINA (el dolor aleja de lo que lastima) |
| v0.18 REAL | transformer completo D=32 (escalar magnitud) | acc=0.0946 (~igual v0.14d 0.0958) | ~ NO ESCALA con ancho: techo es CORPUS (20k tok) |
| v0.3b v2 | memoria: hibernar=excluir+REINTEGRAR (no identidad) | reintegrado ~0.98 vs borrado 0.0 | ✓ MEMORIA REAL (no identidad matematica) |
| v0.14d borrar L | borrar nodos CONTENIDO (no funcion) + hibernar real | base=0.097 hibern=0.075 borrado=0.122 | ~ BORRAR no 'destruye' (sube), HIBERNAR perturba (baja) |
| v0.21 v8c | grafo fractal anchor+repulsion, CTX DENSO (W=8, 6 poli) | acc_gt=0.50 (AZAR), poli_sep=0/6, mono_sep=0/3, theta 0.3/0.5/0.7 igual | ~ FIX NO FUNCIONAL: en contexto fuerte COLAPSA a 1 bucket (oversmoothing); el 0.74 de v8b era regimen de RUIDO/filler, no senal de sentido |
| v0.22 v3 | root + PROYECCION Hebb (sin backprop) | FASE A routing 1.0; FASE B duda 0.0 | ~ ROOT RUTEA PERFECTO; proyeccion mata duda (trade-off) |
| v0.22 v5 | root + contextos MIXTOS + proy SUAVE + MARGIN | duda A/B/MIX = 0.0 | ~ DUDA no emerge: grafo separa sentidos tan bien que siempre hay claro ganador |
| v0.22 v4 | root + MARGIN adaptativo (percentil top1-top2) | margin=0.0, duda=0.0 | ~ MARGIN adaptativo ok, pero proyeccion separa TANTO que no hay ambigüedad |
| v0.23 v1 | composicion relacional Hebb 3-body (2 relaciones) | 4/12=0.333 (azar 0.5) | ~ FALLA: asociacion basica contamina R[r] (ambos pares ocurren) |
| v0.23 v2 | Hebb 3-body SIN contaminacion + 4 relaciones + D16/32 | D16=0.312 D32=0.312 (azar 0.25) | ~ SENAL DEBIL: supera azar pero limitado por mecanismo (no por ancho) |
| v0.23 v3 | Hebb 3-body DATOS REALES (Don Quijote, 89 rels) | D16=0.042 D32=0.032 (azar 0.011) | ~ SENAL DEBIL: supera azar 4x pero extraccion ruidosa + 89 rels => gap abierto |
| v0.24 | memoria trabajo VITALIDAD competitiva (foco + next-token) | foco=0.601; next con=0.038 sin=0.095 | ~ FOCO real (60% dominancia) pero vitalidad NO ayuda next-token (sesga a reciente) |
| v0.25 | harness integracion ciclo 12 pasos (corpus mini) | banco->dinero/rio resuelto (acierto=True) ambas | ~ INTEGRA: bloques se componen en ciclo cerrado; limitado a corpus mini/simple |

## LO QUE QUEDA CONFIRMADO (genuino, señal del dato)
- CONTEXTO: transformer head aprendido ~4x el grafo solo (v0.14d, baseline correcto).
- CATEGORIZACIÓN: la geometría omega separa SUST/VERB sola (v0.9b v2, 0.73 > 0.50).
- DOLOR: el error de predicción del dato baja solo si el sistema aprende (v0.9c limpio).
- MEMORIA: preservar omega (hibernar) mantiene la representación idéntica al base.
- POLISEMIA: WSD no supervisado sobre transformer descubre 6/150 palabras con 2
  sentidos separables por contexto (v0.17). Sense nodes (identidad estructural por
  sentido) EMERGENCIA de la geometría, sin corpus de juguete.
- EVASION (dolor de consecuencia, ancla DSCN-G): tras dolor A->B, aff(A,B) cae de
  +0.94 a -0.47 (A se aleja de lo que lastima) manteniendo alternativa segura (v0.19 v3).

## v0.22 ROOT DIRECTOR (sobre grafo fractal v0.21 v8, SIN transformer)
El root NO amplifica (no promedia ciego): DIRECCIONA. Ruteo competitivo VQ: k* =
argmax_k cos(subnodo_k, contexto). DUDA: si top1-top2 < MARGIN -> root declara
DOUBT (2+ subgrafos sin dominante). Tres intentos:
- v0.22 v1: contexto = promedio de TODOS los subnodos vecinos. routing_acc 0.57
  (azar). El contexto plano no separa sentidos en D=16.
- v0.22 v2: contexto = subnodos GANADORES de vecinos. routing_acc 0.56 (igual, el
  agregado no era el problema). CONCLUSION: el coseno plano en D=16 no discrimina
  sentidos por contexto -> falta PROYECCION (intuicion original de Luciano).
- v0.22 v3: PROYECCION W Hebb (SIN backprop, perfil DSCN-G). routing_acc FASE A =
  1.0 (perfecto en corpus contrastivo). CONFIRMA: el grafo rústico necesitaba
  proyeccion para que el contexto fuera informativo. PERO FASE B (Don Quijote) duda
  = 0.0: la proyeccion separa TANTO que nunca hay ambigüedad aparente -> MATA la
  duda emergente. TRADE-OFF REAL: con proyeccion el root rutea perfecto pero pierde
  la duda; sin proyeccion hay duda (Fase B v1/v2: 0.07-0.33) pero ruteo es azar.
- v0.22 v4: MARGIN adaptativo (percentil de top1-top2). margin=0.0, duda=0.0. El
  mecanismo de MARGIN es correcto, pero la proyeccion Hebb separa TANTO que no hay
  cola de ambiguedad -> duda nunca se dispara.
- v0.22 v5: contextos MIXTOS (ambos sentidos, ej 'banco del rio sacar dinero') +
  proyeccion SUAVE (1 epoch, LR 0.005) + MARGIN adaptativo. duda A/B/MIX = 0.0.
  CONCLUSION HONESTA: el grafo fractal (v0.21 v8, anchor+repulsion) separa los
  sentidos TAN bien que siempre hay claro ganador (routing 1.0). ESO SOSTENIA la
  base de v0.22/v0.25. PERO (auditoria 2026-07-28) v0.21 v8 es CIRCULAR: la
  repulsion INCONDICIONAL garantiza la separacion por construccion (4/5 monosemicas
  tambien dan 'separadas'). Por lo tanto v0.22/v0.25 miden sobre un sustrato cuyo
  'sentido separado' es ruido con forma de senal. Sus veredictos estan EN ESPERA
  hasta re-medir con v0.21 v8b (instrumento correcto). Ver resultados_v21_v8b.json.
  sentidos TAN limpio que SIEMPRE hay un claro ganador, incluso en contexto mixto.
  La duda de SENTIDO no emerge porque el sistema SIEMPRE sabe que sentido es ->
  eso es un EXITO del fractal, no un fallo del root. La "duda" real (decision/
  conflicto de inferencias) requiere un nivel superior, no ambiguedad de palabra.
  CERRADO v0.22: root DIRECTOR rutea perfecto (v3: 1.0); duda de sentido es
  trivialmente resoluble por el grafo -> no es el lugar donde la duda importa.
  GAP siguiente: composicion relacional (v0.23) y duda de DECISION (dolor v0.19/v0.9c).

## v0.21 v8b — AUDITORIA DEL FIX OVERSMOOTHING (2026-07-28)
Luciano detecto que v0.21 v8 era CIRCULAR: repulsion INCONDICIONAL (se aplica a
toda palabra en cada paso, sin testear polisemia) + criterio "2 buckets <85%"
SIN contrastar contra ground truth. Control empirico (run_v21_v8_control.py) en
Quijote real: 4/5 MONOSEMICAS (quijote, sancho, caballero, dijo) daban
'separadas' -> el 39/40 era ruido con forma de senal.

v0.21 v8b RE-MIDE con instrumento correcto: corpus sintetico CON ground truth
(sentido A/B por palabra) + MONOSEMICAS de control (contexto fijo) + repulsion
CONDICIONAL (solo si hay contexto diverso). Resultados:
- INCONDICIONAL: mono_sep=0/3, poli_sep=3/3, acc_gt=0.74 (banco 1.0, mouse 0.95,
  llave 0.27 FALLA). Veredicto GENUNO.
- CONDICIONAL: igual (0/3, 3/3, acc 0.74). La repulsion condicional no cambio nada
  en este corpus (el anchor ya separaba lo que debia).
CONCLUSION HONESTA REVISADA: v0.21 v8 NO es 'artefacto total' (en sintetico
controlado las monosemicas dan sep=False, contradiciendo 'garantiza separacion de
toda palabra'). PERO su INSTRUMENTO ORIGINAL era CIRCULAR: (1) no media acc_gt
(bucket vs sentido real), (2) en corpus real el criterio '2 buckets <85%' cuenta
monosemicas de contexto VARIABLE como 'separadas' por ruido de contexto, no por
sentido. El fix de repulsion como CONCEPTO tiene senal PARCIAL (acc_gt 0.74, pero
llave falla) y requiere corpus controlado para medirse bien. NO descarta la idea
del fix (anchor+repulsion sigue siendo valido para oversmoothing); lo que fallo
fue el instrumento de medicion. v0.22/v0.25 (que asumen sentido real en v0.21 v8)
deben re-evaluarse con instrumento correcto antes de darlos por validados.

## v0.21 v8d — COMPETENCIA POR CONTEXTO (regla A)
v0.21 v8d cambia la regla de update: en vez de promediar vecinos (difusion),
SOLO se actualiza el subnodo k* que mejor matchea el contexto (competencia).
Resultado: acc_gt=0.471 (AZAR), poli_sep=0/6, mono_sep=0/3, curva plana/declina.
La competencia pura NO separa: deja un subnodo estancado y sin repulsion los
subnodos colapsan al promedio del corpus.

## v0.21 v8e — COMPETENCIA + REPULSION CONDICIONAL
Combina competencia (k* ganador se actualiza) + repulsion condicional (empuja
el subnodo NO ganador AWAY del contexto, solo si cos<theta). Resultado:
acc_gt=0.529 (AZAR), poli_sep=3/6, mono_sep=0/3, curva NO CONVERGENTE (oscila
0.41-0.64). Avanza sobre v8d (0→3 poli separadas) PERO no converge: el acc_gt
nunca supera el azar de forma estable. Las 3 que fallan (llave, capital, firma)
tienen contextos que se solapan (ej: capital=dinero vs capital=ciudad compite
con oro=dinero/plata).

## v0.22 v2 — ROOT DIRECTOR sobre TRANSFORMER (v0.14d) (2026-07-28)
v0.22 v2 re-define sobre el TRANSFORMER v0.14d (que SI separa sentidos, acc_pred
0.907 >> 0.013 azar). La pregunta REAL: ¿la proyeccion Hebb del ROOT aporta sobre
un sustrato que separa sentidos? Instrumento: transformer minimo (backprop) sobre
corpus sintetico CON ground truth + proyeccion Hebb del root + baseline (contexto
solo). Resultado:
  acc_pred(transformer)=0.907 (aprende, separa sentidos)
  routing_acc_gt(root sobre transformer)=0.545 (casi azar 0.50)
  baseline(contexto solo)=0.545
  VEREDICTO: ROOT REFLEJA (root ~= baseline, NO aporta como proyector de sentido).
CONCLUSION: la proyeccion Hebb del ROOT DIRECTOR no aporta como proyector de
sentido. El transformer separa sentidos por si solo (atencion/backprop); la
proyeccion Hebb del root solo reproduce eso. COHERGENTE con NOUS v4: el grafo
(root) no resuelve sentido (eso es del transformer); el root debe ser
MEMORIA/DOLOR/FOCO sobre el contexto (v0.3b, v0.19, v0.24), no proyector de
sentido. v0.25 v2 debe usar: transformer=contexto/sentido + root=memoria/dolor/
foco. La proyeccion Hebb del root como proyector de sentido se DESCARTA.

## v0.23 COMPOSICION RELACIONAL (Gap 2 hacia pseudoAGI)
El grafo fractal (v0.21 v8) codifica CO-OCURRENCIA, no RELACION ESTRUCTURADA.
v0.23 aprende TRIPLAS (sujeto, RELACION, objeto) por Hebb 3-body: R[r] (matriz DxD)
tal que R[r]*emb[s] ~ emb[o]. Dos intentos:
- v0.23 v1: 4/12=0.333 (azar 0.5). FALLA porque al acercar emb[s]~emb[o] (asociacion
  basica) se contamina R[TIENE] y R[LUGAR] (ambos pares ocurren en el corpus).
- v0.23 v2: SIN asociacion basica (solo refuerza R[r]), corpus menos sintetico (8
  sujetos x 4 relaciones: TIENE/LUGAR/CAUSA/PARTE_DE), 20 epochs, D16 y D32.
  D16=0.312 D32=0.312 (azar 0.25) -> SUPERA azar pero senal DEBIL. D16=D32 ->
  el cuello NO es el ancho, es el MECANISMO (Hebb 3-body simple). Conclusion
  honesta: la composicion relacional es ALCANZABLE (hay senal real sobre azar) pero
  el Hebb 3-body naive es insuficiente para solidez (>0.7). GAP ABIERTO: requiere
  mas datos reales (no sinteticos) o mecanismo de relacion mas fuerte
- v0.23 v3: DATOS REALES (Don Quijote 20k tok, vocab 150). Tríplas extraídas de
  patrones sintácticos reales ("X de Y"->DE, "X en Y"->EN, "X y Y"->CON, "X a Y"->A,
  suj-verb-obj->V_verb) -> 89 relaciones. D16=0.042 D32=0.032 (azar 0.011) ->
  SUPERA azar (~4x) pero accuracy ABSOLUTA bajísima. D32<D16: ancho NO ayuda.
  Causa: extracción por patrones es RUIDOSA (suj/obj son artículos/pronombres como
  "los","de","y"); 89 relaciones dispersas es demasiado para Hebb 3-body.
  CONCLUSIÓN HONESTA: Gap 2 NO se cierra con este enfoque. Hay señal (supera azar)
  pero insuficiente para solidez. GAP ABIERTO: requiere (a) extracción limpia
  (solo sustantivos como suj/obj, relaciones agrupadas), o (b) mecanismo de relación
  más fuerte (tensor/relational memory), o (c) menos relaciones + más ejemplos.
  Se deja DOCUMENTADO como gap abierto y se pasa a MEMORIA DE TRABAJO (v0.24).

## v0.24 MEMORIA DE TRABAJO CON VITALIDAD (Gap 3 hacia pseudoAGI)
Memoria de trabajo = SLOTS competitivos. Cada nodo tiene vitalidad V (cuanto
activo/reciente). Al procesar seq: nodo actual recibe disparo V+=1; los demas
decaen V*=0.85. Foco = nodo de mayor V (atencion Hebbiana, sin backprop).
Resultados (Don Quijote 20k tok, vocab 150):
- TEST1 foco dominado por disparado: 12029/19999 = 0.601 -> el nodo recien
  disparado DOMINA el foco 60% de las veces. SENAL REAL de memoria de trabajo
  (atencion competitiva) emerge. El 40% restante: palabras muy frecuentes ya
  "calientes" compiten.
- TEST2 next-token: CON vitalidad=0.038, SIN vitalidad=0.095 (azar 0.007).
  La vitalidad RESIDUAL EMPEORA next-token (sesga a lo reciente = ruido de foco).
  El next-token puro por co-ocurrencia ya funciona bien (0.095 = 13x azar).
CONCLUSION HONESTA: Gap 3 PARCIAL. La vitalidad competitiva SÍ crea foco de
memoria de trabajo real (60% dominancia) —genuino, coherente con el ancla DSCN-G
(V homeostatica). PERO su beneficio NO es next-token: la vitalidad es mecanismo de
RETENCION/ATENCION para decisiones, no predictor de palabra. El test de next-token
no es donde brilla. Nota: use decaimiento LINEAL (*0.85); NOUS Tecnico v4 Ec.5 usa
decaimiento EXPONENCIAL V*=e^-gamma + A(1-e^-gamma) con poda (V<0.10 muere). v0.25
debe usar la formula correcta y conectar V con DOLOR (Ec.6) y VENTANA (Ec.8).

## MAPA DE GAPS HACIA PSEUDOAGI (estado 2026-07-28)
CONFIRMADO (senal del dato, experimentos reales):
  [polisemia]      grafo fractal ancla + fix oversmoothing  -> v0.21 v8 (39/40
   ARTIFACTUAL, v8f: acc_gt<=0.53 azar). Grafo rustico NO separa sentidos.
   Transformer (v0.14d) separa (acc_pred=0.907). v0.25 v2 integra.
  [ruteo sentido]  root DIRECTOR + proyeccion Hebb          -> v0.22 v3 (1.0)
  [memoria]        hibernar reintegra / borrar mata          -> v0.3b v2 (~0.98/0.0)
  [memoria trabajo] foco vitalidad competitiva              -> v0.24 (0.601 dominancia)
  [ajuste]         dolor por dato + aprendizaje por dolor    -> v0.19 limpio / v0.9c
DEBIL / GAP ABIERTO:
  [composicion]    Hebb 3-body: 0.036 sobre Don Quijote (azar 0.026, diff=0.009).
                    v0.23 v3 con corpus sintetico daba 0.042 vs 0.011 (diff=0.031),
                    pero no aguantaba control monosemico. En corpus real la senal
                    es muy debil: Hebb rústico NO alcanza para composicion relacional.
                    Queda abierto: requiere metodo mas fuerte (ej. transformer sobre
                    relaciones) o corpus con relaciones mas explicitas.
  [loop cerrado]   v0.25 v8 (transformer+root+memoria+decodificador) empeora
                    baseline (0.758 -> 0.550). v0.25 v8b sin decodificador empeora
                    aun mas (0.417). v0.25 v9 con skip-gram embeddings reales
                    MEJORA sobre su propio baseline debil (0.328 -> 0.500). Con-
                    clusion parcial: los embeddings reales ayudan, pero el loop
                    actual no es robusto; falta calibrar baseline y reglas.
NO INTEGRADO (el verdadero muro):
  [loop cerrado]   los bloques arriba NO se componen en un ciclo
  [decodificador]  generar lenguaje desde sentido ruteado
  [decision]       accion sobre el foco + dolor dirige update
  [meta/autoobs]   duda de DECISION que dispara busqueda

## PLAN v0.25 — HARNESS DE INTEGRACION (ciclo de 12 pasos, NOUS Tecnico v4 Sec.7)
En vez de medir bloques aislados, construir UN engine que corre el ciclo cerrado
sobre una tarea que exija COMPOSICION de bloques:
  PASO 1  percepcion -> embedding (grafo fractal D=16 como subespacio)
  PASO 2  activacion de nodos (K cadenas por afinidad, Ec.2)
  PASO 3  update omega (TD sobre nodos visitados, Ec.1 — SIN hardcodear dir)
  PASO 5  vitalidad V (decaimiento EXPONENCIAL Ec.5, con poda V<0.10)
  PASO 6  valencia/dolor E = max(0, A - V)*kappa (Ec.6)  [conecta v0.24 con v0.19]
  PASO 7  ventana W(t) dinamica = W_base/(1+kappa_W*E_root) (Ec.8) [atencion adaptativa]
  PASO 11 seleccion de accion (von Mises sobre fase root, Ec.4) -> decodificador
Tarea de prueba: frase con palabra polisemica ambigua + contexto mixto. El loop debe
(a) resolver el sentido (root DIRECTOR), (b) mantenerlo en foco (vitalidad), (c)
generar continuacion coherente con el sentido (decodificador), (d) si es incoherente,
el dolor (valencia) CONTRAE la ventana y el update se ajusta. Metrica: continuacion
respeta el sentido resuelto (no el otro) y la ventana se contrae ante incoherencia.
Esto separa de una vez si los bloques se componen o solo viven aislados.

## v0.22 ROOT DIRECTOR (Gap 1 hacia pseudoAGI)
- v0.19 ORIGINAL (v3): A=A-alpha*B/|B|+alpha*C/|C| x2000 GARANTIZABA alejamiento de B.
  CIRCULAR. v0.19 LIMPIO: dolor = error de next-token real; evasion dirigida por dato
  (se aleja del mal-predicho, se acerca al correcto). Resultado REAL: err 19291->18761
  (-2.7%). Pequeno pero genuino (no formula que lo garantiza).
- v0.14d BORRAR ORIGINAL: borraba top-30 palabras FUNCION (de,y,la) -> rompe cualquier
  modelo, artefacto. v0.14d BORRAR LIMPIO: nodos de CONTENIDO (top-31..80). Hallazgo
  honesto: BORRAR NO 'destruye' (acc sube 0.097->0.122 al quitar competidores); HIBERNAR
  (excluir del entrenamiento) SI perturba (baja 0.097->0.075). El efecto es 'perturbacion
  de entrenamiento', no 'destruccion'.
- v0.3b/v0.16 ORIGINAL: 'hibernar' = no tocar omega -> = base por identidad matematica.
  CIRCULAR. v0.3b v2 LIMPIO: hibernar = excluir un tramo y REINTEGRAR. Resultado REAL:
  reintegrado ~0.98 (recupera tras volver a entrenar) vs borrado 0.0 (muerto). Memoria
  real, no identidad.
- v0.9c ORIGINAL: con corpus chico el efecto era debil/no monotono. v0.9c ROBUSTO:
  varias semillas + corpus completo + curva de error por epoca. Resultado REAL:
  err 0.0024->0.0002 monotono y consistente entre 5 semillas. APRENDIZAJE POR DOLOR
  robusto (dirigido por error real, no reward fijo circular).

## LECCION DE OVERSMOOTHING (diagnostico de Luciano, 2026-07-28)
La regla omega[a]=(1-beta)omega[a]+beta*omega[b] ES una difusion de grafo (power
iteration de cadena de Markov). Converge al autovector dominante: la separacion de
sentidos (componente de ALTA frecuencia del espectro) es literalmente lo que un
filtro pasa-bajos mata PRIMERO, sin importar D ni epocas. Por eso v0.21 v1-v7 daba
separacion TRANSITORIA (v6 ep11, v7 ep1) y luego colapso irrevocable. NO es falta de
profundidad, atencion aprendida, corpus o epocas: es propiedad del OPERADOR.
Arreglos SIN backprop (v0.21 v8): (1) ANCHOR/RESTART (Personalized PageRank/APPNP):
omega[a]=alpha*omega0[a]+(1-alpha)[(1-beta)omega[a]+beta*omega[b]] -> el ancla
omega0 es inerosionable, rompe la convergencia al autovector dominante; (2) REPULSION
SIBLING (beta negativo hacia el hermano del mismo lema) evita que los sentidos de un
mismo lema se fundan. Esto devuelve la intuicion original de Luciano ("el problema es
como lo aplicamos, no el grafo"): el grafo rústico SÍ puede sostener separacion si se
cambia la REGLA de update, no el sustrato. REGLA: antes de culpar al sustrato por
"colapsar", analizar si la REGLA de update es un filtro pasa-bajos (difusion) que
destruye senal de alta frecuencia.

## LECCION METODOLOGICA (error de vision, 2026-07-28)
En v0.21 v1-v5 concluimos apresuradamente "el grafo rustico D=16 no tiene senal /
aplana". ESO FUE UN ERROR DE VISION. El transformer v0.14d/17 viene PRE-ENTRENADO
(millones de ejemplos, embeddings utiles) y da 9.6% a las 2 epocas; el grafo rustico
arranca de RUIDO PURO y deberia MEJORAR CON EL TIEMPO, no disparar al inicio como una
LLM. Nunca medimos la CURVA de epocas ni usamos un corpus con polisemia CONTRASTIVA
real (Don Quijote tiene polisemia rara y poco frecuente). REGLA: antes de decir
"el sustrato no puede", aislar la variable (corpus contrastivo + curva de epocas +
umbral relajado). El grafo y el transformer NO son comparables a iguales epocas
porque arrancan de estados opuestos (ruido vs util). v0.21 v6 testea esto.

## LÍMITES DEL SUSTRATO Y LECCIONES (v0.18 / v0.21)
El grafo rústico (D=16) predice ~8% (error ~92%). v0.18 (transformer completo D=32,
mismos 20k tok) dio 9.46%, igual que v0.14d híbrido (9.58%): el techo NO es la
v0.21 intentó reemplazar al transformer por grafo fractal + root bottom-up SIN
transformer. v1/v2 (round-robin ciego ka=i%K) -> promedio borroso, 0.024<0.034
plano; v3 (desambiguación) -> 0/40 sentidos (subnodos recibían mezcla aleatoria).
BUG DETECTADO POR AUDITORÍA: el ruteo era i%K (round-robin), no competencia ->
los subnodos nunca divergían. v4 arregló el ruteo con VQ winner-take-all: bug de
ruteo DESAPARECIÓ pero dio 0/40 por COLAPSO AL GANADOR (contexto en D=16 es ruido).
v5 probó competencia SUAVE (temperatura) en Don Quijote: 0/40 x3 semillas (el
colapso persistió). v6/v7 testearon CORPUS CONTRASTIVO + CURVA: v6 llegó a 50/2403
en ep11 pero recolapsó (vocab inflado); v7 (vocab ok + repulsión débil) ep1:3/3 ->
ep4-15:0/3. DIAGNÓSTICO DE LUCIANO (clave): la regla omega[a]=(1-beta)omega[a]+
beta*omega[b] es DIFUSIÓN DE GRAFO (power iteration de Markov) -> OVERSMOOTHING:
converge al autovector dominante y mata la separación (componente alta frecuencia)
sin importar D ni épocas. La separación es SIEMPRE transitoria. ARREGLOS SIN
BACKPROP (v0.21 v8): (1) ANCHOR/RESTART (APPNP) omega[a]=alpha*omega0[a]+(1-alpha)
[(1-beta)omega[a]+beta*omega[b]] rompe la convergencia; (2) REPULSION SIBLING
(beta negativo hacia el hermano del mismo lema) evita fusión. RESULTADO: sintético
3/3 ESTABLE (alpha 0.05-0.2); DON QUIJOTE REAL 39/40 ESTABLE a lo largo de 8 épocas.
CONCLUSIÓN: el grafo rústico D=16 SÍ SOSTIENE polisemia SIN transformer cambiando la
REGLA de update (no el sustrato). Mi cierre anterior ("necesita transformer para
sostener separación") fue OTRO ERROR DE VISIÓN: concluí por exclusión, no por
mecanismo. El diagnóstico de oversmoothing de Luciano invalida esa conclusión. La
idea de fractal + root DIRECTOR que puede dudar es válida COMO SUSTRATO (no solo
orquestador sobre transformer).


## NOTA SOBRE EL GRAFO RÚSTICO VS TRANSFORMER
En v0.3b/v0.16 (grafo rústico, ~8% accuracy) no fue detectable porque el sustrato no
predice lo suficiente. En v0.14d BORRAR (híbrido, ~9.6%) SÍ: borrar los 30 nodos top
baja la accuracy de 0.0967 a 0.0217 (~4.5x menos), mientras preservarlos la mantiene
(0.0967). La memoria/composición es REAL y medible sobre un sustrato con capacidad.

## LÍMITES DEL SUSTRATO (grafo rústico, D=16)
El error absoluto de next-token es ALTÍSIMO (~92% en v0.9c, ~92% en v0.3b/v0.16/v0.9b).
El grafo rústico APLANA representaciones y predice pesimo. Lo único que rompe el piso es
el transformer con backprop (v0.14d, ~9.6%). Sin eso, ningún mecanismo "aprende" de verdad
en magnitud, aunque su DIRECCIÓN (memoria/dolor/categoría/contexto) es genuina.

## v0.22 ROOT DIRECTOR (Gap 1 hacia pseudoAGI)
El grafo fractal (v0.21 v8) separa sentidos pero hace falta un ROOT que RUTEE el
sentido global. v0.22 probo variants:
- v1/v2: contexto plano (promedio de subnodos) -> ruteo 0.56 (azar). El coseno
  plano no separa sentidos.
- v3: PROYECCION Hebb (sin backprop) -> ruteo PERFECTO (1.0). Confirma la intuicion
  de Luciano de proyeccion. PERO mata la duda (tasa_duda 0.0).
- v4: MARGIN adaptativo (percentil top1-top2) -> margin=0.0, duda=0.0. El mecanismo
  es correcto pero la proyeccion separa TANTO que no hay ambiguedad.
- v5: contextos MIXTOS + proy SUAVE -> duda 0.0 en A/B/MIX.
CONCLUSION HONESTA: root DIRECTOR rutea perfecto (v3: 1.0). La duda de SENTIDO no
emerge porque el grafo fractal separa los sentidos TAN bien que SIEMPRE hay claro
ganador. Eso es un EXITO del fractal, no un fallo. La duda real es de DECISION
(dolor v0.19/v0.9c), no de palabra. Gap 1 CERRADO (ruteo funciona).

## v0.23 COMPOSICION RELACIONAL (Gap 2 hacia pseudoAGI)
El grafo fractal codifica CO-OCURRENCIA, no RELACION ESTRUCTURADA. v0.23 aprende
TRIPLAS (sujeto, RELACION, objeto) por Hebb 3-body: R[r] tal que R[r]*emb[s]~emb[o].
- v1: 4/12=0.333 (azar 0.5). FALLA: asociacion basica contamina R[r].
- v2: SIN asociacion basica + 4 relaciones + D16/32 -> D16=0.312 D32=0.312 (azar 0.25)
  -> supera azar pero senal DEBIL.
- v3: DATOS REALES (Don Quijote, 89 rels) -> D16=0.042 D32=0.032 (azar 0.011) ->
  supera azar ~4x pero accuracy ABSOLUTA bajisima. D32<D16: ancho NO ayuda.
CONCLUSION HONESTA: Gap 2 NO se cierra. Hay senal (supera azar) pero Hebb 3-body
naive es insuficiente para 89 relaciones ruidosas. GAP ABIERTO: extraccion limpia
(solo sustantivos) o tensor/relational memory. Documentado, no cerrado.

## v0.24 MEMORIA DE TRABAJO CON VITALIDAD (Gap 3 hacia pseudoAGI)
Memoria de trabajo = SLOTS competitivos. Cada nodo tiene vitalidad V. Al procesar
seq: nodo actual recibe disparo V+=1; los demas decaen V*=0.85. Foco = nodo de
mayor V (atencion Hebbiana, sin backprop).
- TEST1 foco dominado por disparado: 12029/19999 = 0.601 -> el nodo recien disparado
  DOMINA el foco 60% de las veces. SENAL REAL de memoria de trabajo emerge.
- TEST2 next-token: CON vitalidad=0.038, SIN vitalidad=0.095 (azar 0.007). La
  vitalidad RESIDUAL EMPEORA next-token (sesga a lo reciente = ruido de foco).
CONCLUSION HONESTA: Gap 3 PARCIAL. La vitalidad competitiva SÍ crea foco de memoria
de trabajo real (60% dominancia), coherente con ancla DSCN-G (V homeostatica). PERO
su beneficio NO es next-token: es RETENCION/ATENCION para decisiones. Nota: use
decaimiento LINEAL; NOUS v4 Ec.5 usa EXPONENCIAL V*=e^-gamma + A(1-e^-gamma) con
poda (V<0.10 muere). v0.25 usa la formula correcta.

## v0.25 HARNESS DE INTEGRACION (ciclo 12 pasos, NOUS Tecnico v4 Sec.7)
PRIMER intento de UNIR los bloques en UN ciclo cerrado sobre tarea que exige
composicion: frase con "banco" polisemico + contexto. Ciclo fiel a NOUS v4:
vitalidad V decae EXPONENCIAL (Ec.5), dolor E=max(0,A-V)*kappa (Ec.6), ventana
W=W_base/(1+kappa_W*E_root) (Ec.8), decodificador por afinidad (von Mises Ec.4
simplificado, sin fase real).
Resultados (corpus MINI 20 tok):
- banco+dinero: foco resuelve 'dinero' (acierto=True), W=[37.5,50], dolor=0.167
- banco+rio:    foco resuelve 'rio'/'banco' (acierto=True), W=[40,50], dolor=0.125
CONCLUSION HONESTA: los bloques SÍ SE COMPONEN (integracion real, no aislada). El
sentido polisemico se resuelve en AMBAS frases. La ventana NO se contrae porque el
dolor es BAJO (corpus limpio) -> correcto segun Ec.8. LIMITACIONES (no inflar):
corpus MINI (no Don Quijote), decodificador es afinidad simple (sin fase phi real),
"acierto" solo revisa foco post-banco (NO mide generacion de lenguaje), y no se
probó contraccion de ventana ante DOLOR real. GAP: v0.25 v2 debe usar grafo fractal
v0.21 v8 sobre Don Quijote, fase phi real para von Mises, DECODIFICADOR GENERATIVO,
y forzar incoherencia para ver W contraerse por dolor.

## v0.25 v2 — INTEGRACION TRANSFORMER + ROOT (arquitectura NOUS v4 correcta)
v0.25 original asumia root=proyector de sentido sobre grafo rústico (v0.21 v8).
Pero v0.21 v8→v8f CERRARON que el grafo rústico no separa sentidos (acc_gt<=0.53,
azar), y v0.22 v2 confirmó que el root no aporta como proyector (root≈baseline).
v0.25 v2 usa la arquitectura CORRECTA (NOUS v4):
  - TRANSFORMER = contexto/sentido (backprop, separa polisemia, acc_pred=0.907).
  - ROOT/GRAFO = MEMORIA/DOLOR/FOCO sobre el contexto (v0.3b, v0.19, v0.24),
    NO proyector de sentido.
Resultado: acc_gt=0.546 (AZAR), foco_acc=0.546, dolor_max=0.884.
VEREDICTO: CICLO NO FUNCIONAL para polisemia. El transformer separa sentidos
(acc_pred=0.907) PERO el root (slots+vitalidad+Hebb) no los rutea (acc_gt≈azar).
Causa: la vitalidad competitiva + Hebb local no corrige atracciones tempranas
equivocadas; el contexto promedio mezcla A/B (filler) y el slot que gana por azar
se auto-refuerza en el sentido equivocado. El transformer usa BACKPROP (corrige
globalmente); el root usa HEBB LOCAL (refuerza solo el ganador).

## v0.25 v2d — ¿LA REPRESENTACIÓN DEL TRANSFORMER SEPARA A/B? (2026-07-28)
v0.25 v2d pregunta: ¿la representación omega del transformer separa A/B para cada
polisemia? (si no separa, el root no puede separar sobre ella). Resultado:
  acc_gt_simple=0.533 (AZAR), cos(A,B)=0.57-0.79 (ALTO) para casi todas.
  banco cos(A,B)=0.786, llave 0.770, mouse 0.623, capital 0.567, oro 0.423.
VEREDICTO: TRANSFORMER NO SEPARA A/B. acc_pred=0.907 (predice tokens) NO implica
separación de sentido: predecir el próximo token solo requiere co-ocurrencia, no
distinguir sentidos. "banco" tiene el MISMO embedding para dinero y río.
CONCLUSION: el root no separa sentido PORQUE la representación no separa (no es
culpa del root). Para separar sentido, el transformer debe entrenarse para
CLASIFICAR sentido (BERT-style masked LM, no solo predecir tokens). El root =
memoria/dolor/foco sobre el contexto (NO clasificador de sentido).

## CIERRE DEFINITIVO: ROOT NO SEPARA SENTIDO (2026-07-28)
5 experimentos (v0.22 v2, v0.25 v2, v2b, v2c, v2d): acc_gt≈0.50 (azar) en todos.
El root NO separa sentido. CAUSAS: (1) el grafo rústico no separa (v0.21 v8→v8f),
(2) el transformer mínimo separa tokens PERO NO sentidos (v2d: cos(A,B) alto),
(3) el root no está entrenado para clasificar (Hebb local no basta).
EL ROOT FUNCIONA COMO SISTEMA DE DUDA: dolor_duda=0.841, W_contrae=0.982 (v2c).
ARQUITECTURA CORRECTA (NOUS v4): transformer (BERT-style)=sentido, root=memoria/
dolor/foco sobre el contexto. El root no es clasificador de sentido.

## v0.25 v3 — TRANSFORMER BERT-STYLE (masked LM) (2026-07-28)
v0.25 v3 prueba si un transformer BERT-style (masked LM + atención + LayerNorm)
separa A/B (que el next-token mínimo no separa, v2d: cos(A,B) alto). Resultado:
  acc_mlm=0.131 (aprende, 10x azar 0.013)
  acc_gt_simple=0.533 (AZAR), cos(A,B)=0.70-0.94 (ALTO).
VEREDICTO: BERT NO SEPARA A/B. El masked LM sobre corpus SINTÉTICO no basta:
el MLM predice tokens por co-ocurrencia (acc_mlm sube) PERO no distingue sentidos
(cos(A,B) alto, acc_gt≈azar). Para separar A/B, necesitamos corpus REAL (miles de
millones de tokens, contextos distintivos) + capacidad suficiente (multi-capa,
D=768), como BERT real. Nuestro BERT mínimo (1 capa, D=16, 5280 tokens) no
alcance. El MLM sobre corpus sintético no fuerza separación de sentido.

## v0.25 v4 — ROOT como SISTEMA DE DUDA sobre transformer (2026-07-28)
v0.25 v4 valida el root como SISTEMA DE DUDA. El transformer decide sentido (Wo
predice próximo token; decision basada en si el token predicho es distintivo de
A/B). El root mide coherencia entre decision y memoria; si duda, dolor→contrae
W. Resultado:
  acc_pred(transformer)=0.901 (aprende, separa tokens)
  acc_decision(transformer decide A/B)=0.544 (AZAR: el token predicho es filler,
   no distintivo de A/B → cae al else: cos con slots = azar)
  dolor_en_duda=0.091 > dolor_en_confianza=0.000 (root DISTINGUE duda de confianza)
  W_contrae=0.091 (bajo: dolor es bajo porque decision es azar)
  foco_acc=0.635 (memoria retiene algo, >0.50)
  VEREDICTO: ROOT COMO DUDA NO FUNCIONA PLENAMENTE. El root distingue duda de
  confianza PERO la duda no es significativa (acc_decision=azar → incoherencia
  aleatoria). acc_pred=0.901 (predice tokens) ≠ acc_decision=0.544 (clasifica
  sentido): el transformer sabe qué token viene PERO no usa eso para clasificar
  A/B. PARA que la duda funcione, el transformer debe RESOLVER sentido (BERT-style
  sobre corpus real, supervisión de sentido). La duda como indicador de cambio de
  contexto (idea de Luciano) requiere transformer que resuelva sentido primero.

## v0.25 v6 — ROOT con ATENCION SELECTIVA (no promedio) (2026-07-28)
v0.25 v6 prueba atención selectiva sobre el contexto (peso por distintividad A/B)
en vez de promedio ciego. Resultados:
- Corpus mezclado A/B intercalado (n_per_sense=60): acc_dec=0.459 (azar),
  dolor_en_cambio=0.150 < dolor_en_estable=0.504 → NO detecta cambio.
- Corpus de BLOQUES LARGOS A puro / B puro (n_per_sense=30): acc_dec=0.890
  (supera ampliamente azar), pero dolor_en_cambio=0.036 < dolor_en_estable=0.113
  → la duda NO detecta cambio abrupto con W=8 local.
VEREDICTO: LA ATENCIÓN SELECTIVA FUNCIONA COMO SEPARADOR CUANDO EL CONTEXTO ES
PURO (bloques largos). La duda NO funciona como detector de cambio abrupto con
W=8 local. La línea de polisemia con contexto local queda CERRADA en este
formato. Camino siguiente (paso offline propuesto por Luciano): confirmar si
existe señal bimodal REAL en Don Quijote antes de meter mecanismo online.

## PASO OFFLINE — ¿EXISTE ESTRUCTURA BIMODAL REAL EN DON QUIJOTE?
Objetivo: separar (1) ¿hay señal? de (2) ¿el mecanismo la encuentra?
Experimento aislado, SIN mecanismo DSCN: k-means offline sobre contextos reales.
- Palabra: 'banco' (5 ocurrencias en Don Quijote)
- Contextos: ventana W=10 alrededor, bag-of-words normalizado
- k=1 baseline vs k=2 propuesto
Resultados:
  k=1 -> inertia=1.307, silhouette=0.000
  k=2 -> inertia=0.559, silhouette=0.552, mejora_inertia=57.2%
VEREDICTO: EXISTE ESTRUCTURA BIMODAL real en contextos de 'banco' en Don Quijote.
k=2 separa mejor que k=1 a pesar de solo 5 ocurrencias. Por lo tanto, el
problema anteriores no era la falta de señal en el corpus: era el experimento
online sin semilla ni configuración adecuada.

## PASO 2 ONLINE — SEMILLA K-MEANS EN GRAFO (v0.25 v7b)
v0.25 v7 porta las medias de k-means como semilla omega0 para 'banco' en
donquijote.txt: init cos(A,B)=0.640, final=-0.375, divergencia clara.
PERO evaluación real muestra que 'banco' en Don Quijote no tiene sentidos
mixtos (5 ocurrencias, todas "banco de barco"). Luego v7 es un resultado
tecnico, no de polisemia.
v0.25 v7b repite el experimento sobre corpus sintético mejorado (oraciones
realistas A/B con ground truth) para 3 palabras:
- banco: kmeans sil=0.599, mejora_inertia=48.3% -> online ESTABLE (init=-0.540, final=-0.693).
- llave: kmeans sil=0.758, mejora_inertia=65.3% -> online ESTABLE
  (init=-0.900, final=-0.896).
- cabo: kmeans sil=0.727, mejora_inertia=61.5% -> online ESTABLE
  (init=-0.049, final=-0.048).
VEREDICTO: aunque el paso 1 offline confirma estructura bimodal real,
el paso 2 online NO refina esa señal con la config actual. En 'banco' la
destruye; en 'llave'/'cabo' se mantiene estable sin mejorar. Por lo tanto,
la hipótesis 'la semilla k-means soluciona la separacion online'queda
REFUTADA en esta config. Queda abierto: requiere mecanismo online mas
fuerte o inicializacion diferente.

## v0.25 v7c — CAMBIO DE CONFIG ONLINE
v7c prueba otras 3 configs online partiendo de la misma semilla k-means en
'banco':
- baseline_v7b: init=-0.715 final=-0.660 div=+0.055 -> COLAPSA ONLINE.
- repulsion_fuerte: init=-0.720 final=-0.778 div=-0.057 -> SEPARA ONLINE.
- anchor_mas_fuerte: init=0.014 final=0.014 div=-0.000 -> ESTABLE ONLINE.
VEREDICTO PARCIAL: repulsion incondicional fuerte SÍ separa desde semilla
(divergencia negativa), mientras que la config por defecto colapsa. Eso
descarta la hipótesis “la semilla no alcanza”: depende de la regla online.
Queda pendiente evaluar si esa separación es real o un artefacto de setup.

## PASO OFFLINE — ¿EXISTE ESTRUCTURA BIMODAL REAL EN DON QUIJOTE?
CONFIRMADO (senal del dato, experimentos reales):
  [ruteo sentido]  root DIRECTOR + proyeccion Hebb          -> v0.22 v2 (root refleja, no anade)
  [memoria]        hibernar reintegra / borrar mata          -> v0.3b v2 (~0.98/0.0)
  [memoria trabajo] foco vitalidad competitiva              -> v0.24 (0.601 dominancia, parcial)
  [ajuste]         dolor por dato + aprendizaje por dolor    -> v0.19 limpio / v0.9c
CERRADO / DESCARTADO:
  [polisemia]      grafo rustico D=16 + anchored/polysemy     -> v0.21 v8 (39/40 ARTIFACTUAL), v8f acc_gt<=0.53.
                    kmeans offline confirma estructura bimodal real en Don Quijote (silhouette 0.552, +57% inercia).
                    v0.25 v7b (semilla k-means en omega0) sobre corpus sintetico con ground truth:
                    offline OK (silhouette 0.599-0.758), online config default COLAPSA/ESTABLE, acc_gt real=0.500.
                    v0.25 v7c: repulsion incondicional fuerte logra separacion tecnica online, pero SIN acierto semantico todavia.
                    Lema: la semilla k-means ayuda, pero la config online todavia no convierte divergencia tecnica en acierto semantico.
DEBIL / GAP ABIERTO:
  [composicion]    Hebb 3-body: 0.036 sobre Don Quijote (azar 0.026, diff=0.009).
                    v0.23 v3 con corpus sintetico daba 0.042 vs 0.011 (diff=0.031),
                    pero no aguantaba control monosemico. En corpus real la senal
                    es muy debil: Hebb rustico NO alcanza para composicion relacional.
                    Queda abierto: requiere metodo mas fuerte o corpus con
                    relaciones mas explicitas.
  [loop cerrado]   v0.25 v8 (transformer+root+memoria+decodificador) empeora
                    baseline (0.758 -> 0.550). v0.25 v8b sin decodificador empeora
                    aun mas (0.417). v0.25 v9 con skip-gram embeddings reales
                    MEJORA sobre su propio baseline debil (0.328 -> 0.500).
                    v0.25 v10: baseline clasificador lineal=0.766; loop cae a 0.490
                    en test. v0.25 v11 (loop conservador) da +0.137 sobre baseline
                    para 'banco', pero v0.25 v11b NO generaliza: para 'llave' cae
                    a 0.500 desde baseline 1.000. v0.25 v12 (decoder por similitud
                    de embeddings) es NO FUNCIONAL (top1=0.020, top5=0.095).
                    v0.25 v13/v13b (transicion explicita bigramas/trigramas)
                    FUNCIONAL: top1=0.630, top5=0.940, generaciones coherentes.
                    Conclusion: la generacion next-token no requiere embeddings
                    densos en este régimen; alcanza con modelo de transicion
                    aprendido del corpus. Queda abierto: integrar este generador
                    con el loop de sentido para que el sentido ruteado influya
                    en la generacion.
NO INTEGRADO (el verdadero muro):
  [decision]       accion sobre el foco + dolor dirige update
  [meta/autoobs]   duda de DECISION que dispara busqueda

## v0.25 v20 — SUSTRATO REAL RESTRINGIDO (Don Quijote, palabra "tiempo")
Entrenamiento skip-gram sobre fragmentos acotados (~60 tokens por ocurrencia):
  tokens_sub=19620, fragmentos=327, vocab_sub=3847.
Resultado: embeddings densos coherentes en bajo recursos; top-15 vecinos de "tiempo"
con similitudes 0.60-0.71.
Veredicto: PARCIAL. El sustrato real funciona, pero sobre "tiempo" en Don Quijote no
se alcanza una separación de sentido clara; queda como pista para palabras con polisemia
más marcada o corpus mejor curado.

## v0.25 v2_core — MODULARIZACION (core + MetricLogger)
Se extraen clases reutilizables a `dscng_core.py` y `test_dscng_core.py`:
- Core: dot/norm/cos/softmax, SimpleTransformer, RootMemory, LinearSenseClassifier,
  SkipGram, build_polysemy_corpus, MetricLogger.
- Tests unitarios validados: dot 32, norm 5.0, cos 0.0, softmax suma 1.0,
  transformer contexto, root memoria, clasificador lineal, skip-gram+corpus.
- Script canónico `run_v25_v2_core.py` importa solo desde core y persiste métricas
  canónicas acc_pred/acc_gt/dolor/foco_acc/W_actual en `results_v25_v2_core.json`.
Resultado: acc_pred_avg=0.331, acc_gt_avg=0.331, foco_acc_avg=0.601, W_actual_avg=8.
Veredicto: modularizacion sin regresion; señal presente pero no suficiente para loop funcional.

## ESTADO CONSOLIDADO post-revalidacion completa 2026-07-30
Rerun real de v2-v21 sobre `dscng_core.py`:
- v2c MODULAR OK: acc_pred=0.331, acc_gt=0.331, foco_acc=0.601, W_actual=8.
- v2b ROOT sobre transformer NO FUNCIONAL: acc_gt_root=0.543, dolor_en_duda=0.000.
- v2d TRANSFORMER NO SEPARA SENTIDO: acc_gt_simple≈0.53, cos(A,B)≈0.57-0.93.
- v3/VAR BERT-like: no separa A/B sin más supervisión.
- v4 DUDA: dolor_en_duda=0.091; no distingue ambigüedad.
- v5 DUDA/CAMBIO: no detecta cambio; dolor_en_cambio=0.150 vs estable=0.504 sin señal útil.
- v6 ATENCIÓN SELECTIVA FUNCIONAL: acc_decision=0.890; separa A/B por vitalidad.
- v7 no disponible en home, queda documentado como pendiente de rerun.
- v8 LOOP EMPEORA: 0.758->0.550; v8b sin decoder empeora más (0.417).
- v9 FALLBACK sintético: loop mejora baseline débil 0.328->0.500.
- v10 CLASIFICADOR=0.766; LOOP=0.490 (destruye señal).
- v11 LOOP CONSERVADOR +0.137 sobre 'banco'; v11b NO GENERALIZA en 'llave' (1.000->0.500).
- v12 DECODER embeddings: NO FUNCIONAL (top1=0.020, top5=0.095).
- v13 TRANSICIÓN EXPLÍCITA: top1=0.850, FUNCIONAL.
- v14 SENTIDO A/B: pureza=1.000, generación coherente.
- v15 LOOP SENTIDO: acc_sense=0.938.
- v16 MEMORIA COMPETITIVA: coherencia sentido activo=0.750.
- v17 Don Quijote k-means: NO FUNCIONAL.
- v18 DQ "cabo": FUNCIONAL PARCIAL — k=3, tamaños 61/22/7, cohesión 0.42-0.62, overlap top-30=1. Hay estructura diferenciable, pero desbalanceado; queda como pista, no como generalización.
- v19 DQ "tiempo": señal parcial, k-means sugerente.
- v22 skip-gram "tiempo" DQ: embeddings coherentes, separación débil.
- v21 CLASSIFIER LOOP: 1.000 en régimen controlado sobre corpus sintético.

### v0.25 v22b — COHERENCIA SOBRE CORPUS REAL (Don Quijote, "tiempo")
Valida v22 sobre texto real Don Quijote (386k tokens):
- k-means sobre contextos de "tiempo": k=2, tamaños 147/180.
- Generación por cluster: scores de coherencia = 0.0 en ambos clusters (overlap vocabulario dominio = 0).
- VEREDICTO: NO FUNCIONAL sobre corpus real. El método k-means + bigramas sobre BoW locales no produce coherencia de dominio en Don Quijote, a diferencia del corpus sintético controlado (v22 FUNCIONAL).

### v0.25 v3-v6, v12 — REVALIDACIÓN SOBRE dscng_core.py
Revalidación real de experimentos sobre el core modular:
- v3_core BERT-style: acc_pred=0.022, acc_clf=0.057, cos(A,B)=0.92. NO FUNCIONAL.
- v4_core root duda: acc_decision=0.500 (azar), dolor_en_duda=1.0. NO FUNCIONAL.
- v5_core duda cambio: dolor_en_cambio=0.137 vs dolor_en_estable=0.150. NO FUNCIONAL.
- v6_core atención selectiva: acc_decision=0.291. NO FUNCIONAL sobre corpus sintético mezclado.
- v12_core decoder embeddings: top1=0.011, top5=0.117. NO FUNCIONAL.

### v0.25 v7/v7b/v7c — estado
Scripts `run_v25_v7*.py` no disponibles en home ni vault (solo JSON de resultados preservados). Documentados como pendientes de recuperación. Los resultados v7c (acc_gt=0.500 para "banco", repulsion_fuerte separa técnicamente pero no semánticamente) permanecen válidos.
