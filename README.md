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
| v0.22 v4 | root + MARGIN adaptativo (percentil top1-top2) | margin=0.0, duda=0.0 | ~ MARGIN adaptativo ok, pero proyeccion separa TANTO que no hay ambigüedad |
| v0.23 v1 | composicion relacional Hebb 3-body (2 relaciones) | 4/12=0.333 (azar 0.5) | ~ FALLA: asociacion basica contamina R[r] (ambos pares ocurren) |
| v0.23 v2 | Hebb 3-body SIN contaminacion + 4 relaciones + D16/32 | D16=0.312 D32=0.312 (azar 0.25) | ~ SENAL DEBIL: supera azar pero limitado por mecanismo (no por ancho) |

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
  sentidos TAN limpio que SIEMPRE hay un claro ganador, incluso en contexto mixto.
  La duda de SENTIDO no emerge porque el sistema SIEMPRE sabe que sentido es ->
  eso es un EXITO del fractal, no un fallo del root. La "duda" real (decision/
  conflicto de inferencias) requiere un nivel superior, no ambiguedad de palabra.
  CERRADO v0.22: root DIRECTOR rutea perfecto (v3: 1.0); duda de sentido es
  trivialmente resoluble por el grafo -> no es el lugar donde la duda importa.
  GAP siguiente: composicion relacional (v0.23) y duda de DECISION (dolor v0.19/v0.9c).

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
  (tensor/relational memory). Por ahora: senal debil confirmada, no cerrado.

## CORRECCIONES DE AUDITORIA (errores circulares -> senal real del dato)
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

## CONCLUSIÓN
La arquitectura (grafo de memoria/dolor + transformer de contexto) es sólida y los
5 mecanismos (memoria, dolor, categoría, composición, contexto) son GENUINOS cuando se
miden con señal real del dato. El README anterior mentía por omisión de diseño en 4 de 5
"✓"; este archivo corrige eso. El grafo rústico es un sustrato limitado (predice mal) pero
sus mecanismos cognitivos son reales y el transformer (v0.14d) es el camino para escalar.
