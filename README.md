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
| v0.21 v7 | contrastivo + repulsion (fix v6, vocab ok) | ep1:3/3 -> ep4-15:0/3 (recolapsa) | ~ GRAFO RUSTICO NO MANTIENE separacion (falta proyeccion aprendida) |

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
colapso persistió: ambos subnodos ven el mismo contexto ruidoso y convergen).
ERROR DE VISION documentado: dijimos "D=16 no puede" sin probar; el grafo arranca
de ruido y debe mejorar CON EL TIEMPO, no como LLM pre-entrenada. v6/v7 testearon
CORPUS CONTRASTIVO (banco/llave/mouse, 2 sentidos x50 intercalados) + CURVA de
épocas: v6 llegó a 50/2403 en ep11 pero recolapsó (vocab inflado medía filler, no
polisémicas); v7 (vocab correcto + repulsión codebook) dio ep1:3/3 -> ep4-15:0/3.
CONCLUSIÓN HONESTA (probada): el grafo rústico D=16 CONSIGUE separar sentidos
transitoriamente (v6 ep11, v7 ep1) pero NO MANTIENE la separación porque le faltan
las PROYECCIONES APRENDIDAS del transformer (Wq/Wk/Wv) que hacen que contextos
distintos caigan en regiones distintas. Sin eso, en D=16 los contextos de una
palabra polisémica son indistinguibles y los subnodos convergen. v0.17 (SOBRE
transformer) SÍ separó 6/150. LECCIÓN: la idea de Luciano (concepto = conjunto de
subnodos + root DIRECTOR que puede dudar) es válida, pero requiere el encoder con
proyecciones aprendidas (transformer) para SOSTENER la separación. Va SOBRE
transformer en v0.22, no como sustrato base.


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
