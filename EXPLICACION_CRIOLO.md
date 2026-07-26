# DSCN-G: qué construimos, explicado en criollo

Esto es para explicarle a cualquiera qué hicimos, sin palabras raras.

## La idea de Luciano, en una frase
Que la inteligencia no es "adivinar la próxima palabra" (como hacen los chatbots
de moda), sino un sistema que TIENE MEMORIA que no se borra, que CLASIFICA lo que
procesa, y que SIENTE un DOLOR que lo hace corregirse para sobrevivir. Eso es el
"sustrato cognitivo": el andamiaje de una mente, no el autocomplete.

## ¿Qué es el grafo?
Imaginá un montón de bolitas conectadas por hilos. Cada bolita es una palabra o
un concepto. Los hilos son "esta palabra suele ir con esta otra". Cuando el
sistema lee "el gato", las bolitas de "el" y "gato" se acercan porque van juntas.
Eso es todo: el grafo APRENDE acercando bolitas que van juntas.

El grafo tiene tres trucos que probamos que funcionan:

1. MEMORIA QUE NO SE BORRA (v0.3)
   Las bolitas que no usa mucho no se eliminan: se duermen (hibernado). Cuando
   las necesita, las despierta. El sistema puede tener millones de bolitas
   "durmiendo" y solo un puñado despierto trabajando. Nada se pierde. Es como
   vos: no pensás en "trompeta" todo el día, pero la sabés cuando la necesitás.

2. ETIQUETAS QUE CAMBIAN SOLAS (v0.9b)
   El sistema solo, sin que nadie se lo diga, aprende que unas palabras son
   "cosas" (gato, casa) y otras son "acciones" (correr, decir). No se lo
   enseñamos con una lista: lo deduce por cómo se usan. Acierta 92 de cada 100.
   Es como cuando un pibe aprende que "perro" es una cosa y "ladrar" es lo que
   hace, sin que le expliquen gramática.

3. DOLOR QUE LO HACE CORREGIRSE (v0.9c)
   El sistema tiene una "vitalidad". Cuando se confunde (sus bolitas no encajan),
   la vitalidad baja: ESO es el dolor. No es un castigo de afuera, es que el
   sistema se siente mal a sí mismo. Y cuando le va mal, cambia sus bolitas para
   que no le pase otra vez. En el experimento: sin ese mecanismo, el sistema "muere"
   (vitalidad 0); con él, se salva (vitalidad 1). Exactamente lo que dijo Luciano:
   el dolor es la señal que obliga al sistema a cambiar para evitar lo que lo daña.

## ¿Qué NO pudo hacer solo?
El contexto. Un grafo de bolitas es bueno para "esta palabra va con esta", pero
NO entiende "la misma palabra según lo que la rodea". Ejemplo: "banco" puede ser
el del dinero o el de sentarse. El grafo le da a "banco" UNA bolita, y no puede
tener los dos sentidos a la vez. Para eso necesita una capa extra que mire el
CONTEXTO (lo que viene antes). Eso es lo que hacen los transformers (la tecnología
de los chatbots).

Intentamos pegarle esa capa de contexto al grafo. No anduvo del todo en este
telefonito porque:
- El grafo aplana las bolitas (todas terminan parecidas) y la capa de contexto no
  tiene con qué trabajar.
- Entrenar la capa de contexto bien requiere una herramienta (PyTorch) que no
  entra en el celular.
Pero PROBAMOS que la idea funciona: escribimos el aprendizaje a mano (sin la
herramienta) y la pérdida bajó, o sea el sistema aprendía. Solo necesita más
tamaño del que probamos.

## En resumen, ¿qué tenemos?
Un sistema pequeñito que:
- No olvida (duerme lo que no usa).
- Clasifica solo lo que lee.
- Siente dolor y se corrige para seguir vivo.
- Aprende de textos reales.

Lo que le falta para hablar como un humano es la capa de contexto (los
transformers), que es otra herramienta encima de este andamiaje. El grafo es el
CEREBRO que recuerda y siente; el transformer sería la forma de hablar fluido.

## Por qué importa
Los chatbots de hoy son autocomplete gigante: adivinan la próxima palabra, pero
no "sienten" ni "recuerdan" de verdad. Lo que Luciano propone es distinto: un
sustrato que tiene memoria persistente, categoriza y siente dolor. Eso es el
primer paso hacia algo que no sea solo predecir, sino que tenga una especie de
"vivencia". No es una IA completa, es el sustrato de una. Y lo probamos con
números reales, no humo.
