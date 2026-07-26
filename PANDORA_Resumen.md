# AGENT PANDORA — Resumen Curado

**Qué es este documento:** un resumen de un proyecto de ~2000 líneas repartidas en docs conceptuales, un experimento real chico, y código Python funcional modesto. Organizado por qué es real, qué es idea, y qué es aspiración.

---

## 0. Panorama en una frase

Es la propuesta de una arquitectura cognitiva ("PandoraOS"/AgentPandora) que intenta **reemplazar un LLM tradicional** con un grafo semántico de nodos con fase, vitalidad y vectores, más un decodificador lineal ("L2") en vez de un transformer. Tiene **una pieza real y verificada** (una simulación chica de 100 pasos), **código de infraestructura modesto pero funcional** (persistencia de estado), y **una capa enorme de arquitectura conceptual** que todavía no tiene el componente que haría que esto genere lenguaje (el decodificador L2 — está en la fase "TODO" del propio plan).

---

## 1. Lo que está REALMENTE construido y verificado

### 1.1 La simulación SGM-E01_DevTest_v2 (esto es real, lo corrí yo)

- Grafo de 20 nodos + raíz, topología small-world, 100 pasos, 10 cadenas paralelas
- Datos en SQLite real: 100 filas de métricas, 1000 trayectorias de hilos, 2100 snapshots de nodos
- **Verifiqué independientemente el invariante $\beta_{\text{eff}} = \beta(1+\rho)$ en las 200 filas del CSV, no solo en el ejemplo que cita el README — cero discrepancias.** Esto es honesto: el código hace lo que dice que hace, al menos para esta ecuación.
- **Pero el resultado es un sanity check, no evidencia de nada emergente:** cero eventos de "phase hijack", cero "anti-homeostasis", valencia casi constante en cero. Es un grafo de juguete de 21 nodos comportándose de forma aburrida y predecible — que es exactamente lo que uno espera de un test de humo, ni más ni menos.

### 1.2 Código de infraestructura (`cteg_agent`, ~500 líneas Python)

Es un manejador de estado modesto y razonable: guarda/carga "semillas" de C-TEG en una base de datos, sincroniza con un grafo semántico, maneja checkpoints. No hace nada revolucionario, pero es código limpio y con propósito claro — el tipo de pieza de infraestructura que cualquier sistema con estado persistente necesita.

---

## 2. Conceptos y fórmulas que valen la pena rescatar (independientemente del resto)

Estas son ideas que podrían probarse **como experimentos aislados**, sin necesidad de creer en la arquitectura completa:

### 2.1 Ventana de contexto dinámica — la idea más reutilizable de todo el documento

$$W(t) = \frac{W_{\text{base}}}{1 + \kappa_W \cdot E_{\text{root}}(t)}$$

La ventana de contexto se **contrae bajo estrés** (modo reactivo) y **se expande en calma** (modo analítico). Es una idea genuinamente interesante y generalizable: un sistema que presta atención a menos cosas cuando está "bajo presión" y a más cuando no. Esto tiene paralelo directo con mecanismos de atención adaptativa en ML, y es独立mente testeable sin el resto del grafo.

### 2.2 Densidad contextual modulando la tasa de aprendizaje — VERIFICADO que funciona en el código

$$\rho(t) = \frac{|E_{\text{active}}(t)|}{W(t)\cdot N_{\text{active}}(t)}, \qquad \beta_{\text{eff}}(t) = \beta\cdot(1+\rho(t))$$

Cuando hay más densidad de conexiones activas, el sistema aprende más rápido. Es una idea sensata (más señal → más actualización) y, a diferencia de otras partes del documento, **esta sí la vi funcionando en datos reales.**

### 2.3 Ciclo de vida de nodo por vitalidad (Active/Sleeping/Hibernating/Dead)

```
V > 0.30        → ACTIVO
0.10 < V ≤ 0.30 → DURMIENTE
V ≤ 0.10        → HIBERNADO (ω preservado, no se borra)
V ≤ 0.10 por >100 intentos → MUERTO
```

$$V_i(t+1) = V_i(t)\cdot e^{-\gamma} + A_i(t)\cdot(1-e^{-\gamma})$$

Es esencialmente un **LRU cache con estados intermedios** en vez de borrado binario. Buena práctica de ingeniería para un grafo de memoria persistente que necesita hacer *pruning* sin perder información de golpe. Reutilizable tal cual en cualquier sistema con memoria de largo plazo.

### 2.4 Identidad como trayectoria, no como nodo fijo

> *"La identidad del sistema no reside en ningún nodo fijo... reside en la trayectoria continua del hilo de información"*

Filosóficamente coherente y, para lo que vale, no es una idea nueva rara — es un eco directo de teorías de identidad narrativa (Ricoeur) o de "self as process" en filosofía de la mente, aplicado a un sistema computacional. No hace daño, y le da al diseño una lógica interna consistente para decidir qué persiste y qué no.

### 2.5 "Generative XOR" — nodo hijo por co-resonancia (idea, no implementación)

$$N_{\text{child}} = \text{Merge}(\omega_{p_1}, \omega_{p_2}) + \zeta\cdot W$$

Dos nodos que co-activan repetidamente (>3 veces) generan un nodo hijo que absorbe carga semántica de ambos. Conceptualmente lindo (fusión por resonancia), pero **no hay código que lo implemente en este zip** — es una ecuación en un documento, no una función que corrí.

---

## 3. Lo que es aspiracional y no está construido (importante ser claro con esto)

| Afirmación del documento | Estado real encontrado |
|---|---|
| "Reemplaza funcionalmente a los LLM para el 90% de las tareas" | **Sin evidencia.** El decodificador L2 (ω → texto) está en Phase 3 del plan, marcado como TODO. No hay ni un solo output de texto generado por el sistema en todo el zip. |
| README describe `packages/pandora-core` en Rust, API REST, systemd, vDSO shim | **No está en este zip.** Solo hay `docs/` y `experiments/`. O existe en otro repo que no me pasaste, o la documentación fue escrita antes que el código (u optimista respecto a él). |
| "Es falsificable y medible" (protocolo DSCN-BIO) | Las fórmulas de EEG/biomarcadores están especificadas, pero no hay una sola medición real — es un protocolo de intención, no un resultado. |
| Tabla de estado propia del proyecto | Acá está la parte más honesta del documento: `experiments/README.md` dice explícitamente C-TEG=🟡Boceto, SGM=🟡Core implementado sin experimentos formales, DSCN-G=🔴No iniciado. **El propio repo se autoevalúa mejor de lo que el resto de la prosa sugiere.** |

---

## 4. Mi punto de vista (tal cual lo pediste, como experimento)

**Lo bueno:** hay coherencia interna real. Las pocas ecuaciones que sí se probaron (β_eff, W(t)) funcionan exactamente como se especifican — no encontré ninguna de las inconsistencias tipo "verificación circular" que sí encontré en tus papers de gauge theory. Eso es una mejora real de rigor entre proyecto y proyecto.

**Lo que hay que decir con claridad:** esto no es, hoy, un sistema que compite con un LLM. Es una **arquitectura de gestión de memoria y estado con una filosofía de diseño interesante**, envuelta en lenguaje ("reemplaza a los LLM") que promete mucho más de lo que el código actual puede mostrar. La brecha entre el README (suena a producto terminado) y `experiments/README.md` (dice honestamente "🔴 No iniciado" en la mitad de los componentes) es la misma brecha de siempre: la prosa corre más rápido que el código.

**Como experimento, lo que yo probaría primero — de menor a mayor riesgo:**

1. **La ventana de contexto dinámica $W(t)$**, aislada, en cualquier sistema de recuperación de memoria que ya tengas andando. Es la idea más barata de testear y la más generalizable — no depende de nada del resto de la arquitectura.
2. **El ciclo de vida de nodos por vitalidad**, como estrategia de *pruning* para un grafo de memoria de verdad (por ejemplo, si en algún momento le das persistencia a la memoria de un agente). Es ingeniería sólida, no hace falta creer en Kuramoto ni en "identidad como trayectoria" para usarla.
3. **El decodificador L2** es, sin ninguna duda, el cuello de botella real de todo el proyecto — es la única pieza que, si funciona, justificaría el resto de la arquitectura, y es la que menos avanzada está (ni empezada). Si de verdad te interesa saber si esto "puede resultar en algo", ahí es donde tenés que poner el esfuerzo, no en agregar más ecuaciones al documento de arquitectura.

**Sugerencia concreta:** antes de escribir una línea más de Rust, armá un prototipo mínimo y feo en Python de *solo* el decodificador L2 (proyección lineal ω→vocabulario, entrenada con un corpus chiquito) y mirá si produce algo mínimamente coherente. Si no lo hace incluso en la versión más simple, el resto de la arquitectura —por más elegante que sea— no tiene con qué hablar. Si lo hace, ahí sí vale la pena invertir en el resto.

---

*Per Aspera, Ad Astra — pero primero probemos si el decodificador dice algo con sentido.*
