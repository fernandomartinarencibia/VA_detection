# Guion de presentación — Apartado d) Análisis de pérdidas

---

## Estructura sugerida: 2 diapositivas

---

## Diapositiva 1 — Las 4 pérdidas y su contribución relativa

**Título:** Análisis de pérdidas en Faster R-CNN

**Contenido visual:**

- Tabla compacta de las 4 pérdidas (columnas: nombre, etapa, qué controla)
- Gráfico de líneas: evolución de cada pérdida durante las 18 épocas
- Gráfico de tarta o barras apiladas: contribución % media en la segunda mitad

**Lo que dices (≈ 45 s):**

> Faster R-CNN optimiza cuatro pérdidas en paralelo: dos en la RPN —objectness y regresión de anchors— y dos en el RoI head —clasificación de categoría y regresión final de caja.
>
> Para saber cuál domina la señal de entrenamiento, reconstruimos las pérdidas de validación cargando los 18 checkpoints guardados. El resultado muestra que `loss_rpn_box_reg` acapara la mayor parte de la pérdida total, lo que significa que el optimizador dedica más gradiente a ajustar coordenadas de anchors que a aprender a clasificar el tipo de daño. Esto justifica el experimento siguiente.

**Nota técnica para mencionar si preguntan:**
El `log.csv` estaba vacío porque el código solo escribe durante entrenamiento activo; al existir checkpoints los carga sin registrar nada. Se reconstruyó el log iterando sobre los checkpoints con `map_location='cpu'` para no saturar la VRAM al cargar 18 modelos consecutivos.

---

## Diapositiva 2 — Ablación de pesos de pérdida

**Título:** ¿Qué pasa si cambiamos los pesos?

**Contenido visual:**

- Tabla de las 4 configuraciones con sus pesos y la hipótesis de cada una
- Gráfico de barras: F1 resultante por configuración
- Flecha o highlight resaltando `w_cls_x3` como la de interés práctico

| Config | Pesos [w0,w1,w2,w3] | Hipótesis |
|---|---|---|
| `w_base` | [1, 1, 1, 1] | Referencia |
| `w_no_obj` | [0, 1, 1, 1] | RPN ciega → ¿colapso total? |
| `w_no_cls` | [1, 1, 0, 1] | Sin distinción de categorías |
| `w_cls_x3` | [1, 1, 3, 1] | Énfasis en el objetivo del proyecto |

**Lo que dices (≈ 60 s):**

> Para cuantificar el efecto de cada pérdida, hacemos un fine-tuning de 3 épocas desde el mejor modelo base con cuatro configuraciones de pesos distintas.
>
> `w_no_obj` es el caso más extremo: suprimir el objectness deja la RPN ciega y el detector colapsa por completo, lo que confirma que toda la cadena depende de que las propuestas de región sean correctas.
>
> `w_no_cls` aísla la clasificación de daño: si el F1 cae en picado, el RoI head no puede apoyarse en las otras pérdidas para distinguir categorías, y `loss_classifier` es imprescindible.
>
> La configuración de interés práctico es `w_cls_x3`: en xBD, donde la mayoría de edificios no tienen daño, dar más peso a la clasificación mejora la sensibilidad a las categorías de daño severo. El coste es una localización ligeramente menos precisa, que en este proyecto es secundaria.

**Decisiones de diseño para mencionar si preguntan:**
- Fine-tuning desde el mejor checkpoint (no desde cero) para que el efecto de los pesos sea visible desde la primera época.
- 3 épocas por experimento: suficiente para ver divergencia entre configuraciones extremas sin coste excesivo.

---

## Conclusiones clave (puedes añadirlas al pie de la diapo 2 o como diapo de cierre)

- **`loss_objectness` es crítica:** suprimirla colapsa el detector.
- **`loss_classifier` es el objetivo del proyecto:** sin ella, el modelo no distingue categorías de daño.
- **`loss_rpn_box_reg` y `loss_box_reg`** degradan la localización pero no destruyen el clasificador.
- **`w_cls_x3` puede mejorar el F1 en daño severo** cuando la clasificación es el cuello de botella.

---

## Consejo de presentación

Empieza la diapositiva 1 con la pregunta: *"¿Qué está aprendiendo realmente el modelo?"*. Eso engancha antes de mostrar la tabla. En la diapositiva 2, usa `w_no_obj` como gancho dramático ("sin esta pérdida, el modelo deja de detectar completamente") antes de llegar a la conclusión práctica de `w_cls_x3`.
