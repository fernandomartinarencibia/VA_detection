# Apartado d) — Análisis de pérdidas

**Asignatura:** APAI — Aprendizaje Profundo Aplicado a la Imagen  
**Proyecto 2B:** Detección de objetos sobre imágenes de satélite  
**Notebook:** `APAI_Pr2B_ObjectDetection_2025_2026.ipynb` — Celdas 94–100

---

## Motivación

Faster R-CNN optimiza simultáneamente cuatro pérdidas que controlan distintas etapas de la cadena de detección. Entender la contribución relativa de cada una y el efecto de modificar su peso permite diagnosticar cuellos de botella y orientar mejoras futuras. Esto es especialmente relevante en xBD, donde la clasificación fina de daño (4 categorías) es el objetivo principal y puede no estar suficientemente priorizada con los pesos por defecto.

---

## Las cuatro pérdidas de Faster R-CNN

| Índice | Pérdida | Etapa | Qué controla |
|--------|---------|-------|--------------|
| `w0` | `loss_objectness` | RPN | Si hay objeto en el anchor (clasificación binaria) |
| `w1` | `loss_rpn_box_reg` | RPN | Refinamiento de coordenadas del anchor |
| `w2` | `loss_classifier` | RoI head | Clasificación de la categoría de daño |
| `w3` | `loss_box_reg` | RoI head | Refinamiento de coordenadas de la propuesta final |

---

## Análisis 1 — Contribución relativa de cada pérdida

**Qué se hace:** Se reconstruyen las pérdidas de validación para cada una de las 18 épocas del entrenamiento base, cargando cada checkpoint guardado y pasándolo por `eval_one_epoch`. Se visualiza la evolución temporal y la contribución porcentual media en la segunda mitad del entrenamiento.

**Utilidad:** Revela qué componente domina la señal de entrenamiento. Si `loss_rpn_box_reg` (regresión de anchors) acapara la mayor parte de la pérdida total, el optimizador dedica más gradiente a ajustar coordenadas que a aprender a clasificar el tipo de daño. Este diagnóstico justifica los experimentos de ablación del análisis 2.

**Nota de implementación:** El `log.csv` generado por el bucle de entrenamiento estaba vacío porque el código solo escribe al CSV durante el entrenamiento activo; al existir los checkpoints, los carga sin registrar nada. La solución reconstruye el log iterando sobre los checkpoints con `map_location='cpu'` para evitar saturar la VRAM al cargar 18 modelos consecutivos.

---

## Análisis 2 — Ablación de pesos de pérdida

**Qué se hace:** Se entrena el modelo con 4 configuraciones de pesos distintas, partiendo del mejor modelo base (fine-tuning de 3 épocas con `lr × 0.1`) y midiendo el F1 resultante:

| Configuración | Pesos `[w0,w1,w2,w3]` | Hipótesis |
|---------------|----------------------|-----------|
| `w_base`   | `[1, 1, 1, 1]` | Referencia — pesos iguales |
| `w_no_obj` | `[0, 1, 1, 1]` | Sin objectness → la RPN es ciega al fondo/objeto |
| `w_no_cls` | `[1, 1, 0, 1]` | Sin clasificación → el RoI head no distingue categorías de daño |
| `w_cls_x3` | `[1, 1, 3, 1]` | Mayor peso en clasificación → énfasis en el objetivo principal del proyecto |

**Utilidad de cada configuración:**

- **`w_no_obj`** es el caso más revelador: si suprimir el objectness colapsa el detector, confirma que la RPN es el cuello de botella y que toda la cadena depende de que las propuestas de región sean correctas.
- **`w_no_cls`** aísla el efecto de la clasificación de daño. Si el F1 cae drásticamente, indica que el RoI head no puede apoyarse en las otras pérdidas para aprender categorías, y que `loss_classifier` es imprescindible.
- **`w_cls_x3`** es la configuración de interés práctico: en un dataset desequilibrado como xBD (mayoría de edificios sin daño), dar más peso a la clasificación puede mejorar la sensibilidad a las categorías de daño severo (`major-damage`, `destroyed`).

**Decisiones de diseño para reducir el coste computacional:**
- Fine-tuning desde el mejor modelo base (en lugar de entrenar desde cero): el efecto de los pesos es visible desde la primera época, sin gastar épocas aprendiendo características generales.
- 3 épocas por experimento: suficiente para observar divergencia entre configuraciones extremas.
- 4 experimentos en lugar de 7: se priorizan los casos con mayor valor analítico para la tarea de clasificación de daño.

---

## Conclusiones esperadas

- **`loss_objectness`** es la pérdida más crítica: suprimirla impide que la RPN proponga regiones de interés y colapsa el detector completo independientemente del resto de pesos.
- **`loss_classifier`** es la pérdida directamente ligada al objetivo del proyecto. Su supresión elimina la capacidad del modelo de distinguir categorías de daño, aunque la localización de edificios puede sobrevivir gracias a la RPN.
- **`loss_rpn_box_reg`** y **`loss_box_reg`** afectan principalmente a la precisión geométrica de las bounding boxes (IoU) pero no a la clasificación; su supresión degrada la localización sin destruir el clasificador.
- Enfatizar `loss_classifier` (`w_cls_x3`) puede mejorar el F1 cuando la clasificación de daño es el cuello de botella, al coste de una localización ligeramente menos precisa.
