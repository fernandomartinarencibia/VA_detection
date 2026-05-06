# Análisis del Primer Experimento: Detección de Daños con Faster R-CNN en xBD

**Asignatura:** APAI — Aprendizaje Profundo Aplicado a la Imagen  
**Proyecto 2B:** Detección de objetos sobre imágenes de satélite  
**Fecha:** 2026-05-06  
**Notebook:** `APAI_Pr2B_ObjectDetection_2025_2026.ipynb` — Parte 4

---

## 1. Descripción del problema y configuración experimental

El objetivo es adaptar Faster R-CNN al dataset **xBD** para detectar y clasificar edificios según su nivel de daño tras un desastre natural. A diferencia del Proyecto 1 (clasificación sobre patches 64×64 con localización conocida), aquí el modelo debe **localizar y clasificar simultáneamente** sobre la imagen completa de 1024×1024 píxeles.

Las cinco clases del problema son:

| Etiqueta | Clase | Descripción |
|----------|-------|-------------|
| 0 | `__background__` | Fondo (reservado internamente por Faster R-CNN) |
| 1 | `no-damage` | Edificio sin daño visible |
| 2 | `minor-damage` | Daño menor |
| 3 | `major-damage` | Daño mayor |
| 4 | `destroyed` | Edificio destruido |

**Configuración base:**
- Backbone: ResNet-50 FPN preentrenado en COCO (`pretrained=True`)
- `num_epochs = 18`, `lr = 0.001`, `step_size = 6` (StepLR ×0.1)
- Optimizer: SGD, `momentum=0.9`, `weight_decay=0.0005`
- `batch_size = 1` (restricción del código de `torchvision_05`, detallada en §3.1)
- Threshold de evaluación: `TH_SCORE = 0.5`, `TH_IOU = 0.5`

---

## 2. Análisis del dataset

### 2.1 Splits disponibles

| Split | Imágenes | Edificios etiquetados |
|-------|----------|-----------------------|
| Train | 256      | 61.171 |
| Val   | 45       | 8.495  |
| Test  | 63       | 0 (**sin etiquetas de daño**) |

> **Observación crítica sobre el split de test:** El conjunto de test público de xBD no contiene anotaciones de nivel de daño (todos los edificios aparecen como `unlabelled`). Esto impide cualquier evaluación cuantitativa significativa sobre test. **Todo el análisis de rendimiento se basa exclusivamente en el conjunto de validación.** Este es un punto relevante para la memoria: los resultados de test (P=0, R=0, F1=0) no reflejan el rendimiento real del modelo, sino la ausencia de etiquetas de referencia.

### 2.2 Desbalance de clases

La distribución en train muestra un **desbalance extremo** entre clases:

| Clase | Edificios (train) | Porcentaje |
|-------|-------------------|------------|
| no-damage    | 50.928 | **83.3 %** |
| minor-damage |  4.659 |  7.6 % |
| major-damage |  2.357 |  3.9 % |
| destroyed    |  3.227 |  5.3 % |

La clase dominante (`no-damage`) supera en un factor ×10 a las demás. En un escenario de detección de daños en crisis, las clases más raras (destruido, daño mayor) son precisamente las de mayor relevancia operativa. Este desbalance es una de las principales fuentes de dificultad del problema.

### 2.3 Distribución de tamaños de edificios

Analizados sobre las imágenes de train (1024×1024 px):

| Estadístico | Valor (px) |
|-------------|-----------|
| p5          | 9.5  |
| p25         | 17.7 |
| **mediana** | **26.7** |
| p75         | 35.9 |
| p95         | 48.0 |

El aspecto mediano (ancho/alto) es **1.00** — los edificios son prácticamente cuadrados. El rango p5–p95 se sitúa entre 9 y 48 píxeles. En comparación, los objetos de PASCAL VOC o COCO típicamente ocupan entre el 10% y el 50% de la imagen, mientras que aquí la mediana representa solo el 2.6% del lado de la imagen. Este es el fundamento para usar anchors personalizados (§3.2).

---

## 3. Decisiones de diseño

### 3.1 `batch_size = 1` — restricción de `torchvision_05`

La versión modificada del profesor (`torchvision_05`) incluye una modificación en `rpn.py:filter_proposals` que devuelve los anchors usados durante la inferencia (información no disponible en torchvision estándar). Sin embargo, la implementación contiene un bug al manejar lotes de tamaño > 1:

```python
# rpn.py (torchvision_05), función filter_proposals:
anchors = anchors[0]              # [N, 4] — toma solo la imagen 0
anchors = anchors[top_n_idx, :]  # top_n_idx es [batch, K] → [batch, K, 4]
for ...:
    anchors = anchors.squeeze(0)[keep, :]  # squeeze(0) no actúa si batch > 1
                                            # keep tiene valores 0..2000 → OOB en GPU
```

Con `batch_size > 1`, el `squeeze(0)` no elimina la dimensión de batch (que tiene tamaño > 1), y `keep` — con índices hasta ~2000 — intenta indexar una dimensión de tamaño 2, causando un error CUDA `device-side assert` en el kernel de gather. La solución es forzar `batch_size = 1`, lo que tiene un coste mínimo en throughput dado que las imágenes son de 1024×1024 px.

### 3.2 AnchorGenerator personalizado

Los anchors por defecto de Faster R-CNN (`[32, 64, 128, 256, 512]`) están calibrados para COCO, donde los objetos son mucho más grandes. Dado que la mediana de los edificios en xBD es ~27 px y el p95 es ~48 px, se usan:

```python
AnchorGenerator(
    sizes=((8,), (16,), (32,), (64,), (128,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5  # 5 niveles FPN
)
```

| Anchor | Cobre |
|--------|-------|
| 8 px   | Edificios muy pequeños (p5 ≈ 9.5 px) |
| 16 px  | Cuartil inferior (p25 ≈ 17.7 px) |
| 32 px  | Mediana (26.7 px) |
| 64 px  | Cuartil superior (p75–p95) |
| 128 px | Edificios grandes y outliers |

Los aspect ratios `(0.5, 1.0, 2.0)` cubren el rango observado de 0.55 a 1.82.

### 3.3 Normalización de imagen

Las imágenes xBD son `uint16` con valores efectivos en el rango 0–510 (no 0–65535). Dividir por `np.iinfo(uint16).max` deja los píxeles en ~0.008, lo que produce feature maps con valores extremos en el backbone, propuestas NaN en el RPN y fallo de CUDA en `batched_nms`. La corrección es:

```python
image = np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)
```

Coherente con el tratamiento del Proyecto 1.

---

## 4. Resultados del entrenamiento

### 4.1 Evolución de la pérdida

Resultados completos por época (evaluación sobre validación):

| Época | Train Loss | Val Loss | Precisión | Recall  | F1     |
|-------|-----------|---------|-----------|---------|--------|
| 0     | 47.28     | 45.60   | 0.258     | 0.002   | 0.004  |
| 1     | 54.68     | 45.88   | 0.600     | 0.057   | 0.104  |
| 2     | 48.20     | 38.54   | 0.645     | 0.023   | 0.044  |
| 3     | 47.58     | 44.64   | 0.705     | 0.041   | 0.078  |
| 4     | 44.20     | 38.97   | 0.507     | 0.020   | 0.038  |
| **5** | **50.26** | **31.80**| **0.675** | **0.075**| **0.135** |
| 6 *(LR×0.1)* | 35.06 | 39.50 | 0.651 | 0.061 | 0.111 |
| 7     | 37.91     | 39.81   | 0.623     | 0.034   | 0.064  |
| 8     | 43.79     | 40.61   | 0.552     | 0.040   | 0.075  |
| 9     | 42.51     | 38.72   | 0.609     | 0.044   | 0.083  |
| 10    | 43.80     | 39.77   | 0.577     | 0.039   | 0.073  |
| 11    | 43.96     | 40.23   | 0.501     | 0.043   | 0.079  |
| 12 *(LR×0.01)* | 42.30 | 40.16 | 0.500 | 0.041 | 0.076 |
| 13    | 42.59     | 40.15   | 0.515     | 0.041   | 0.077  |
| 14    | 42.85     | 39.39   | 0.508     | 0.041   | 0.076  |
| 15    | 42.13     | 38.58   | 0.538     | 0.039   | 0.073  |
| 16    | 42.26     | 38.96   | 0.526     | 0.038   | 0.071  |
| 17    | 41.78     | 37.94   | 0.510     | 0.039   | 0.072  |

**Mejor modelo:** época 5, F1 = **0.135** en validación.

### 4.2 Resultados finales (mejor modelo, validación)

```
Precision global : 0.675
Recall global    : 0.075
F1 global        : 0.135
GT total         : 8.495 edificios
TP               : 635
Predicciones     : 941
```

### 4.3 Resultados en test

```
Precision : 0.000  |  Recall : 0.000  |  F1 : 0.000
GT = 0  (test sin anotaciones de daño)
```

Los resultados nulos en test no son un indicador de fracaso del modelo sino una consecuencia de que el split de test de xBD no tiene anotaciones de nivel de daño. El modelo sigue detectando 3.036 edificios, pero no hay ground truth con el que comparar.

---

## 5. Análisis y diagnóstico

### 5.1 Pérdida RPN (rpn_box_reg) dominante

La pérdida `rpn_box_reg` representa aproximadamente el **90% del total** en todas las épocas:

| Componente de pérdida | Época 5 (val) | Proporción |
|-----------------------|--------------|------------|
| `rpn_box_reg`         | 30.93        | **97.2 %** |
| `classifier`          | 0.68         | 2.1 % |
| `box_reg`             | 0.19         | 0.6 % |
| `objectness`          | 0.008        | 0.02 % |

Esto indica que el **RPN está teniendo muchas dificultades para ajustar las coordenadas de los anchors** a los edificios reales. Las causas probables son:

1. **Los anchors predefinidos aún no están bien calibrados** para el rango exacto de tamaños del dataset. Los 8 px son muy pequeños y producen un número enorme de anchors en P2 del FPN (200×200×3 ≈ 120.000 anchors por imagen), la mayoría irrelevantes.
2. **La imagen completa (1024×1024) redimensionada a 800 px** reduce los edificios medianos de 26 px a ~20 px, en el límite inferior de lo que el RPN puede regresar con fiabilidad.
3. El **backbone preentrenado en COCO** tiene sesgos hacia objetos más grandes, y la distribución de edificios satelitales es muy distinta.

### 5.2 Alta precisión, recall muy bajo

El modelo muestra un comportamiento muy conservador: cuando predice, suele acertar (precisión ≈ 0.5–0.7), pero detecta una fracción muy pequeña de los edificios reales (recall ≈ 0.02–0.075).

Causas probables:
- **Umbral `TH_SCORE = 0.5` demasiado alto** para un modelo que no ha convergido. El RPN con anchors pequeños asigna scores de objectness bajos incluso a propuestas correctas. Bajar el umbral a 0.3 o 0.2 mejoraría el recall significativamente.
- **Dominio muy diferente al preentrenamiento (COCO)**: el backbone tarda muchas épocas en adaptarse a texturas de satélite.
- **18 épocas insuficientes** para convergencia dada la magnitud de la pérdida rpn_box_reg (~40 vs. <1 en COCO fine-tuning típico).

### 5.3 Efecto del scheduler StepLR

La reducción de LR en la época 6 (×0.1, de 0.001 a 0.0001) coincide con una **caída brusca del F1** (de 0.135 a 0.064). Esto es contraintuitivo: normalmente un LR menor ayuda a refinar. La explicación probable es que el modelo aún estaba en una fase de búsqueda activa del espacio de parámetros y el LR más pequeño lo "congela" en un estado subóptimo. En épocas 12–17 (LR=1e-5) las métricas se estabilizan alrededor de F1≈0.07 sin mejorar.

**Conclusión:** el schedule actual (paso a LR×0.1 en época 6 de 18) es demasiado agresivo para este problema. Un schedule más tardío (paso en época 12) o warm-up previo daría mejores resultados.

### 5.4 Desbalance de clases y métricas por clase

Con el desbalance 83%-7.6%-3.9%-5.3%, el modelo tiene incentivos para detectar únicamente edificios `no-damage`. En los TP observados durante validación, la mayoría pertenecen a esa clase. Las clases minoritarias (`minor-damage`, `major-damage`, `destroyed`) no tienen suficientes ejemplos positivos en un batch de 1 imagen para que el RoI head aprenda a distinguirlas del fondo.

---

## 6. Propuestas de mejora (siguiendo las pautas del profesor)

### a) Reducir el umbral de confianza

Antes de cambiar arquitectura, evaluar con `TH_SCORE ∈ {0.1, 0.2, 0.3, 0.4}` para construir la curva Precisión-Recall. En detección de crisis, un recall alto (no perder edificios dañados) suele ser prioritario sobre la precisión.

### b) Optimización de anchors con k-means

El profesor propone aplicar k-means a los tamaños de las bounding boxes de train para obtener anchors óptimos para xBD. Con los datos observados:

```python
# Datos del dataset: sqrt(área) → p5=9.5, p25=17.7, med=26.7, p75=35.9, p95=48
# k-means con k=5 sobre (widths, heights) daría clusters aproximados:
# (9, 9), (17, 18), (26, 27), (35, 36), (48, 47) → anchors más ajustados
```

Esto debería reducir drásticamente la pérdida `rpn_box_reg` inicial.

### c) Gestión del desbalance de clases

Dos estrategias complementarias:
1. **Focal Loss** en la cabeza de clasificación del RoI: penaliza menos los ejemplos fáciles (no-damage bien predicho) y se centra en los difíciles (major-damage, destroyed).
2. **Oversampling** de imágenes con más edificios dañados durante el shuffle del DataLoader.

### d) Schedule de LR menos agresivo

Reemplazar StepLR con `CosineAnnealingLR` o usar un step_size mayor:

```python
# Opción 1: reducción más tardía
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
# Opción 2: coseno sin pasos bruscos
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
```

### e) Más épocas de entrenamiento

Con solo 256 imágenes y la alta pérdida rpn_box_reg inicial, el modelo necesita más iteraciones para que el backbone adapte sus filtros al dominio satelital. Se recomienda al menos 40–60 épocas con el LR correcto.

### f) Análisis por tipo de desastre

El split de train cubre 7 desastres distintos (`joplin-tornado`, `moore-tornado`, `tuscaloosa-tornado`, `sunda-tsunami`, `portugal-wildfire`, `pinery-bushfire`, `nepal-flooding`). Cada tipo de desastre produce patrones de daño muy diferentes. Estratificar la evaluación por tipo de desastre identificaría qué escenarios son más difíciles para el modelo.

---

## 7. Resumen ejecutivo

| Aspecto | Estado | Prioridad de mejora |
|---------|--------|---------------------|
| Dataset train/val operativo | ✓ | — |
| Test sin anotaciones de daño | ⚠ | Usar val como referencia |
| Mejor F1 (val)               | 0.135 (época 5) | Alta |
| Recall muy bajo (0.075)      | ⚠ | Reducir TH_SCORE |
| rpn_box_reg dominante (~97%) | ⚠ | k-means anchors |
| Desbalance extremo (83% no-damage) | ⚠ | Focal Loss |
| StepLR demasiado agresivo    | ⚠ | CosineAnnealing |
| batch_size=1 (bug torchvision_05) | ✓ (documentado) | — |

El experimento base establece una línea de referencia funcional. El modelo aprende a detectar edificios con precisión razonable pero recall muy bajo, lo que lo hace inapropiado para uso real en gestión de emergencias. Las mejoras propuestas atacan los tres problemas raíz: anchors mal calibrados, umbral demasiado restrictivo y desbalance de clases no gestionado.
