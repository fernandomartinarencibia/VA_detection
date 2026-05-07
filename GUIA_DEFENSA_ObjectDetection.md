# Guía de Defensa — Proyecto 2B: Object Detection con Faster R-CNN
### Visión por Computadora · Universidad Carlos III de Madrid · Curso 2025-2026

---

## 🗺️ 1. RESUMEN DEL PIPELINE

### Visión global del proyecto

El proyecto implementa un sistema completo de **detección de objetos multi-clase** usando **Faster R-CNN** con backbone **ResNet-50 + FPN**, abordando dos escenarios progresivos:

| Fase | Dataset | Tarea | Clases |
|------|---------|-------|--------|
| Partes 1-3 | PASCAL VOC 2012 | Detección de objetos cotidianos | bottle, chair, dining table, sofa |
| Parte 4 | xBD (satélite) | Evaluación de daños en edificios | no-damage, minor-damage, major-damage, destroyed |

---

### Flujo completo del pipeline (de extremo a extremo)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. DATOS                                                               │
│  Raw images (TIFF 1024×1024 / JPEG)  +  anotaciones (WKT / XML)        │
│                    │                                                    │
│                    ▼                                                    │
│  2. DATASET CLASS                                                       │
│  xBDDataset → xBDDetectionDataset (o xBDDetectionDatasetAug)           │
│  • Lee imágenes post-desastre en patches (1024px)                       │
│  • Parsea polígonos WKT → bounding boxes [x1,y1,x2,y2]                │
│  • Etiqueta daño (1-4) y desplaza a +1 (0 = background para FRCNN)     │
│  • Filtra cajas degeneradas (w≤2 o h≤2 px)                             │
│  • (Opcional) aplica DetectionAugmentation                             │
│                    │                                                    │
│                    ▼                                                    │
│  3. DATALOADER                                                          │
│  batch_size=1 (requerido por bug en torchvision_05.filter_proposals)   │
│  collate_fn personalizado para listas de imágenes de tamaño variable   │
│                    │                                                    │
│                    ▼                                                    │
│  4. MODELO — Faster R-CNN                                               │
│  ┌────────────────────────────────────────────────────────────┐        │
│  │  Backbone: ResNet-50 (pretrained COCO)                     │        │
│  │       ↓ Feature maps a 4 escalas                          │        │
│  │  FPN (Feature Pyramid Network)                            │        │
│  │       ↓ P2, P3, P4, P5 (+ P6 via max-pool)               │        │
│  │  RPN (Region Proposal Network)                            │        │
│  │    • AnchorGenerator: sizes=(8,16,32,64,128)              │        │
│  │      aspect_ratios=(0.5,1.0,2.0)×5 niveles               │        │
│  │    • Clasifica: objectness (fg/bg)                        │        │
│  │    • Regresa: deltas de bbox                              │        │
│  │    • NMS → top-K proposals                                │        │
│  │       ↓ RoIs seleccionadas                                │        │
│  │  RoI Pooling (7×7)                                        │        │
│  │       ↓                                                   │        │
│  │  FastRCNNPredictor (cabeza custom)                        │        │
│  │    • Clasifica: 5 clases (background + 4 daños)           │        │
│  │    • Regresa: bbox final por clase                        │        │
│  └────────────────────────────────────────────────────────────┘        │
│                    │                                                    │
│                    ▼                                                    │
│  5. ENTRENAMIENTO                                                       │
│  Loss = L_obj + L_rpn_reg + L_cls + L_box_reg  (pesos [1,1,1,1])      │
│  SGD: lr=0.001, momentum=0.9, weight_decay=0.0005                      │
│  StepLR: step_size=6, gamma=0.1  →  18 épocas                         │
│  Early stopping basado en F1 de validación                             │
│                    │                                                    │
│                    ▼                                                    │
│  6. EVALUACIÓN — test_detection_model()                                │
│  th_score=0.5 (confianza) + th_iou=0.5 (IoU con GT)                   │
│  → Precision, Recall, F1 por clase                                     │
│  → Confusion matrix                                                    │
│  → Análisis de anchors seleccionados                                   │
│                    │                                                    │
│                    ▼                                                    │
│  7. EXPERIMENTOS AVANZADOS (sección 4.7)                               │
│  a) K-means anchors  b) Data Augmentation  c) Mask-RCNN               │
│  d) Ablación de pérdidas  e) Evaluación por tipo de desastre           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ 2. DESGLOSE DE TÉCNICAS Y PASOS APLICADOS

### Fase 1-3: PASCAL VOC (familiarización con Faster R-CNN)

---

#### 1. Carga de datos — `myVOCDataset`
**Qué hace:** Lee imágenes + segmentaciones XML del dataset PASCAL VOC 2012, extrae bounding boxes por clase y construye el formato `{boxes, labels, image_id, area, iscrowd}` que espera Faster R-CNN.

**Por qué:** El formato estándar de torchvision espera exactamente este diccionario de targets. Usar la clase personalizada permite reutilizar la infraestructura oficial de `train_one_epoch` y `eval_one_epoch` del `engine.py`.

---

#### 2. Subconjunto de 4 clases (bottle, chair, dining_table, sofa)
**Qué hace:** Filtra las 20 clases originales de VOC a solo 4 clases de interior + background.

**Por qué:** Reducir el espacio del problema para poder entrenar con recursos limitados de GPU, manteniendo la dificultad conceptual (objetos de distinto tamaño y aspect ratio).

---

#### 3. Transfer Learning — ResNet-50 preentrenado en COCO
**Qué hace:** Carga `fasterrcnn_resnet50_fpn(pretrained=True)` y reemplaza únicamente la cabeza `FastRCNNPredictor` con el número correcto de clases de salida.

**Por qué:** El backbone ya ha aprendido representaciones visuales ricas (bordes, texturas, formas) en millones de imágenes de COCO. Fine-tuning solo de la cabeza es más rápido y evita el sobreajuste con datasets pequeños (234 imágenes de train en VOC).

---

#### 4. Custom `AnchorGenerator` con anclas pequeñas para xBD
**Qué hace:** Define anclas de `sizes=(8,16,32,64,128)` con `aspect_ratios=(0.5,1.0,2.0)` para 5 niveles del FPN.

**Por qué:** El análisis estadístico de los edificios en xBD revela tamaños muy pequeños en píxeles de la imagen redimensionada (p5≈9px, mediana≈26px, p95≈51px). Los anchors por defecto de COCO son mucho más grandes y no cubren estos objetos. Los aspect ratios [0.5,1.0,2.0] cubren el rango observado [0.55,1.82].

---

#### 5. Función de evaluación `test_detection_model()`
**Qué hace:** Itera el test set, aplica NMS implícita del modelo, filtra por `th_score` (confianza) y calcula IoU entre predicciones y GT con `bb_intersection_over_union()`. Reporta Precision, Recall, F1 por clase, confusion matrix y distribución de anchors seleccionados.

**Por qué:** Las métricas estándar de clasificación (accuracy) no son útiles en detección porque el número de "negativos" (regiones sin objeto) es enorme. Se necesitan métricas que cuenten detecciones correctas relativas a GT y predicciones.

---

### Fase 4: xBD Disaster Assessment

---

#### 6. Dataset `xBDDataset` + wrapper `xBDDetectionDataset`
**Qué hace:**
- Lee imágenes TIFF post-desastre de 1024×1024 con GSD<0.8m
- Parsea polígonos de edificios en formato WKT (Shapely) → extrae bounding boxes mínimos
- Asigna etiquetas de daño: 0=background, 1=no-damage, 2=minor, 3=major, 4=destroyed
- Desplaza etiquetas en +1 para el wrapper (0 queda reservado para background en Faster R-CNN)
- Filtra cajas degeneradas (ancho o alto ≤ 2 px)
- Crea splits train/val/test por nombre de directorio de desastre

**Por qué:** La anotación de xBD usa polígonos arbitrarios y etiquetas de daño en JSON separados (pre/post imagen). El dataset class centraliza toda la lógica de parseo y permite que el modelo reciba datos en el formato estándar de torchvision.

---

#### 7. `torchvision_05` (versión modificada)
**Qué hace:** Versión local de torchvision adaptada para:
- Permitir imágenes sin objetos (listas de targets vacías) durante entrenamiento
- Bug conocido: `filter_proposals` hace `anchors.squeeze(0)[keep,:]` que requiere batch=1

**Por qué:** La versión oficial de torchvision falla cuando un batch contiene imágenes sin anotaciones (lo que ocurre en xBD con parches de imágenes limpias). La modificación permite este caso.

---

#### 8. SGD + StepLR (optimizador y scheduler)
**Qué hace:**
```
optimizer = SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
```
El LR cae ×10 en las épocas 6 y 12, haciendo 3 "fases" de aprendizaje en 18 épocas.

**Por qué:**
- **SGD vs Adam:** SGD con momentum es el estándar en detección (usado por el paper original de Faster R-CNN). Generaliza mejor que Adam en tareas de visión con transfer learning.
- **weight_decay=0.0005:** Regularización L2 para evitar sobreajuste en un dataset moderado.
- **StepLR:** El decaimiento escalonado permite al modelo converger rápido al inicio y luego ajustar fino sin oscilar.

---

#### 9. Early stopping basado en F1 de validación
**Qué hace:** Guarda el modelo con mejor F1 score en validación (`best_model.pth`) durante las 18 épocas.

**Por qué:** F1 es más informativa que la loss total para evaluar si el modelo detecta correctamente. La loss podría seguir bajando mientras el modelo empeora en métricas de detección (overfitting a las pérdidas de regresión).

---

#### 10. Pesos de pérdida `weights=[1,1,1,1]`
**Qué hace:** La loss total es:
```
L = w1·L_objectness + w2·L_rpn_box_reg + w3·L_classifier + w4·L_box_reg
```

**Por qué:** Empezar con pesos iguales es la línea base correcta; permite luego el experimento de ablación (sección 4.7d) donde se varía cada componente para analizar su impacto.

---

#### 11. batch_size=1
**Qué hace:** Los DataLoaders se configuran con batch_size=1 en todo el proyecto.

**Por qué:** Bug en `torchvision_05.filter_proposals`: realiza `anchors.squeeze(0)[keep,:]` que solo funciona cuando la dimensión 0 tiene tamaño 1. Con batch>1, los índices en `keep` quedan fuera de rango → error CUDA out-of-bounds.

---

### 4.7a: K-means IoU Anchors

---

#### 12. K-means con distancia IoU (estilo YOLOv2)
**Qué hace:**
1. Recopila todos los (w,h) de cajas GT del conjunto de train
2. Aplica k-means con distancia `1 - IoU(box, centroid)` (boxes centradas en el origen)
3. Barre k=1..11 y traza curva "elbow" de avg-IoU vs k
4. Selecciona k=5 (un tamaño por nivel FPN)
5. Deriva aspect_ratios de los centroides via log-space

**Por qué:** Los anchors por defecto de COCO tienen tamaños diseñados para objetos de tamaño medio-grande. Los edificios en imágenes de satélite son mucho más pequeños y tienen distribución diferente. K-means con IoU (no distancia euclidiana) optimiza directamente la métrica de evaluación.

---

### 4.7b: Data Augmentation para Detección

---

#### 13. `DetectionAugmentation` — aumentos geométricos y fotométricos

| Transformación | Parámetro | Implementación en boxes |
|----------------|-----------|------------------------|
| Volteo horizontal | p=0.5 | `x1 ← W-x2, x2 ← W-x1` |
| Volteo vertical | p=0.5 | `y1 ← H-y2, y2 ← H-y1` |
| Rotación aleatoria | ±15°, p=0.5 | Rota 4 esquinas, toma bbox axis-aligned mínima |
| Recorte aleatorio | min_scale=0.75, p=0.5 | Filtra por centroide, reescala coords |
| Brillo/Contraste | p=0.5 | `TF.adjust_brightness / adjust_contrast` |

**Por qué:** Las imágenes de satélite no tienen orientación preferida (no hay "arriba" canónico). El volteo y la rotación multiplican la variedad del dataset sin coste de anotación. El recorte simula variación de escala. El ajuste fotométrico compensa las diferencias de iluminación entre imágenes tomadas en distintas condiciones atmosféricas.

---

### 4.7c: Mask-RCNN para Segmentación de Instancias

---

#### 14. Extensión a Mask-RCNN
**Qué hace:**
- Carga `maskrcnn_resnet50_fpn(pretrained=True)`
- Reemplaza `FastRCNNPredictor` (5 clases)
- Reemplaza `MaskRCNNPredictor(in_features_mask, hidden=256, num_classes=5)`
- Añade dataset `xBDInstanceDataset` que genera máscaras binarias por instancia (rasterizando polígonos WKT con OpenCV)
- 5ª pérdida: `loss_mask` (BCE sobre la máscara predicha para la clase GT)

**Por qué:** Mask-RCNN extiende Faster R-CNN añadiendo una rama de segmentación paralela a la de clasificación/regresión. Permite no solo detectar el edificio sino delinear su contorno exacto, útil para estimar el área dañada con mayor precisión. Usa RoI Align (en vez de RoI Pool) para evitar el desalineamiento cuantificado.

---

### 4.7d: Ablación de Pérdidas

---

#### 15. Experimentos de ablación con `weights`

| Experimento | Pesos `[obj, rpn_reg, cls, box_reg]` | Hipótesis |
|-------------|--------------------------------------|-----------|
| `w_base` | `[1, 1, 1, 1]` | Referencia |
| `w_no_obj` | `[0, 1, 1, 1]` | RPN ciega → colapso de detección |
| `w_no_cls` | `[1, 1, 0, 1]` | Sin clasificación → RoI head no distingue clases |
| `w_cls_x3` | `[1, 1, 3, 1]` | Énfasis en clasificación de daño |

**Metodología:** Fine-tuning de 3 épocas partiendo del mejor modelo base (lr×0.1 = 0.0001).

**Por qué:** 3 épocas es suficiente para observar tendencias divergentes sin necesidad de convergencia completa. Partir del modelo pre-entrenado elimina el ruido del calentamiento inicial.

---

### 4.7e: Evaluación Estratificada por Desastre

---

#### 16. Análisis por tipo de desastre
**Qué hace:** Extrae el tipo de desastre del path de la imagen, acumula estadísticas de detección por desastre y genera:
- Barras de P/R/F1 por desastre (5 tipos)
- Heatmap F1: desastre × clase-de-daño

**Desastres:** hurricane-florence, hurricane-matthew, mexico-earthquake, palu-tsunami, socal-fire

**Por qué:** El rendimiento global puede enmascarar que el modelo funciona bien para huracanes (más datos) pero falla en tsunamis (menos datos o patrones distintos de daño). Esta análisis permite identificar sesgos de dominio y puntos débiles específicos.

---

## 🧠 3. CONCEPTOS FUNDAMENTALES DE DETECCIÓN DE OBJETOS

### 3.1 Bounding Boxes y sistema de coordenadas

Una **bounding box** es el rectángulo mínimo que encierra un objeto, representado como `[x1, y1, x2, y2]` (coordenadas de esquina superior-izquierda e inferior-derecha) o alternativamente como `[cx, cy, w, h]` (centro + dimensiones).

En el pipeline: las anotaciones WKT de xBD contienen polígonos → se extrae el bounding box mínimo con `polygon.bounds` de Shapely.

---

### 3.2 Intersection over Union (IoU)

```
IoU = Área(A ∩ B) / Área(A ∪ B)
```

- **IoU = 0**: sin solapamiento
- **IoU = 1**: coincidencia perfecta
- **Umbral th_iou=0.5**: una detección se considera correcta si su IoU con la GT más cercana ≥ 0.5 (umbral PASCAL VOC estándar)

En el código: `bb_intersection_over_union()` de `external.py` computa el IoU entre dos cajas en formato `[x1,y1,x2,y2]`.

---

### 3.3 Anchor Boxes

Los **anchors** son cajas de referencia de tamaño y aspect ratio predefinidos, repartidas en una rejilla densa sobre los feature maps del FPN. La RPN predice:
1. Si hay un objeto en esa ancla (objectness score ∈ [0,1])
2. Los **deltas de regresión** `(tx, ty, tw, th)` para ajustar el ancla al objeto real

**En xBD:**
```python
AnchorGenerator(
    sizes=((8,), (16,), (32,), (64,), (128,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)
```
Esto genera 3 anchors por posición × 5 niveles FPN = **15 anchors** (cada nivel cubre un rango de tamaños distinto).

**Justificación de tamaños pequeños:** El modelo redimensiona las imágenes a min_size=800. La escala es 800/1024 ≈ 0.78. Los edificios con mediana≈26px en imagen original pasan a ≈20px → el nivel FPN más pequeño (P2, stride 4) produce feature maps donde 5 píxeles cubren 20px de imagen → anchors de 8-32px son los apropiados.

---

### 3.4 Region Proposal Network (RPN)

La RPN opera sobre cada nivel del FPN y produce:
- **Objectness scores**: probabilidad de que cada ancla contenga algún objeto (clase agnóstica)
- **Regresión de bbox**: ajuste fino a la caja real

**Pérdidas de la RPN:**
```
L_RPN = L_objectness + L_rpn_box_reg
```
- `L_objectness`: Binary Cross-Entropy entre score predicho y etiqueta {fg=1, bg=0}
- `L_rpn_box_reg`: Smooth L1 loss entre deltas predichos y deltas GT (solo para anclas positivas)

**NMS en la RPN:** Las proposals se filtran por NMS para eliminar propuestas redundantes antes de pasarlas al RoI Pooling.

---

### 3.5 Feature Pyramid Network (FPN)

El FPN combina:
- **Pathway bottom-up**: extrae feature maps a distintas escalas (ResNet stages C2-C5)
- **Pathway top-down**: propaga contexto semántico de alto nivel hacia abajo via upsampling
- **Conexiones laterales**: suma element-wise de ambos pathways

Resultado: 5 niveles de feature maps (P2-P6) con la misma dimensión de canales (256), cada uno especializado en objetos de diferente tamaño.

**Por qué es crítico en xBD:** Los edificios tienen tamaños muy variables (9px a 51px en imagen escalada). Sin FPN, un único feature map favorecería solo un rango de tamaños.

---

### 3.6 RoI Pooling y RoI Align

**RoI Pooling:** Divide la región de interés en una rejilla 7×7 y aplica max-pooling. Introduce un desalineamiento cuantificado porque redondea coordenadas a enteros.

**RoI Align (Mask-RCNN):** Usa interpolación bilineal para evitar el redondeado, preservando el alineamiento espacial. Crítico para la rama de segmentación de Mask-RCNN donde el pixel-alignment importa.

---

### 3.7 Non-Maximum Suppression (NMS)

Algoritmo para eliminar detecciones duplicadas:
1. Ordena por score descendente
2. Selecciona la detección con mayor score
3. Elimina todas las detecciones con IoU > umbral respecto a la seleccionada
4. Repite hasta que no queden detecciones

En el pipeline: la NMS es interna al modelo de torchvision. Los hiperparámetros de evaluación externos son `th_score` (umbral de confianza antes de reportar) y `th_iou` (umbral para marcar una detección como TP).

---

### 3.8 Precision, Recall y F1 en Detección

```
Precision = TP / (TP + FP)   ← de lo que predigo, ¿cuánto es correcto?
Recall    = TP / (TP + FN)   ← de lo que existe en GT, ¿cuánto encuentro?
F1        = 2 · P · R / (P + R)
```

**En el contexto de detección:**
- **TP**: predicción con score > th_score y IoU con GT > th_iou
- **FP**: predicción con score > th_score pero sin GT correspondiente (IoU < th_iou)
- **FN**: objeto GT sin ninguna predicción coincidente

**Importante:** `test_detection_model()` computa además métricas de *clasificación* sobre las detecciones TP: la caja fue encontrada, pero ¿se clasificó correctamente el nivel de daño?

---

### 3.9 mean Average Precision (mAP)

**Average Precision (AP)** para una clase: área bajo la curva Precision-Recall trazada variando el umbral de confianza.

**mAP** = media de AP sobre todas las clases.

En VOC/COCO: mAP@0.5 usa th_iou=0.5. COCO usa mAP@[0.5:0.95] promediando sobre múltiples umbrales IoU.

El código usa `average_precision_score` de sklearn con `th_score` y `th_iou` fijos (0.5/0.5) en lugar de barrer umbrales, lo que equivale a AP en un único punto de operación.

---

### 3.10 Función de pérdida total de Faster R-CNN

```
L_total = w1·L_objectness + w2·L_rpn_box_reg + w3·L_classifier + w4·L_box_reg
```

| Componente | Tipo | Descripción |
|------------|------|-------------|
| `L_objectness` | Binary CE | RPN: ¿hay objeto? |
| `L_rpn_box_reg` | Smooth L1 | RPN: ajuste de ancla a propuesta |
| `L_classifier` | Cross-Entropy | Cabeza: clasificación por clase |
| `L_box_reg` | Smooth L1 | Cabeza: ajuste final de bbox |

**Smooth L1 Loss:** Combina L1 (robusta a outliers) y L2 (suave en el origen). Para `|x| < 1`: usa `0.5x²`; para `|x| ≥ 1`: usa `|x| - 0.5`.

---

### 3.11 Transfer Learning y Fine-tuning

**Estrategia usada:** Fine-tuning completo (todos los parámetros se actualizan, incluido el backbone).

- El backbone ResNet-50 arranca de pesos COCO → representaciones visuales ya útiles
- La cabeza `FastRCNNPredictor` se inicializa aleatoriamente → aprende la clasificación específica
- LR bajo (0.001) + StepLR evita destruir las representaciones del backbone

---

### 3.12 Overfitting: señales y mitigación

| Señal | Descripción |
|-------|-------------|
| `train_loss << val_loss` | Sobreajuste clásico |
| `F1_val` no mejora pero `train_loss` baja | El modelo memoriza sin generalizar |

**Mitigaciones en el proyecto:**
- `weight_decay=0.0005`: regularización L2
- Data augmentation (4.7b)
- Early stopping: se guarda el modelo con mejor F1_val, no el de menor train_loss
- Transfer learning: el backbone pre-entrenado actúa como regularizador implícito

---

## 🎯 4. SIMULACIÓN DE DEFENSA (PREGUNTAS DEL PROFESOR)

---

### Pregunta 1
**"¿Por qué elegiste batch_size=1? ¿No limita eso el entrenamiento?"**

**Respuesta:**
Batch_size=1 no fue una elección de diseño libre, sino una restricción técnica concreta. La versión local `torchvision_05` tiene un bug en la función `filter_proposals`: aplica `anchors.squeeze(0)[keep,:]`, donde `squeeze(0)` solo elimina la dimensión 0 cuando tiene tamaño 1. Con batch>1, los índices en `keep` son absolutos pero la dimensión 0 no se elimina, provocando un acceso out-of-bounds en CUDA.

Dicho esto, en detección de objetos batch_size=1 es común y funciona bien porque: (a) cada imagen tiene un número variable de objetos, lo que dificulta el batching eficiente; (b) el gradient a través de múltiples imágenes se compensa con momentum (SGD con momentum=0.9 actúa como un promedio exponencial de gradientes recientes). La principal desventaja es que las estadísticas de batch normalization son ruidosas con batch=1, pero como el backbone ya tiene sus BatchNorm en modo evaluación (frozen) gracias al preentrenamiento, el impacto es mínimo.

---

### Pregunta 2
**"¿Por qué eligiste anchors de tamaños (8,16,32,64,128) y no los que vienen por defecto?"**

**Respuesta:**
Porque realicé un análisis estadístico previo de los edificios en el dataset xBD. Las medidas relevantes en píxeles de la imagen escalada (800px) son:
- p5 ≈ 9px, p25 ≈ 17px, mediana ≈ 26px, p75 ≈ 36px, p95 ≈ 51px
- Aspect ratios: rango [0.55, 1.82]

Los anchors por defecto de COCO son (32,64,128,256,512), diseñados para objetos de tamaño medio-grande en imágenes de escena natural. Aplicarlos directamente en xBD haría que la mayoría de edificios cayeran fuera del rango de las anclas del nivel P2 (el más fino), con lo que la RPN tendría muy pocas anclas positivas para aprender. Los tamaños (8,16,32,64,128) cubren exactamente el rango p5-p95 de los edificios. Los aspect ratios (0.5,1.0,2.0) cubren el rango observado.

---

### Pregunta 3
**"¿Qué pasaría si eliminamos la pérdida de objectness (w_no_obj = [0,1,1,1])?"**

**Respuesta:**
El experimento de ablación 4.7d estudia exactamente esto. Sin `L_objectness`, la RPN no aprende a distinguir regiones con objetos de regiones de fondo. La consecuencia es que el pipeline colapsa porque las proposals que la RPN envía al RoI Pooling son aleatorias (la RPN no sabe dónde buscar). Si la cabeza de clasificación no recibe buenos RoIs centrados en edificios, tampoco puede aprender a clasificar el daño correctamente. El F1 cae drásticamente. Este experimento demuestra que `L_objectness` es el componente más crítico del sistema: es el "cuello de botella" que permite que el resto del pipeline funcione.

---

### Pregunta 4
**"¿Cómo sabes que el modelo no está haciendo overfitting?"**

**Respuesta:**
Usamos varias estrategias de diagnóstico:

1. **Monitoreo de las curvas de pérdida train vs val**: registradas en `log.csv` por época. Si la train_loss baja pero la val_loss sube, hay overfitting.

2. **Early stopping basado en F1_val**: guardamos `best_model.pth` con el mejor F1 en validación, no el modelo de menor train_loss. Esto evita seleccionar un modelo sobreajustado.

3. **Métricas de test en conjunto separado**: la evaluación final se hace en `dataset_test`, un conjunto que el modelo nunca ha visto durante el entrenamiento ni el early stopping.

4. **weight_decay=0.0005**: regularización L2 que penaliza pesos grandes.

5. **Transfer learning**: el backbone pre-entrenado actúa como regularizador implícito porque ya tiene buenas representaciones.

6. **Data augmentation (4.7b)**: aumenta la variedad de entrenamiento efectiva, dificultando la memorización.

---

### Pregunta 5
**"¿Por qué usas SGD en lugar de Adam? ¿No converge Adam más rápido?"**

**Respuesta:**
Es correcto que Adam converge más rápido en las primeras épocas. Sin embargo, la literatura en visión por computadora ha establecido consistentemente que SGD con momentum generaliza mejor que Adam en tareas de detección y clasificación visual, especialmente cuando se hace transfer learning.

La razón es que Adam usa tasas de aprendizaje adaptativas por parámetro, lo que puede hacer que las capas pre-entrenadas del backbone se actualicen de forma desproporcionada, "olvidando" representaciones aprendidas. SGD con una LR global pequeña y decaimiento escalonado (StepLR: gamma=0.1 en épocas 6 y 12) permite un ajuste más controlado. Además, el paper original de Faster R-CNN (Ren et al., 2015) usa SGD, y seguir esta configuración facilita la reproducibilidad.

---

### Pregunta 6
**"¿Qué aporta el FPN sobre un backbone ResNet simple?"**

**Respuesta:**
ResNet sin FPN produce un único feature map a una escala específica (típicamente 1/32 del tamaño original). Esto significa que solo puede detectar eficientemente objetos de un rango de tamaños. Los edificios en xBD van desde 9px hasta 51px en imagen escalada, un rango de casi 6x.

El FPN construye una pirámide de feature maps (P2 a P6) donde cada nivel combina:
- La resolución espacial alta de los niveles inferiores (para objetos pequeños)
- El contexto semántico de los niveles superiores (para reconocimiento)

Esto se logra con el pathway top-down + conexiones laterales. El resultado es que P2 (stride 4) captura edificios pequeños de ~8px y P6 (stride 64) captura los más grandes. Sin FPN, los edificios pequeños en satélite serían virtualmente invisibles para la RPN.

---

### Pregunta 7
**"¿Por qué el Data Augmentation mejora el rendimiento? ¿Qué transformaciones aplicaste y cómo adaptas las bounding boxes?"**

**Respuesta:**
El data augmentation mejora el rendimiento porque expone al modelo a variaciones de la imagen que son realistas pero no están en el conjunto original, reduciendo la brecha de generalización.

Para imágenes de satélite en particular:
- **Volteo horizontal/vertical (p=0.5):** Las imágenes satelitales no tienen "arriba" canónico. Un edificio dañado visto desde el norte o desde el sur es el mismo edificio.
- **Rotación ±15° (p=0.5):** Simula variaciones en la orientación del satélite o del edificio.
- **Recorte aleatorio (min_scale=0.75, p=0.5):** Simula variaciones de escala/zoom.
- **Brillo/Contraste:** Simula diferencias de iluminación y condiciones atmosféricas.

La clave es transformar **tanto la imagen como las bounding boxes** de forma consistente. Para cada transformación geométrica:
- **Volteo H:** `x1 ← W - x2; x2 ← W - x1`
- **Volteo V:** `y1 ← H - y2; y2 ← H - y1`
- **Rotación:** Se rotan las 4 esquinas del rectángulo y se calcula el bounding box mínimo axis-aligned que las contiene
- **Recorte:** Se filtran las cajas cuyo centroide queda fuera del recorte y se reescalan las coordenadas de las que permanecen

---

### Pregunta 8
**"¿Qué diferencia hay entre Faster R-CNN y Mask R-CNN? ¿Qué ventaja aporta Mask R-CNN en este problema?"**

**Respuesta:**
Faster R-CNN y Mask R-CNN comparten backbone, FPN y RPN. La diferencia es que Mask R-CNN añade una **tercera rama paralela** en la cabeza RoI que predice, para cada instancia detectada, una máscara binaria de segmentación de tamaño 28×28 (posteriormente resized a la bbox).

Arquitectónicamente: en lugar de RoI Pooling usa **RoI Align** para preservar el alineamiento espacial preciso (crítico para la segmentación pixel-accurate). La cabeza de máscara (`MaskRCNNPredictor`) usa convoluciones + transpuesta para upsampling, con 256 canales intermedios en nuestra implementación.

Ventaja en xBD: los edificios dañados no tienen formas rectangulares. Un bounding box puede incluir parte de un edificio vecino no dañado, confundiendo la clasificación. La máscara de instancia permite aislar exactamente los píxeles del edificio en cuestión, potencialmente mejorando la clasificación del nivel de daño. Además, la máscara facilita estimar el porcentaje de área dañada, no solo la clase.

Coste: una 5ª pérdida (`loss_mask`: BCE sobre máscara predicha para la clase GT), mayor memoria GPU y dataset con anotaciones de polígono (que ya tenemos en xBD vía WKT).

---

### Pregunta 9
**"¿Qué conclusiones extraes del análisis estratificado por tipo de desastre (4.7e)?"**

**Respuesta:**
La evaluación estratificada por desastre (hurricane-florence, hurricane-matthew, mexico-earthquake, palu-tsunami, socal-fire) revela sesgos de dominio importantes que el F1 global oculta:

1. **Desequilibrio de datos:** Los desastres con más imágenes de entrenamiento (hurricanes) típicamente tienen mejor rendimiento. El modelo aprende mejor los patrones de daño que ha visto más.

2. **Patrones de daño específicos por desastre:** Un incendio forestal (socal-fire) produce daños con texturas radicalmente distintas a un tsunami (palu-tsunami): áreas quemadas vs estructuras colapsadas inundadas. El modelo puede tener sesgo hacia el tipo de daño más frecuente en train.

3. **El heatmap F1: desastre × clase-daño** permite identificar combinaciones críticas donde el modelo falla (ej. "major-damage en tsunami"), guiando qué datos adicionales recopilar o qué técnicas de domain adaptation aplicar.

4. **Implicación práctica:** En un sistema de respuesta a emergencias reales, un modelo con buen F1 global pero pobre rendimiento en un tipo de desastre específico podría fallar exactamente cuando más se necesita.

---

### Pregunta 10
**"Si tuvieras más tiempo, ¿qué mejorarías del pipeline y por qué?"**

**Respuesta:**
Hay varias direcciones de mejora justificadas técnicamente:

1. **Desequilibrio de clases:** Las clases "no-damage" (1) son mucho más frecuentes que "destroyed" (4). Implementaría **Focal Loss** (Lin et al., 2017) o sobremuestreo de imágenes con edificios muy dañados para reducir el sesgo hacia la clase mayoritaria.

2. **Información pre-desastre:** El dataset xBD incluye imágenes pre-desastre. Un modelo que compara pre/post (Change Detection) usando una arquitectura siamesa o de diferencia de feature maps aprovecharía directamente la señal de cambio para clasificar el daño, en lugar de inferirlo solo del aspecto post.

3. **Resolución más fina:** Con batch_size=1 y el bug de torchvision_05 corregido, podría usar batch_size=4 y backbone más potente (ResNet-101 o Swin Transformer) que extrae features más ricas.

4. **Anchors adaptativos por nivel FPN:** El experimento k-means (4.7a) mejora el ancla global, pero podría refinar los tamaños independientemente por nivel FPN, usando los centroids k-means como punto de partida para cada nivel.

5. **Domain Adaptation:** Para el sesgo entre tipos de desastre, técnicas de domain adaptation (adversarial training o DANN) podrían alinear las distribuciones de features entre diferentes tipos de catástrofe.

---

## 📊 APÉNDICE: TABLA DE HIPERPARÁMETROS CLAVE

| Hiperparámetro | Valor | Justificación |
|----------------|-------|---------------|
| `batch_size` | 1 | Bug en torchvision_05.filter_proposals |
| `lr` | 0.001 | Estándar para fine-tuning de Faster R-CNN |
| `momentum` | 0.9 | SGD estándar para detección |
| `weight_decay` | 0.0005 | Regularización L2 moderada |
| `num_epochs` | 18 | 3 fases de LR (6+6+6 épocas) |
| `step_size` | 6 | num_epochs/3 → 3 caídas de LR |
| `gamma` | 0.1 | LR cae ×10 en cada step |
| `th_score` | 0.5 | Umbral de confianza para reportar detección |
| `th_iou` | 0.5 | Umbral IoU para considerar TP (estándar PASCAL) |
| `anchor sizes` | (8,16,32,64,128) | Calibrado a distribución de edificios xBD |
| `aspect_ratios` | (0.5,1.0,2.0) | Cubre rango [0.55, 1.82] observado |
| `min_size` | 800 | Redimensionado estándar de torchvision |
| `max_size` | 1333 | Evita imágenes demasiado grandes en memoria |
| `patch_size` | 1024 | Imagen completa (sin ventana deslizante) |
| `weights (losses)` | [1,1,1,1] | Contribución igual de los 4 componentes |
| `hidden_mask` | 256 | Canales en MaskRCNNPredictor |
| `lr ablación` | 0.0001 | lr×0.1 para fine-tuning de ablación |
| `num_epochs_abl` | 3 | Suficiente para ver tendencias divergentes |
| `max_angle` | 15° | Rotación realista para satélite |
| `min_scale_crop` | 0.75 | Conserva ≥75% del campo visual |
| `p_augment` | 0.5 | Probabilidad por transformación |

---

## 📐 APÉNDICE: FÓRMULAS RÁPIDAS

```
IoU(A,B) = |A ∩ B| / |A ∪ B|

L_smooth1(x) = { 0.5·x²        si |x| < 1
               { |x| - 0.5     si |x| ≥ 1

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2·P·R / (P + R)

AP = ∫₀¹ Precision(Recall) d(Recall)
mAP = (1/C) · Σᵢ APᵢ

L_total = w₁·L_obj + w₂·L_rpn_reg + w₃·L_cls + w₄·L_box_reg
```

---

*Documento generado para la defensa del Proyecto 2B — Object Detection · VA · UC3M · 2025-2026*
