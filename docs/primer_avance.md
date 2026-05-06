# Primer Avance: Detección de Edificios en xBD con Faster R-CNN

**Fecha:** 2026-05-06  
**Notebook generado:** `APAI_Pr2B_xBD_Detection.ipynb`

---

## 1. Contexto y punto de partida

Este proyecto es la continuación del Proyecto 1, donde se clasificaba el nivel de daño de edificios en imágenes de satélite del dataset **xBD** usando patches de 64×64 píxeles y una ResNet18. El Proyecto 1 ya conocía la localización de cada edificio (extraída de las anotaciones JSON) y simplemente clasificaba si estaba dañado o no.

El Proyecto 2B cambia el problema: hay que **detectar** los edificios directamente sobre la imagen completa de 1024×1024 píxeles, sin saber a priori dónde están. Esto es un problema clásico de **detección de objetos**: localización + clasificación simultáneas.

La metodología base es la del Notebook del profesor (`APAI_Pr2B_ObjectDetection_2025_2026.ipynb`), que entrena Faster R-CNN con ResNet-50 FPN sobre PASCAL VOC. El trabajo de este avance ha sido:

1. Entender ambos notebooks (Proyecto 1 y Notebook 2B del profesor).
2. Analizar el dataset xBD en detalle (estructura de archivos, formato de anotaciones, tamaños reales de edificios).
3. Adaptar la metodología Faster R-CNN al dominio de las imágenes de satélite.
4. Producir un notebook ejecutable completo.

---

## 2. Análisis del dataset xBD

### 2.1 Estructura de archivos

```
data/xBD_UC3M/
├── train/   (7 desastres, 256 imágenes post-disaster)
│   ├── joplin-tornado/
│   │   ├── images/  → *_post_disaster.tif  (1024×1024 px, RGB o multiespectral)
│   │   └── labels/  → *_post_disaster.json
│   ├── moore-tornado/
│   └── ...
├── val/     (4 desastres, 45 imágenes)
└── test/    (5 desastres, 63 imágenes)
```

Cada imagen tiene su JSON de anotaciones correspondiente. Las anotaciones están disponibles tanto para imágenes pre-disaster como post-disaster; para detección usamos las **post-disaster**, que contienen el tipo de daño de cada edificio.

### 2.2 Formato de las anotaciones JSON

Inspeccionando un JSON real (`joplin-tornado_00000000_post_disaster.json`):

```python
data['features']['xy']  # lista de objetos anotados
# Cada objeto:
{
    'wkt': 'POLYGON ((208.98 22.09, 209.95 3.57, ...))',  # coordenadas en píxeles
    'properties': {
        'feature_type': 'building',
        'subtype': 'destroyed',      # o 'no-damage', 'minor-damage', 'major-damage'
        'uid': 'b6aa615e-...'
    }
}
```

Puntos clave:
- Los polígonos están en formato **WKT** (*Well-Known Text*), parseable con `shapely.wkt.loads()`.
- Las coordenadas están en **espacio píxel** (campo `xy`), no en coordenadas geográficas (campo `lng_lat`).
- Solo hay objetos con `feature_type == 'building'`; no hay otros tipos relevantes en este subset.
- Un JSON típico tiene ~185 edificios por imagen (muy denso).

### 2.3 Análisis de tamaños de los edificios

Se analizaron los **64.078 edificios** del conjunto de train. Para cada polígono WKT se calculó la bounding box `[x_min, y_min, x_max, y_max]` y se extrajeron anchura, altura y el lado del cuadrado de área equivalente (√área):

| Estadístico | Ancho (px) | Alto (px) | √Área (px) |
|-------------|-----------|-----------|------------|
| p5          | 9.0       | 9.0       | 9.4        |
| p25         | 17.0      | 17.4      | 17.5       |
| **mediana** | **26.0**  | **25.7**  | **26.3**   |
| p75         | 36.3      | 36.1      | 35.5       |
| p95         | 51.1      | 50.9      | 47.9       |
| máximo      | 435.7     | 547.9     | —          |

El **aspecto (w/h)** tiene mediana 1.00 y rango [0.55, 1.82], confirmando que la mayoría de edificios son aproximadamente cuadrados.

Esta distribución es radicalmente distinta a la de PASCAL VOC o COCO, donde los objetos suelen ocupar entre el 10% y el 50% de la imagen. Aquí los edificios son pequeños (mediana 26 px sobre una imagen de 1024 px, es decir, ≈2.5% del lado).

---

## 3. Decisiones de diseño y justificaciones

### 3.1 Usar la imagen completa (1024×1024) sin cropping previo

**Decisión:** `xBDDetectionDataset` devuelve la imagen completa. Faster R-CNN, a través de su módulo `GeneralizedRCNNTransform`, redimensiona automáticamente la imagen para que su lado más corto sea `min_size=800` y el más largo no supere `max_size=1333`.

**Justificación:** Con 1024×1024 la escala resultante es 800/1024 ≈ 0.78, lo que significa que los edificios de tamaño mediano (26 px) quedan en ~20 px tras el resize. Esta resolución es suficiente para que el RPN los detecte si los anchors son los adecuados. Hacer cropping previo en patches más pequeños obligaría a gestionar los edificios en el borde de los crops (truncados) y complicaría el dataloader innecesariamente.

### 3.2 Anchors personalizados: `(8, 16, 32, 64, 128)`

**Decisión:** Sustituir los anchors por defecto de Faster R-CNN `(32, 64, 128, 256, 512)` por `(8, 16, 32, 64, 128)` con aspect ratios `(0.5, 1.0, 2.0)`.

**Justificación:** Los anchor sizes por defecto están diseñados para COCO, donde los objetos son mucho más grandes. En xBD:

- El anchor de **8 px** cubre los edificios muy pequeños (p5 ≈ 9 px).
- El anchor de **16 px** cubre el rango p5–p25.
- El anchor de **32 px** cubre la mediana (26 px).
- El anchor de **64 px** cubre el rango p75–p95.
- El anchor de **128 px** captura los edificios grandes y outliers.

Los aspect ratios `(0.5, 1.0, 2.0)` cubren el rango observado [0.55, 1.82] de la distribución real. Al ser la mediana exactamente 1.0, el ratio cuadrado es el más importante.

Hay un punto crítico: Faster R-CNN redimensiona la imagen antes del RPN. Con la escala 0.78, un anchor de 8 px en el espacio original equivale a 8/0.78 ≈ 10 px en la imagen redimensionada. El `AnchorGenerator` trabaja sobre los feature maps del FPN, por lo que los tamaños se especifican en píxeles de la imagen de entrada al backbone (post-resize). Para que el anchor de nivel P2 del FPN cubra edificios de ~10 px (post-resize), el tamaño 8 es el mínimo razonable.

### 3.3 Solo imágenes post-disaster para detección

**Decisión:** El dataset usa únicamente las imágenes `*_post_disaster.tif` y sus JSON correspondientes.

**Justificación:** Las anotaciones post-disaster tienen el campo `subtype` con el nivel de daño, y contienen todos los edificios del área. Las pre-disaster tienen los mismos edificios pero sin etiqueta de daño (siempre `no-damage`). Para detección binaria (building vs fondo), cualquiera serviría, pero las post-disaster son el objetivo real del sistema y están anotadas de forma más completa.

### 3.4 Etiqueta única: `label=1` para todos los edificios

**Decisión:** Se asigna la misma etiqueta (1 = building) a todos los edificios, ignorando el subtipo de daño.

**Justificación:** La tarea de detección se desacopla de la clasificación del daño. Un sistema de dos fases (primero detectar, luego clasificar el daño de cada detección) es más robusto que intentar detectar y clasificar simultáneamente con una sola clase por nivel de daño. Esto también simplifica el entrenamiento inicial y permite validar que el RPN funciona correctamente antes de añadir complejidad.

### 3.5 Normalización de la imagen compatible con uint8 y uint16

**Decisión:** `image = image / np.iinfo(image.dtype).max` en lugar de `/ 255.0`.

**Justificación:** Las imágenes TIF de xBD pueden estar en formato uint8 (rango 0–255) o uint16 (rango 0–65535, dependiendo del sensor). Usando `np.iinfo(dtype).max` la normalización es siempre correcta independientemente del tipo de dato, sin necesidad de inspeccionarlo manualmente para cada desastre.

### 3.6 Filtrado de bounding boxes degeneradas

**Decisión:** Se descartan polígonos cuya bounding box tenga ancho o alto < 2 px.

**Justificación:** Faster R-CNN requiere que todas las cajas sean válidas (área > 0). Polígonos degenerados (puntos, líneas, o polígonos casi colapsados) producen errores en el cálculo de pérdidas o en las operaciones de NMS. El umbral de 2 px es conservador: excluye los casos imposibles de detectar y elimina potenciales NaN en el cálculo de IoU.

### 3.7 `torchvision_05` en lugar de torchvision estándar

**Decisión:** Usar la versión modificada del profesor ubicada en `/home/a472259/uni25-26/VA/VA_Pr2B_ObjectDetection_2025_2026/torchvision_05`.

**Justificación:** La versión estándar de torchvision no permite entrenar con imágenes que no tienen ningún edificio (targets vacíos), lo que rompería el dataloader porque xBD sí tiene imágenes sin edificios. `torchvision_05` también expone los anchors del RPN en la salida de inferencia (`pred[0]['anchors']`), lo que permite el análisis de anchors de `test_detection_model`.

---

## 4. Estructura del notebook generado

El notebook `APAI_Pr2B_xBD_Detection.ipynb` tiene 8 secciones:

| Sección | Contenido |
|---------|-----------|
| 0. Setup | Imports, rutas a `torchvision_05` y `engine.py`, detección de GPU |
| 1. Dataset | Clase `xBDDetectionDataset` + `collate_fn` + verificación de un ejemplo |
| 2. Anchor Analysis | Análisis de la distribución real de tamaños, histograma, justificación |
| 3. Modelo | `get_model()` con `AnchorGenerator` personalizado y `FastRCNNPredictor` de 2 clases |
| 4. DataLoaders | train (batch=2, shuffle), val/test (batch=1) |
| 5. Entrenamiento | Loop SGD + StepLR, checkpoints por época, `best_model.pth` por F1 |
| 6. Evaluación | `test_detection_model()` con IoU matching, métricas P/R/F1, visualización opcional |
| 7. Curvas | Plot de pérdidas y F1 desde el CSV de log |
| 8. Visualización | Predicciones sobre imágenes de test con GT (verde) y pred (rojo) |

---

## 5. Hiperparámetros del pipeline de entrenamiento

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `BATCH_SIZE` | 2 | Imágenes de 1024×1024 + backbone ResNet-50 consumen ~3–4 GB por imagen en GPU; 2 es el máximo razonable en GPU de 8–12 GB |
| `LR` | 0.001 | Valor típico para fine-tuning de Faster R-CNN; menor que el COCO original (0.02) porque el dataset es pequeño (256 imágenes) |
| `NUM_EPOCHS` | 20 | Con 256 imágenes y batch=2 hay 128 iteraciones/época; 20 épocas dan ~2560 steps de optimización, suficiente para convergencia |
| `STEP_SIZE` | 8 | Reduce LR en la época 8 (×0.1) y en la 16 (×0.01); cubre los dos tercios finales del entrenamiento con LR decreciente |
| `TH_SCORE` | 0.5 | Umbral de confianza estándar para filtrar predicciones de baja calidad |
| `TH_IOU` | 0.5 | Estándar PASCAL VOC para considerar una detección como verdadero positivo |

---

## 6. Puntos de atención para la ejecución

1. **Orden de celdas**: La función `test_detection_model` está definida en la sección 6, pero el loop de entrenamiento (sección 5) la llama para evaluar en validación al final de cada época. Hay que ejecutar la celda de definición de la función **antes** de lanzar el entrenamiento.

2. **Memoria GPU**: Si hay errores OOM con `BATCH_SIZE=2`, reducir a 1. Alternativamente, reducir `min_size` de 800 a 600 en `get_model()` para que el resize sea más agresivo.

3. **`engine.py` y `external.py`**: Son módulos del Notebook 2B del profesor. El notebook añade su directorio a `sys.path` en la primera celda. Si se mueve el notebook a otro directorio, hay que actualizar `PR2B_DIR`.

4. **Kernel de Jupyter**: Usar el kernel del entorno conda `VA_practica`, que tiene torch 2.11, torchvision 0.26 y shapely 2.1 instalados.

---

## 7. Próximos pasos sugeridos

- **Ejecución y ajuste de hiperparámetros**: Lanzar el entrenamiento y observar la curva de pérdida. Si el modelo no converge, probar `LR=0.005` o un warmup inicial.
- **Data augmentation**: Añadir flips horizontales/verticales (los edificios de satélite no tienen orientación preferente) y variaciones de brillo para mejorar la generalización entre desastres.
- **Clasificación multietiqueta**: Una vez validado el pipeline de detección, extender a 5 clases (fondo + 4 niveles de daño) para combinar detección y clasificación del daño en un solo paso.
- **NMS ajustado**: Los edificios están muy juntos (densidad alta). Evaluar si reducir el umbral de NMS (`nms_thresh` en el RPN) mejora la detección de edificios adyacentes.
