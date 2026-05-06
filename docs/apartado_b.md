# Apartado b) — Data augmentation para detección de objetos

## Motivación

En detección de objetos, el modelo debe ser robusto ante variaciones en la orientación, escala e iluminación de los edificios. El dataset xBD contiene imágenes satelitales con una distribución geográfica limitada, lo que aumenta el riesgo de sobreajuste. El data augmentation amplía artificialmente la diversidad del conjunto de entrenamiento sin necesidad de más datos etiquetados.

La diferencia clave respecto al Proyecto 1 (clasificación) es que **cualquier transformación geométrica debe aplicarse de forma consistente a la imagen y a sus bounding boxes**. Si se voltea la imagen pero no las cajas, las anotaciones dejan de corresponder con los objetos y el entrenamiento diverge.

---

## Transformaciones implementadas

Se implementa la clase `DetectionAugmentation`, que aplica cada transformación de forma independiente con probabilidad `p` (por defecto 0.5).

### 1. Volteo horizontal y vertical

**Qué hace:** Refleja la imagen respecto al eje vertical u horizontal.

**Efecto sobre las bounding boxes:**
- Volteo horizontal: `x₁ ← W − x₂`, `x₂ ← W − x₁`
- Volteo vertical: `y₁ ← H − y₂`, `y₂ ← H − y₁`

**Justificación:** Las imágenes satelitales no tienen una orientación canónica: un edificio puede aparecer orientado en cualquier dirección. El volteo duplica el número efectivo de muestras con coste nulo y es la augmentación más habitual en visión por satélite.

---

### 2. Rotación aleatoria (±15°)

**Qué hace:** Rota la imagen un ángulo aleatorio dentro del rango `[−max_angle, +max_angle]` (por defecto 15°).

**Efecto sobre las bounding boxes:** Se rotan las cuatro esquinas de cada caja y se calcula la bounding box envolvente axis-aligned mínima sobre las esquinas rotadas. Esto introduce un pequeño exceso de área asumible, ya que Faster-RCNN también predice cajas axis-aligned.

**Justificación:** Los edificios en imágenes aéreas aparecen con distintas orientaciones según la toma. La rotación moderada (±15°) mejora la invarianza orientacional sin deformar excesivamente las cajas.

---

### 3. Recorte aleatorio (*random crop*)

**Qué hace:** Recorta una subregión aleatoria de la imagen (escala entre `min_scale` e imagen completa, por defecto `min_scale=0.75`) y la redimensiona al tamaño original.

**Efecto sobre las bounding boxes:** Se eliminan las cajas cuyo centroide queda fuera del recorte. Las cajas supervivientes se desplazan al sistema de coordenadas local del recorte y se reescalan proporcionalmente al nuevo tamaño.

**Justificación:** Simula diferentes altitudes de vuelo (zoom) y obliga al modelo a detectar edificios parcialmente visibles en los bordes del encuadre, situación frecuente en imágenes de grandes zonas urbanas.

---

### 4. Cambio de brillo y contraste (*color jitter*)

**Qué hace:** Multiplica aleatoriamente el brillo (factor en `[0.6, 1.4]`) y el contraste (factor en `[0.7, 1.3]`) de la imagen.

**Efecto sobre las bounding boxes:** Ninguno — es una transformación puramente fotométrica.

**Justificación:** Las imágenes satelitales varían en condiciones de iluminación según la hora, la estación y las condiciones atmosféricas. El jitter fotométrico evita que el modelo aprenda a depender del nivel absoluto de brillo.

---

## Integración en el pipeline

La clase `xBDDetectionDatasetAug` es un wrapper sobre `xBDDataset` que activa el augmentador cuando `data_augm=True`. El augmentador solo se aplica durante el entrenamiento; validación y test usan siempre la imagen original.

El dataset aumentado se conecta al mismo `DataLoader` y función `train_one_epoch` que el modelo base, sin modificar el resto del pipeline de entrenamiento.

---

## Experimento comparativo

Se entrena un modelo Faster R-CNN con la misma arquitectura y hiperparámetros que el modelo base del apartado 4.3, cambiando únicamente el dataset de entrenamiento:

| Configuración | Dataset train |
|---|---|
| Base | `xBDDetectionDataset` (sin augmentation) |
| Con augmentation | `xBDDetectionDatasetAug` (con augmentation) |

La comparación se realiza sobre el conjunto de test con los mismos umbrales (`TH_SCORE=0.5`, `TH_IOU=0.5`), reportando Precision, Recall y F1 globales y por clase de daño.

**Resultado esperado:** el modelo con augmentation debería mostrar mejor Recall (menos falsos negativos) a costa de una ligera caída en Precision, resultando en un F1 igual o superior, especialmente en las clases minoritarias (`major-damage`, `destroyed`).
