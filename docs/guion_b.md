# Guion de presentación — Apartado b) Data Augmentation para detección

---

## Diapositiva 1 — Título

**Título:** Data Augmentation para Detección de Objetos

**Subtítulo:** ¿Por qué no basta con las mismas técnicas que en clasificación?

> Frase de apertura (oral): "En clasificación solo transformamos la imagen. En detección, si movemos la imagen pero no las cajas, el modelo aprende basura."

---

## Diapositiva 2 — El problema clave (1 slide)

**Título:** La diferencia respecto a clasificación

**Contenido visual sugerido:** dos columnas

| Clasificación | Detección |
|---|---|
| Transformar imagen ✓ | Transformar imagen ✓ |
| — | Transformar bounding boxes ✓ |
| — | Eliminar cajas que salen del encuadre ✓ |

**Punto clave a remarcar (oral):** "Si volteas la imagen pero no las cajas, las anotaciones dejan de corresponder con los objetos y el entrenamiento diverge."

---

## Diapositiva 3 — Las 4 transformaciones (vista general)

**Título:** `DetectionAugmentation` — 4 transformaciones, probabilidad p=0.5

**Lista visual (iconos o numeración):**

1. **Volteo horizontal / vertical** — refleja imagen y cajas
2. **Rotación aleatoria ±15°** — rota imagen, recalcula bounding box envolvente
3. **Random Crop** — recorta subregión, elimina cajas fuera, reescala las que quedan
4. **Color Jitter** — brillo y contraste aleatorio (solo fotométrico, no afecta cajas)

> Oral: "Cada transformación se aplica de forma independiente con probabilidad 0.5, lo que genera combinaciones diversas."

---

## Diapositiva 4 — Detalle: Volteo

**Título:** Volteo horizontal y vertical

**Fórmulas (en caja destacada):**
```
Volteo horizontal:  x₁ ← W − x₂  |  x₂ ← W − x₁
Volteo vertical:    y₁ ← H − y₂  |  y₂ ← H − y₁
```

**Justificación (bullet):**
- Imágenes satelitales no tienen orientación canónica
- Duplica muestras efectivas con coste computacional cero
- Técnica estándar en visión por satélite

---

## Diapositiva 5 — Detalle: Rotación

**Título:** Rotación aleatoria ±15°

**Diagrama sugerido:** esquema de una caja antes/después de rotar → bounding box envolvente axis-aligned

**Puntos clave:**
- Se rotan las **4 esquinas** de cada caja
- Se calcula la **bounding box axis-aligned mínima** sobre las esquinas rotadas
- Introduce un pequeño exceso de área — asumible porque Faster R-CNN también predice cajas axis-aligned
- Mejora invarianza orientacional sin deformar excesivamente

---

## Diapositiva 6 — Detalle: Random Crop

**Título:** Recorte aleatorio (Random Crop)

**Puntos clave:**
- Escala mínima del recorte: 75% de la imagen original
- La subregión se redimensiona al tamaño original tras el recorte

**Efecto sobre las cajas (dos reglas):**
1. Cajas cuyo **centroide queda fuera** → eliminadas
2. Cajas supervivientes → coordenadas desplazadas al sistema local + reescaladas

**Justificación:** Simula distintas altitudes de vuelo y obliga al modelo a detectar edificios parcialmente visibles en los bordes.

---

## Diapositiva 7 — Detalle: Color Jitter

**Título:** Cambio de brillo y contraste

**Rangos:**
- Brillo: factor ∈ [0.6, 1.4]
- Contraste: factor ∈ [0.7, 1.3]

**Punto clave:** transformación **puramente fotométrica** → no modifica las bounding boxes

**Justificación:** Las imágenes satelitales varían con la hora, estación y condiciones atmosféricas. Evita que el modelo dependa del nivel absoluto de brillo.

---

## Diapositiva 8 — Integración en el pipeline

**Título:** Cómo se conecta al resto del sistema

**Diagrama de flujo sugerido:**
```
xBDDataset (base)
      ↓
xBDDetectionDatasetAug (wrapper)
      ↓  data_augm=True (solo train)
DetectionAugmentation
      ↓
DataLoader → train_one_epoch → Faster R-CNN
```

**Puntos clave:**
- Solo se aplica en entrenamiento; validación y test usan imagen original
- No modifica el DataLoader ni `train_one_epoch` — encapsulado completo en el wrapper

---

## Diapositiva 9 — Experimento comparativo

**Título:** ¿Mejora el augmentation?

**Tabla de configuraciones:**

| Configuración | Dataset train |
|---|---|
| Base | `xBDDetectionDataset` (sin augmentation) |
| Con augmentation | `xBDDetectionDatasetAug` (con augmentation) |

**Métricas de evaluación:** Precision, Recall, F1 — global y por clase de daño
**Umbrales:** `TH_SCORE=0.5`, `TH_IOU=0.5`

**Resultado esperado (bullet):**
- Mayor Recall (menos falsos negativos)
- Ligera caída en Precision
- F1 igual o superior, especialmente en clases minoritarias (`major-damage`, `destroyed`)

---

## Notas de presentación (oral)

- **Tiempo estimado:** 4-5 minutos
- **Énfasis principal:** la slide 2 (diferencia con clasificación) y la slide 6 (random crop, la más compleja)
- **Pregunta anticipada:** "¿Por qué ±15° y no más?" → Rotaciones mayores generan cajas muy holgadas; para objetos pequeños como edificios eso introduce demasiado ruido en las anotaciones
- **Pregunta anticipada:** "¿Por qué eliminar cajas por el centroide y no por solapamiento?" → Es una heurística simple y efectiva; alternativas como área mínima visible son más costosas y no aportan diferencia práctica en este dataset
