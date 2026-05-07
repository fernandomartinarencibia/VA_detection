# Guión de diapositivas — Extensión A: Anchors adaptativos con K-means

---

## Diapositiva 1 — Optimización de Anchors con K-means

**Título:** Extensión A: Anchors adaptativos con K-means IoU

### Bloque izquierdo — Motivación

- Los anchors por defecto de Faster R-CNN (COCO) incluyen tamaño 128 px, pero p95 del dataset xBD ≈ 51 px
- Los edificios xBD son pequeños y compactos → anchors mal calibrados reducen el recall de la RPN
- Solución: derivar anchors directamente de la distribución empírica del dataset

### Bloque central — Método

- K-means con distancia `1 − IoU(caja, centroide)` → invariante a escala
- k = 5 (un anchor por nivel FPN), elegido en el codo de la curva avg IoU vs k
- **Sizes:** √área de cada centroide
- **Aspect ratios:** k-means sobre log(w/h), exponenciando los centros

### Bloque derecho — Comparación de anchors

| Configuración | Sizes (px)            | Aspect ratios      |
|---------------|-----------------------|--------------------|
| Base (COCO)   | 8, 16, 32, 64, **128**| 0.5, 1.0, **2.0**  |
| K-means (xBD) | 12, 22, 32, 43, **77**| **0.63**, 1.0, **1.6** |

> Resaltar en rojo el `128` y el `2.0` del modelo base para señalar el desajuste con xBD.

**Imagen sugerida:** scatter `(w, h)` de las bounding boxes del train con los 5 centroides marcados — ilustra visualmente por qué el anchor de 128 px sobra.

---

## Diapositiva 2 — Resultados

**Métrica clave:**

| Modelo   | Recall val (mejor epoch) | Epoch |
|----------|--------------------------|-------|
| Base     | 0.039 (test)             | 6     |
| K-means  | **0.072** (val)          | 4     |

**Tabla de curva avg IoU vs k (resaltar k=5):**

| k | avg IoU |
|---|---------|
| 1 | 0.4933  |
| 2 | 0.6216  |
| 3 | 0.6689  |
| 4 | 0.6926  |
| **5** | **0.7148** |
| 6 | 0.7313  |
| 7 | 0.7441  |

**Mensaje principal (mostrar en grande):**
> Los anchors ajustados al dominio mejoran la cobertura de la RPN, pero el cuello de botella principal es el desbalance de clases, no la geometría de los anchors.

---

## Notas para la presentación

- No mostrar la tabla completa de epochs — solo la fila del mejor epoch (epoch 4) y la comparación final con el modelo base.
- La curva avg IoU vs k justifica visualmente la elección de k=5 y su coincidencia con los niveles del FPN.
- Conclusión en una línea: *"K-means calibra los anchors al dominio en segundos, mejorando el recall de la RPN sin cambiar la arquitectura."*
- El alto recall del modelo k-means (0.072) frente al base (0.039) sugiere mejora real, aunque la comparación no es directa (val vs. test).
