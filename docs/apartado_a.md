# Apartado 4.7a — Optimización de *anchors* con *k-means*

**Asignatura:** APAI — Aprendizaje Profundo Aplicado a la Imagen  
**Proyecto 2B:** Detección de objetos sobre imágenes de satélite  
**Notebook:** `APAI_Pr2B_ObjectDetection_2025_2026.ipynb` — Celda 77–79

---

## 1. Motivación

El modelo base utiliza los *anchors* por defecto de Faster R-CNN con FPN: tamaños `(8, 16, 32, 64, 128)` píxeles y ratios de aspecto `(0.5, 1.0, 2.0)`. Estos valores están optimizados para COCO, donde los objetos cubren un rango de escala amplio. Sin embargo, el análisis del dataset xBD (§4.2 del notebook) revela que los edificios son objetivos pequeños y compactos:

| Percentil | √Área (px) |
|-----------|-----------|
| p5        | ≈ 9       |
| p25       | ≈ 17      |
| Mediana   | ≈ 26      |
| p75       | ≈ 36      |
| p95       | ≈ 51      |

El valor `128` del conjunto base no tiene representación estadística en el dataset, mientras que el grueso de los edificios se concentra entre 9 y 51 px. Un conjunto de *anchors* mal calibrado reduce la tasa de *recall* de la RPN, ya que propone pocas ventanas que se solapen suficientemente con los objetos reales.

**Objetivo:** derivar *anchors* óptimos mediante *k-means* directamente sobre la distribución empírica de tamaños y formas del dataset xBD, y comparar el rendimiento con el modelo base.

---

## 2. Método: k-means con distancia IoU

### 2.1 Distancia utilizada

Se aplica la variante de *k-means* propuesta en YOLOv2, que emplea `1 − IoU(caja, centroide)` como función de distancia en lugar de la distancia euclídea. La ventaja es que esta métrica es **invariante a la escala**: una caja pequeña con una forma determinada se agrupa con centroides de forma similar, independientemente de su tamaño absoluto, lo que produce *anchors* más representativos de la forma que de la posición.

```
d(box, centroide) = 1 − IoU(box, centroide)
```

donde IoU se calcula asumiendo que las cajas están centradas en el origen (solo importan w y h).

### 2.2 Selección de k

Se ejecuta el algoritmo para `k ∈ [1, 11]` y se traza la curva *avg IoU vs k* para identificar el codo. Con `k = 5` (un *anchor* por nivel del FPN) se obtiene un compromiso razonable entre cobertura y número de *anchors*, manteniendo la misma arquitectura RPN que el modelo base.

### 2.3 Derivación de tamaños y ratios de aspecto

- **Sizes:** raíz cuadrada del área de cada uno de los 5 centroides, ordenados de menor a mayor área. Sustituyen directamente a `(8, 16, 32, 64, 128)`.
- **Aspect ratios:** *k-means* estándar con `k = 3` sobre `log(w/h)` de todas las cajas del train, exponenciando los centros para obtener los tres ratios más representativos del dataset.

Los valores derivados reflejan que los edificios xBD son predominantemente compactos (ratio ≈ 1) y no tienen la elongación extrema (0.5 o 2.0) que sí aparece frecuentemente en COCO (coches, personas, etc.).

---

## 3. Implementación

```python
# K-means IoU sobre (w, h) del train
centroids5, iou5 = kmeans_iou(all_wh, k=5)

# Sizes: √área de cada centroide
km_sizes_list = [max(4, int(round(np.sqrt(c[0]*c[1])))) for c in centroids5]

# Aspect ratios: k-means sobre log(w/h)
km_ar = KMeans(n_clusters=3).fit(np.log(all_wh[:,0]/all_wh[:,1]).reshape(-1,1))
ar_centers = sorted(np.exp(km_ar.cluster_centers_.flatten()))

# AnchorGenerator adaptado
anchor_gen_km = AnchorGenerator(
    sizes=tuple((s,) for s in km_sizes_list),
    aspect_ratios=tuple([tuple(ar_centers)] * 5)
)
```

El modelo `get_model_xbd_km()` es idéntico al base salvo por el `AnchorGenerator`. Se entrena durante los mismos 18 epochs con la misma configuración (SGD, StepLR) y se evalúa con los mismos umbrales (`TH_SCORE = 0.5`, `TH_IOU = 0.5`) sobre el conjunto de test.

Los resultados y checkpoints se guardan en `xbd_kmanchors_results/` para no solaparse con el experimento base en `xbd_damage_detection_results/`.

---

## 4. Análisis y resultados esperados

La comparación entre ambos modelos se resume en:

| Modelo | Anchors (sizes) | Aspect ratios | Precision | Recall | F1 |
|--------|----------------|---------------|-----------|--------|----|
| Base   | 8, 16, 32, 64, 128 | 0.5, 1.0, 2.0 | — | — | — |
| K-means | derivados de xBD | derivados de xBD | — | — | — |

*(Los valores se completan al ejecutar la celda 79 del notebook.)*

Se espera que los *anchors* k-means mejoren el **recall** de la RPN: al estar mejor alineados con los tamaños reales, una mayor proporción de los edificios recibirá al menos una propuesta con IoU > 0.5, lo que alimenta mejor la etapa de clasificación ROI. La mejora puede ser más notable en las clases minoritarias (`major-damage`, `destroyed`) donde el recall es el factor limitante.

Si la mejora es marginal, la interpretación es que los *anchors* base `(8, 16, 32, 64, 128)` ya cubren el rango [9, 51] px con suficiente densidad, y el cuello de botella del modelo se encuentra en la etapa de clasificación o en el desbalance de clases, no en la RPN.

---

## 5. Utilidad de la extensión

- **Transferibilidad:** el mismo procedimiento de k-means sobre `(w, h)` es aplicable a cualquier dataset de detección para calibrar automáticamente los *anchors*, sin necesidad de ajuste manual.
- **Coste bajo:** la derivación de *anchors* es un paso de preprocesado que tarda segundos; el único coste adicional es el entrenamiento completo del nuevo modelo.
- **Diagnóstico:** la curva *avg IoU vs k* y el scatter `(w, h)` con centroides permiten entender cuánto se aleja la distribución del dataset del rango cubierto por los *anchors* por defecto, lo que orienta futuros ajustes de arquitectura.
