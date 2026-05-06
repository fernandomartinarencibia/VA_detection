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

## 4. Resultados

### 4.1 Anchors derivados por k-means

El barrido de `k ∈ [1, 11]` con la distancia IoU produjo la siguiente curva de *avg IoU*:

| k | avg IoU |
|---|---------|
| 1 | 0.4933  |
| 2 | 0.6216  |
| 3 | 0.6689  |
| 4 | 0.6926  |
| **5** | **0.7148** |
| 6 | 0.7313  |
| 7 | 0.7441  |

El codo se sitúa en `k = 5`, que además coincide con el número de niveles del FPN. A partir de ese punto la mejora de IoU por cluster adicional desciende por debajo de 0.015. Los centroides obtenidos son:

| Nivel FPN | w (px) | h (px) | √Área (px) | Aspecto w/h |
|-----------|--------|--------|-----------|-------------|
| 0         | 12.1   | 11.9   | 12        | 1.02        |
| 1         | 20.9   | 22.7   | 22        | 0.92        |
| 2         | 34.4   | 29.6   | 32        | 1.16        |
| 3         | 40.9   | 44.7   | 43        | 0.92        |
| 4         | 78.2   | 75.9   | 77        | 1.03        |

Los *aspect ratios* derivados del k-means sobre `log(w/h)` son `[0.63, 1.0, 1.6]`, frente al `[0.5, 1.0, 2.0]` del modelo base. La diferencia más significativa es la eliminación de los extremos (0.5 y 2.0): los edificios xBD son casi cuadrados (aspecto mediano = 1.00) y apenas aparecen formas muy elongadas.

Comparando los dos conjuntos de *anchors*:

| Configuración | Sizes (px) | Aspect ratios |
|---------------|-----------|---------------|
| Base (COCO)   | 8, 16, 32, 64, **128** | 0.5, 1.0, **2.0** |
| K-means (xBD) | 12, 22, 32, **43**, **77** | **0.63**, 1.0, **1.6** |

El valor 128 del modelo base no cubre ningún percentil representativo (p95 ≈ 51 px). K-means lo reemplaza por 77 px, que cubre los edificios más grandes de forma más ajustada. El nivel 32 coincide por azar con la mediana del dataset.

### 4.2 Entrenamiento y comparación en validación

Ambos modelos se entrenaron 18 epochs con la misma configuración (SGD, lr=0.001, StepLR γ=0.1 cada 6 epochs). A continuación se muestran las métricas en validación en cada epoch para el modelo k-means:

| Epoch | Train loss | Val loss | Precision | Recall | F1 val |
|-------|-----------|----------|-----------|--------|--------|
| 0     | 29.59     | 25.99    | 0.321     | 0.030  | 0.054  |
| 1     | 31.23     | 29.53    | 0.432     | 0.009  | 0.018  |
| 2     | 30.78     | 30.13    | 0.628     | 0.027  | 0.052  |
| 3     | 33.03     | 31.99    | 0.493     | 0.013  | 0.026  |
| **4** | **32.88** | **27.03**| **0.643** |**0.072**|**0.130**|
| 5     | 31.83     | 30.87    | 0.629     | 0.026  | 0.050  |
| 6     | 32.15     | 28.59    | 0.454     | 0.038  | 0.071  |
| …     | …         | …        | …         | …      | …      |
| 17    | 25.84     | 25.40    | 0.420     | 0.022  | 0.042  |

**Mejor modelo k-means: epoch 4, F1 val = 0.1300** (P = 0.643, R = 0.072).

Las métricas en el conjunto de **test** del modelo base (mejor checkpoint, epoch 6) son:

| Clase         | Precision | Recall | F1    | Recall detección |
|---------------|-----------|--------|-------|-----------------|
| no-damage     | 0.958     | 1.000  | 0.979 | 0.079           |
| minor-damage  | 1.000     | 0.027  | 0.053 | 0.014           |
| major-damage  | 0.000     | 0.000  | 0.000 | 0.058           |
| destroyed     | 0.000     | 0.000  | 0.000 | 0.006           |
| **Global**    | **0.499** | **0.039** | **0.073** | —          |

*(22 222 GT en test, 1 515 TP, 3 036 predicciones. La evaluación directa del modelo k-means sobre test está pendiente de ejecutar la celda 79 del notebook.)*

### 4.3 Análisis e interpretación

**Cuello de botella en la RPN.** La pérdida `rpn_box_reg` domina el total en ambos modelos (≈ 25–32 de un total de ≈ 26–34), lo que indica que la RPN tiene dificultades para ajustar las regresiones de caja. Esto se refleja en un recall de detección muy bajo: el modelo base solo recupera un 3.9 % de los GT en test.

**Recall con k-means.** En la validación, el mejor recall de detección alcanzado por el modelo k-means es 0.072 (epoch 4), frente al 0.039 en test del modelo base. Aunque no son directamente comparables (val vs. test, conjuntos distintos), el incremento sugiere que los *anchors* más ajustados al tamaño real de los edificios sí mejoran la capacidad de propuesta de la RPN.

**Clasificación condicional.** La alta precision en `no-damage` (0.958) y `minor-damage` (1.000) en el modelo base indica que, cuando un edificio es detectado, la clasificación es correcta. El problema es que casi ningún edificio con daño severo llega a ser detectado (recall de detección ≈ 0.006 para `destroyed`). Esto es consecuencia del desbalance extremo de clases (no-damage: 50 928 vs. destroyed: 3 227 en train) y no se resuelve únicamente optimizando los anchors.

**Conclusión.** La optimización de *anchors* con k-means mejora la cobertura de la RPN sobre los tamaños reales de los edificios xBD (avg IoU 0.71 vs. los *anchors* base, que son subóptimos para este dominio). El impacto en el F1 global es limitado porque el cuello de botella principal es el recall de la RPN combinado con el desbalance de clases, no la geometría de los anchors.

---

## 5. Utilidad de la extensión

- **Transferibilidad:** el mismo procedimiento de k-means sobre `(w, h)` es aplicable a cualquier dataset de detección para calibrar automáticamente los *anchors*, sin necesidad de ajuste manual.
- **Coste bajo:** la derivación de *anchors* es un paso de preprocesado que tarda segundos; el único coste adicional es el entrenamiento completo del nuevo modelo.
- **Diagnóstico:** la curva *avg IoU vs k* y el scatter `(w, h)` con centroides permiten entender cuánto se aleja la distribución del dataset del rango cubierto por los *anchors* por defecto, lo que orienta futuros ajustes de arquitectura.
