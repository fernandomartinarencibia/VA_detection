# Apartado e) — Evaluación por tipo de desastre

## Motivación

El dataset xBD agrupa imágenes de distintos desastres naturales (huracanes, terremotos, tsunamis, incendios…) que difieren en apariencia visual, densidad urbana y tipo de daño predominante. Evaluar el modelo sobre el conjunto de test global puede enmascarar comportamientos muy dispares entre escenarios: un F1 medio aceptable puede ocultar que el modelo funciona bien en huracanes pero falla casi por completo en terremotos.

La evaluación estratificada por desastre permite:

- Identificar qué escenarios son más difíciles para el detector.
- Detectar si el rendimiento cae en tipos de daño que aparecen principalmente en ciertos desastres (e.g., *destroyed* en tsunamis).
- Orientar futuras mejoras: más augmentation para escenarios difíciles, fine-tuning específico, etc.

---

## Implementación

### Extracción del tipo de desastre

Cada imagen del dataset sigue la ruta `data/xBD_UC3M/test/<desastre>/images/<fichero>.tif`. El tipo de desastre se extrae directamente del path de cada muestra durante la inferencia, sin necesidad de modificar el dataset ni el dataloader:

```python
parts    = path.split('/')
t_idx    = parts.index('test')
disaster = parts[t_idx + 1]   # e.g. 'hurricane-florence'
```

Esto permite reutilizar `data_loader_test2` sin cambios y acumular estadísticas independientes por desastre en un solo paso de inferencia.

### Acumulación de estadísticas

Por cada desastre se mantienen contadores separados:

| Variable | Descripción |
|---|---|
| `rel[c]` | Ground-truth boxes de clase `c` |
| `ret_rel[c]` | True positives de clase `c` (IoU > umbral y clase correcta) |
| `ret` | Total de predicciones (con score > umbral) |
| `y_true`, `y_pred` | Pares GT/pred sobre los TP, para métricas por clase |

El criterio de matching es idéntico al de `test_detection_model`: una predicción es TP si su IoU con la GT supera `th_iou` y coincide con la clase.

---

## Métricas calculadas

### 1. Precision, Recall y F1 globales por desastre

Se calculan siguiendo la misma fórmula que en la evaluación global:

- **Precision** = TP_total / predicciones_total
- **Recall** = media del recall por clase de daño (macro)
- **F1** = media armónica de P y R

Esto da una visión rápida de qué desastres son más fáciles o más difíciles en términos de detección pura.

### 2. F1 por clase de daño × desastre

Usando `precision_recall_fscore_support` de scikit-learn sobre los pares (GT, pred) de los TP, se obtiene el F1 de clasificación del daño por separado para cada combinación desastre × clase. Esto revela si el modelo confunde las categorías de daño de forma diferente según el tipo de evento.

---

## Visualizaciones

### Barras de P / R / F1 por desastre

Tres subgráficas (una por métrica) con barras coloreadas mediante la paleta `RdYlGn` (rojo = bajo, verde = alto). Una línea discontinua marca la media global como referencia. El objetivo es comparar de un vistazo qué desastres quedan por encima o por debajo del promedio.

Guardado en: `xbd_damage_detection_results/disaster_metrics.png`

### Mapa de calor F1 (desastre × clase de daño)

Matriz con los desastres en el eje Y y las clases de daño en el eje X, donde cada celda muestra el F1. La paleta `RdYlGn` facilita localizar las combinaciones problemáticas (celdas rojas). Es especialmente útil para detectar si una clase concreta (e.g., *major-damage*) es sistemáticamente difícil en todos los desastres o solo en algunos.

Guardado en: `xbd_damage_detection_results/disaster_class_f1_heatmap.png`

---

## Utilidad y extensiones posibles

| Observación esperada | Acción sugerida |
|---|---|
| F1 muy bajo en un desastre concreto | Incluir ese desastre en train con más peso o augmentation específica |
| Clase *destroyed* con F1 cercano a 0 en terremotos | Revisar si hay suficientes muestras en train; considerar oversampling |
| Alta precisión pero bajo recall en tsunamis | El modelo es conservador: bajar `th_score` para ese escenario |
| F1 homogéneo entre desastres | El modelo generaliza bien; centrar esfuerzos en mejorar clases difíciles |

La evaluación estratificada es el punto de partida natural para cualquier análisis de sesgo del modelo y para diseñar experimentos de fine-tuning o augmentation dirigidos.
