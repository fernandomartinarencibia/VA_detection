# Conclusiones — Primer Avance: Detección de Edificios con Faster R-CNN en xBD

## 1. Configuración del experimento

| Componente | Valor |
|---|---|
| Modelo | Faster R-CNN + ResNet-50 FPN (pretrained) |
| Parámetros entrenables | 41,076,761 |
| Dataset train/val/test | 256 / 45 / 63 imágenes |
| Edificios en train | 64,078 instancias |
| Resolución | 1024×1024 px |
| Épocas | 20 |
| Optimizador | SGD (lr=0.001, momentum=0.9, wd=0.0005) |
| Scheduler | StepLR (step_size=8, γ=0.1) |
| Batch size | 1 (forzado por bug en torchvision_05) |
| Umbral confianza / IoU | 0.5 / 0.5 |

### Anchors personalizados
Se adaptaron los anchors a la distribución real de los edificios en xBD (muy pequeños para imágenes de satélite):

| Estadístico | √Área (px) |
|---|---|
| p5 | ~9 |
| p25 | ~17 |
| Mediana | ~26 |
| p75 | ~36 |
| p95 | ~51 |

Anchors elegidos: `(8, 16, 32, 64, 128)` px con aspect ratios `(0.5, 1.0, 2.0)`, cubriendo el rango p5–p95 de la distribución y alineados con los 5 niveles del FPN.

---

## 2. Resultados finales en test

El mejor modelo (seleccionado por F1 en validación, epoch 5) obtiene en el conjunto de test:

| Métrica | Valor |
|---|---|
| **Precision** | 0.3861 |
| **Recall** | 0.0837 |
| **F1** | **0.1376** |
| GT buildings | 22,577 |
| True Positives (TP) | 1,890 |
| Predicciones totales | 4,895 |
| Falsos Positivos (FP) | ~3,005 |
| Falsos Negativos (FN) | ~20,687 |

El modelo **detecta solo el 8.4% de los edificios existentes**, lo que indica un fallo grave en recall. La precisión (38.6%) es moderada: de cada 10 bounding boxes predichos, menos de 4 corresponden a un edificio real.

---

## 3. Evolución del entrenamiento

### Pérdidas por época (selección)

| Época | Train total | Val total | Precision | Recall | F1 val |
|---|---|---|---|---|---|
| 0 | 45.45 | 37.51 | 0.200 | 0.0008 | 0.0016 |
| 2 | 53.83 | 30.07 | 0.159 | 0.0831 | 0.1091 |
| 4 | 39.84 | 20.96 | 0.477 | 0.0551 | 0.0988 |
| **5** | **41.84** | **39.56** | **0.437** | **0.119** | **0.187** ← mejor |
| 6 | 36.28 | 20.09 | 0.234 | 0.0051 | 0.0100 |
| 8 | 56.52 | 48.01 | 0.125 | 0.0122 | 0.0222 |
| 10–19 | ~50–58 | ~43–45 | ~0.17–0.25 | ~0.021 | ~0.04 |

### Observaciones clave sobre las curvas

1. **Las pérdidas son extremadamente altas y dominadas por `rpn_box_reg`** (que representa el 97–99% de la pérdida total en la mayoría de épocas). Los demás componentes (`objectness`, `box_reg`, `classifier`) son prácticamente despreciables. Esto sugiere que el RPN tiene dificultades para ajustar los offsets de los anchors a los objetos pequeños.

2. **Inestabilidad extrema** en val loss: oscila entre 20 y 52 a lo largo del entrenamiento sin una tendencia clara de convergencia.

3. **El LR decay en epoch 8 (×0.1 → lr=0.0001) no mejora el modelo**: a partir de esa época las pérdidas de entrenamiento aumentan (~56–58) y las métricas de validación se estabilizan en valores bajos (F1 ≈ 0.04). El scheduler reduce demasiado agresivamente el learning rate.

4. **Mejor epoch en validación: época 5** (F1=0.187). La mejora se interrumpe bruscamente en la época 6, probablemente por sobreajuste a minibatches particulares con batch_size=1 y alta varianza del gradiente.

5. La `val_objectness` cae de 0.015 (época 0) a 0.001 (época 18), lo que indica que el RPN aprende a suprimir propuestas fácilmente negativas, pero el `rpn_box_reg` no mejora en paralelo → las propuestas sobreviven al filtrado pero están mal localizadas.

---

## 4. Problemas identificados

### 4.1 Recall muy bajo (8.4%)
El modelo es incapaz de detectar la gran mayoría de edificios. Con ~22,577 GT y solo 4,895 predicciones para 63 imágenes (≈78 predicciones/imagen), el modelo infradetecta masivamente. Los edificios en xBD son densos (hasta 185 por imagen en train), lo que sugiere que el RPN no genera suficientes propuestas positivas en zonas de alta densidad.

### 4.2 Pérdida rpn_box_reg dominante y sin convergencia
El RPN aprende que hay objetos (objectness cae) pero no consigue refinar las cajas correctamente (rpn_box_reg se mantiene alta). Posibles causas:
- Los anchors, aunque ajustados al rango de tamaños, pueden no estar bien calibrados para imágenes de satélite con ground sampling distance diferente.
- Batch size=1 introduce alta varianza en los gradientes, especialmente con objetos pequeños.
- La pérdida no está ponderada: `rpn_box_reg` domina pero recibe el mismo peso que el resto.

### 4.3 Inestabilidad del entrenamiento con SGD y batch_size=1
La combinación de batch_size=1, lr=0.001 y SGD sin warm-up provoca una optimización muy ruidosa. El modelo oscila entre estados buenos y malos entre épocas consecutivas (e.g., F1=0.187 en época 5 → F1=0.010 en época 6).

### 4.4 Resolución de entrada reducida
Faster R-CNN redimensiona internamente la imagen para que el lado corto sea 800px (min_size=800). Las imágenes 1024×1024 se escalan a ~800×800, lo que reduce los edificios ya pequeños (~26px de mediana) a ~20px, acercándolos al límite de detección del anchor más pequeño (8px).

### 4.5 Normalización de imagen
Se detectó y corrigió un bug crítico: las imágenes xBD son uint16 pero con rango efectivo 0–510 (no 0–65535). Dividir por 65535 dejaba los píxeles en ~0.008, causando NaN en el RPN de CUDA. La solución fue dividir por 255 con clip, igualando el preprocesado del Proyecto 1.

---

## 5. Diagnóstico global

El modelo base **es funcional pero muy limitado** para este dataset. El principal cuello de botella es el **recall extremadamente bajo**, causado por:
1. El RPN no genera propuestas suficientes en zonas densas de pequeños edificios.
2. La inestabilidad del entrenamiento impide consolidar las mejoras.
3. La pérdida `rpn_box_reg` domina sin converger, lo que sugiere que los anchors o la escala de imagen necesitan mayor ajuste.

La **precisión moderada (38.6%)** indica que cuando el modelo detecta algo, en un 38% de los casos es correcto, lo que sugiere que la red backbone y el clasificador tienen capacidad discriminativa, pero el RPN es el componente problemático.

---

## 6. Líneas de mejora para el siguiente avance

| Prioridad | Mejora | Justificación |
|---|---|---|
| Alta | Sustituir SGD por **AdamW** con LR más bajo (~5e-4) | Reduce inestabilidad con batch_size=1 |
| Alta | **Warm-up del LR** (primeras 500 iteraciones) | Evita divergencia inicial del RPN |
| Alta | Ajustar `rpn_nms_thresh` y `rpn_post_nms_top_n` | Generar más propuestas para objetos densos |
| Media | **Data augmentation**: flip, rotación, color jitter | xBD tiene imágenes de satélite con orientación arbitraria |
| Media | Usar `score_thresh=0.3` en evaluación | Recuperar más TPs sacrificando algo de precisión |
| Media | Probar `min_size=1024` (sin downscale) | Preservar el tamaño original de los pequeños edificios |
| Baja | Balancear pesos de pérdidas: reducir peso de `rpn_box_reg` | Evitar que domine el entrenamiento |
| Baja | Probar backbone ResNet-101 FPN o EfficientDet | Mayor capacidad para features de pequeños objetos |

---

*Análisis generado a partir de: `xbd_detection_results/log.csv`, outputs del notebook `APAI_Pr2B_xBD_Detection.ipynb` y checkpoints en `xbd_detection_results/`.*
