# Guión diapositiva — Extensión e) Evaluación por tipo de desastre

---

## Estructura sugerida (1 diapositiva principal + 1 de resultados)

---

### Diapositiva 1 — Motivación e implementación

**Título:** Evaluación estratificada por tipo de desastre

**Cuerpo (bullets breves):**

- El dataset xBD mezcla huracanes, terremotos, tsunamis, incendios… con apariencias y daños muy distintos
- Un F1 medio aceptable puede ocultar fallos graves en escenarios concretos
- **Objetivo:** detectar dónde falla el modelo y orientar mejoras futuras

**Implementación (bloque visual / código pequeño):**

```python
disaster = path.split('/')[t_idx + 1]   # extrae el desastre del path
```

- Sin modificar el dataloader → un único paso de inferencia acumula métricas por desastre
- Métricas calculadas: **Precision · Recall · F1** global por desastre + **F1 por clase × desastre**

---

### Diapositiva 2 — Resultados y visualizaciones

**Título:** ¿Qué revela la evaluación por desastre?

**Columna izquierda — gráfico de barras** (`disaster_metrics.png`):
- Tres subgráficas: P / R / F1 por desastre
- Paleta RdYlGn (rojo = bajo, verde = alto)
- Línea discontinua = media global de referencia
- **Mensaje clave:** identificar desastres por encima/debajo del promedio de un vistazo

**Columna derecha — mapa de calor** (`disaster_class_f1_heatmap.png`):
- Ejes: desastre (Y) × clase de daño (X)
- Cada celda = F1 de clasificación de daño
- **Mensaje clave:** ¿la clase *major-damage* falla en todos los desastres o solo en terremotos?

**Pie de diapositiva / conclusión oral:**

> "Esta evaluación estratificada es el punto de partida para detectar sesgos del modelo y diseñar fine-tuning o augmentation dirigidos a los escenarios más difíciles."

---

## Lo que hay que decir en voz alta (≈60 s)

1. **Problema** — evaluar solo con métricas globales enmascara comportamientos dispares entre escenarios.
2. **Solución** — extraemos el tipo de desastre del path de cada imagen sin tocar el dataloader.
3. **Métricas** — calculamos P/R/F1 global por desastre y además un F1 desglosado por clase de daño para cada desastre.
4. **Visualizaciones** — las barras permiten comparar desastres de un vistazo; el mapa de calor cruza desastre con clase para localizar combinaciones problemáticas.
5. **Utilidad** — si un desastre sale en rojo, la acción es clara: más datos, augmentation específica o ajuste del umbral de score.

---

## Consejos de diseño

| Elemento | Recomendación |
|---|---|
| Paleta de colores | Usar RdYlGn coherente con las gráficas generadas |
| Tamaño de fuente | Título ≥ 28 pt, bullets ≥ 20 pt |
| Imágenes | Insertar `disaster_metrics.png` y `disaster_class_f1_heatmap.png` lado a lado |
| Extensión | Máximo 2 diapositivas; si hay tiempo, añadir tabla de acciones sugeridas |
| Énfasis oral | Señalar una celda concreta del heatmap como ejemplo durante la presentación |
