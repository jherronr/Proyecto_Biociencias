# Justificaciones de Selección de Modelos y Técnicas

## 1. Por qué EfficientNet-B0 para imágenes
- **Eficiencia en parámetros**: EfficientNet-B0 usa bloques MobileNet-like y escalado compuesto, alcanzando alta precisión con relativamente pocos parámetros (≈5M).  
- **Rápido convergence**: Permite entrenar rápido incluso en GPU modestas.  
- **Buen rendimiento en datos médicos**: League of Medical Imaging ha mostrado que EfficientNet-B0 es competitivo.

## 2. Por qué `emilyalsentzer/Bio_ClinicalBERT` para texto
- **Dominio clínico**: Preentrenado con notas médicas (MIMIC-III), vocabulario adaptado a terminología clínica.  
- **Mejora en tareas de downstream**: Estudios documentados (observaciones en la literatura) indican que Bio_ClinicalBERT supera a BERT-base en clasificación de notas médicas.

## 3. Pruebas de hipótesis
- **Chi-cuadrado Treatment vs Condition**:  
  - Hipótesis nula: Distribución de Treatment es independiente de Condition.  
  - Resultado: χ²=1273.788, p<0.001 → Rechazamos H₀: sí hay asociación entre tipo de tumor y tratamiento asignado.  
- **Chi-cuadrado Treatment vs Sex**:  
  - Resultado: χ²=2.833, p=0.418 → No hay asociación significativa entre sexo y tratamiento.  
- **Kruskal-Wallis Edad vs Tratamiento**:  
  - Resultado: H=2.865, p=0.4129 → No hay diferencia estadística en edades entre grupos de tratamiento.  
  - Conclusión: La edad no determina el tipo de tratamiento en esta muestra.

## 4. Normalización de imágenes en [-1,1]
- Norm(0.5,0.5) para centrar en 0 y expandir dinámicamente.  
- Evita saturar la activación de capas convolucionales.

## 5. Truncamiento de texto a 50 tokens
- La mayoría de las notas clínicas tienen entre 30 y 60 tokens.  
- 50 permite cubrir la mayor parte del contexto sin aumentar demasiado la complejidad computacional.  
- Evita overfitting y long sequence costs en BERT.

## 6. API en dos pasos
- **Claridad**: El usuario ve en pantalla primero el tumor detectado antes de decidir tratamiento.  
- **Flexibilidad**: Permite cambiar manualmente el “Condition” si, por ejemplo, el médico detecta un error leve en la clasificación de Imagen y quiere forzar otra clase para ver opciones de tratamiento.  
- **Desacoplamiento**: Se puede escalar y desplegar cada servicio por separado (microservicios).

