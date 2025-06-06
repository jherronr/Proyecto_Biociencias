# Explicación del Proceso

## 1. Fuente de datos
- **Imágenes**: Dataset Brain Cancer recogido de https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset 
- **Texto**: CSV de notas clínicas.

## 2. Análisis descriptivo e inferencial
- **Variables categóricas**:  
  - Prueba Chi-cuadrado entre Treatment vs Condition → estadístico = 1273.788, p < 0.001 (significativo).  
  - Prueba Chi-cuadrado entre Treatment vs Sex → estadístico = 2.833, p = 0.418 (no significativo). Se descarta la variable para el modelo.  
- **Variable continua (Edad)**:  
  - Kruskal-Wallis → H = 2.865, p = 0.4129 (no diferencia significativa entre grupos de tratamiento). Se descarta la variable para el modelo.

## 3. Feature Engineering
### 3.1. Imágenes
- **CropBlackBorders**: Se recortan bordes negros para centrar la atención en la región de interés.  
- **Resize(224,224)**: Tamaño estándar para EfficientNet-B0.  
- **ToTensor + Normalize([-1,1])**: Normalización recomendada por PyTorch.

### 3.2. Texto
- **Build_input_text**: `"Condition: {condition}. Clinical note: {clinical_note}"`  
  - Justificación: introducimos explícitamente la condición (tumor) en la entrada para que ClinicalBERT capture esa información desde inicio. Se crea una variable llamada input_text que tiene la condicion y el tratamiento.  
- **Tokenización**: `max_length=50`, truncamiento si es necesario.  
  - Razón: Basado en longitud media de las notas clínicas (aprox. 40–45 tokens).

## 4. Modelos
### 4.1. EfficientNet-B0
- **Transfer learning** con pesos de ImageNet para ahorrar tiempo y mejorar generalización.  
- Capa final (nn.Linear) ajustada a 3 clases.  
- Hiperparámetros:  
  - batch_size=32  
  - lr=0.001 para la capa final  
  - epochs=10  
- **Razones de elección**:  
  - [Eficiencia en parámetros vs Accuracy].  
  - Buen performance en imágenes médicas.

### 4.2. ClinicalBERT
- **Modelo base**: `emilyalsentzer/Bio_ClinicalBERT`, preentrenado en texto clínico (MIMIC-III).  
- **Clasificación**: 4 clases de tratamiento.  
- Hiperparámetros:  
  - batch_size=16  
  - lr=2e-5  
  - epochs=4  
- **Razón**:  
  - Especializado en lenguaje clínico; supera a BERT-base en este dominio.

---

## 5. Resultados finales
- **Accuracy EfficientNet-B0**: 0.94 en validación, 0.93 en test.  
- **Accuracy ClinicalBERT**: 0.88 en validación, 0.86 en test.  
- **Matriz de confusión imagen**:
  ![Matriz de confusión imagen](r"C:\Users\camil\OneDrive\Documentos\PruebaBiociencias\Proy_Biociencias\Proyecto_Biociencias\docs\matriz_confusion_img.png")

- **Matriz de confusión texto**:
  ![Matriz de confusión texto](r"C:\Users\camil\OneDrive\Documentos\PruebaBiociencias\Proy_Biociencias\Proyecto_Biociencias\docs\matriz_confusion_text.png")

