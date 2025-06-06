# Proyecto_Biociencias

# Proyecto: Clasificación de Tumor Cerebral + Predicción de Tratamiento

Este repositorio contiene todo el proceso de:
1. **Análisis descriptivo e inferencial** de datos clínicos y de imágenes.  
2. **Feature Engineering**:  
   - Preprocesamiento de imágenes (Crop black borders, resize, normalización).  
   - Transformación de texto clínico (pipeline de ClinicalBERT).  
3. **Entrenamiento y validación** de los modelos:  
   - **EfficientNet-B0** para clasificar imágenes de tumores cerebrales.  
   - **ClinicalBERT** para clasificar tratamiento a partir de la condición y nota clínica.  
4. **Despliegue de la API** con FastAPI, dividida en dos endpoints:
   - `/predict_image`: recibe una imagen y devuelve la etiqueta de tumor.  
   - `/predict_treatment`: recibe la etiqueta de tumor (`condition`) y la nota clínica, y devuelve el tratamiento.  

---

## Contenido del repositorio

```text
PROYECTO_BIOCIENCIAS/
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── analisis-brain-tumor.ipynb       # Notebook con todo el proceso
│   └── analisis-brain-tumor.html        # Versión HTML para lectura fácil
│   ├── analisis-inferencial-treatment.ipynb       # Notebook con todo el proceso
│   └── analisis-inferencial-treatment.html        # Versión HTML para lectura fácil
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── brain_conditions_detailed_dataset.csv      # Dataset con los datos de tratamiento
│   │   ├── preprocessing_images.py      # Funciones para preprocesar imágenes
│   │   ├── preprocessing_text.py        # Funciones para texto (tokenizer, build_input_text)
│   │   └── utilities.py                 # Helpers (split, guardar checkpoints)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_image_model.py         # Script para entrenar EfficientNet-B0
│   │   ├── train_text_model.py          # Script para entrenar ClinicalBERT
│   │   └── evaluate_models.py           # Funciones de evaluación y gráficos
│   │
│   ├── api/
│       ├── __init__.py
│       ├── app.py            # FastAPI con dos endpoints: /predict_image y /predict_treatment
│       └── crop_utils.py                # Clase CropBlackBorders para la API
│   
│
└── docs/
    ├── diagramas.png                    # Diagrama de flujo global
    ├── pantallazos_api.pdf              # Capturas de Swagger UI / Postman
    ├── explicación_proceso.md           # Explicación paso a paso con justificaciones
    └── justificaciones_modelos.md       # Por qué se eligieron EfficientNet y BERT, tests de hipótesis, etc.
