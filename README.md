# Evaluación de Técnicas Forenses en la Detección del Origen de Imágenes Digitales

Este repositorio contiene el framework experimental desarrollado para el Trabajo Fin de Grado (TFG) en Ingeniería Informática (Escuela de Ingeniería y Arquitectura, Universidad de Zaragoza). El objetivo del proyecto es auditar y comparar de forma empírica la robustez de los dos paradigmas predominantes en la Identificación de Cámara de Origen (SCI): el enfoque físico clásico (**PRNU**) y el enfoque moderno basado en aprendizaje profundo (**Noiseprint**), analizando el impacto crítico del postprocesado móvil y la compresión en redes sociales.

## Estructura del Proyecto

El repositorio se ha estructurado de manera modular para garantizar la limpieza del código y la fácil localización de los artefactos generados por el pipeline:

```text
├── noiseprint-master/noiseprint-master/
│   ├── noiseprint/          # Código base y modelos CNN de Noiseprint
│   ├── prnu/                # Código del extractor estadístico de PRNU
│   │
│   ├── TFG/                 # Directorio principal de datos del TFG
│   │   ├── dataset/         # Fotografías nativas de control (CSAFE, iPhone 14 Pro, iPhone 15)
│   │   ├── estadisticas/    # Reportes globales y matrices de confusión generadas en PDF
│   │   ├── huellas/         # Mapas de ruido residual individuales extraídos de cada imagen
│   │   ├── maestras/        # Huellas Maestras de referencia calculadas para cada dispositivo
│   │   ├── tests/           # Conjuntos de imágenes bajo escenarios de estrés (WhatsApp, Instagram, Filtros)
│   │   └── visualizacion/   # Mapas de características visuales (.png) extraídos de las huellas
│   │
│   ├── main_TFG.py          # Script principal interactivo (Menú de ejecución del pipeline)
│   └── main_Visualizar.py   # Script de generación y renderizado de mapas de ruido
│
├── .gitignore               # Configuración de exclusión para el control de versiones
└── README.md                # Documentación del 
```

## Requisitos e Instalación

Asegurarse de instalar las dependencias necesarias que se detallan en la memoria, prestando especial atención a las versiones estables y compatibles de TensorFlow, OpenCV y NumPy requeridas para la ejecución de los modelos.


