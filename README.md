# Prácticas IOC: Detección de Objetos y Análisis de Propiedades
## Joan Saurina

Este repositorio contiene el trabajo realizado para las prácticas del Instituto de Organización y Control (IOC), enfocadas en la identificación de objetos, el análisis de sus propiedades y la detección de su posición utilizando técnicas de aprendizaje profundo y visión por computador.

## Fase I: Pipeline para Identificación de Objetos + Detección de Color.

### 1. Creación y Mejora del Dataset

Inicialmente, se intentó crear un dataset utilizando los frames disponibles en el [repositorio YCB Benchmarks](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/). Sin embargo, este enfoque presentó limitaciones significativas:

- Imágenes con un solo objeto
- Fondos no realistas (blancos o neutros)
- Cantidad insuficiente de imágenes
- Perspectivas limitadas, incluso después de la augmentación de datos

Para superar estas limitaciones, se adoptó un dataset más robusto y realista del [YCB Video Dataset](https://chengke.name/ycb-video-dataset-download-mirror/). Este dataset proporciona imágenes anotadas de objetos YCB en escenarios realistas, con un tamaño total de aproximadamente 120GB.

### 2. Arquitectura de Aprendizaje Profundo

Para la detección de objetos, se implementó el modelo YOLO v9, considerado estado del arte (SOTA) en el momento de realización del proyecto. La implementación se basó en el [repositorio oficial de YOLO v9](https://github.com/WongKinYiu/yolov9).

### 3. Entrenamiento y Optimización

El modelo YOLO v9 fue entrenado y optimizado utilizando el dataset YCB mejorado, con un enfoque en evitar el sobreajuste mediante técnicas de augmentación de datos, y provando de encontrar los hiperparámetros óptimos que fueron los utilizados para el entrenamiento definitivo.

### 4. Validación de Resultados

Se realizaron experimentos con objetos reales en el laboratorio para validar la eficacia del modelo entrenado en la detección de objetos YCB.

### 5. Detección y Análisis de Color

Se ha desarrollado un pipeline robusto y eficiente para la detección y análisis de color de los objetos identificados. Este proceso consta de varias etapas interconectadas que aprovechan técnicas avanzadas de visión por computador y aprendizaje profundo:

#### 5.1 Detección de Objetos y Consulta Ontológica

- Se utiliza el modelo YOLO v9 previamente entrenado para detectar objetos en la imagen.
- Para el objeto detectado, se realiza una consulta a una ontología predefinida para obtener información sobre la cantidad esperada de colores.

#### 5.2 Extracción de Regiones de Interés

- Se recorta la imagen original utilizando las coordenadas del bounding box proporcionadas por YOLO v9.
- Este paso permite aislar cada objeto para un análisis más preciso y eficiente.

#### 5.3 Segmentación Semántica

- Se implementa el modelo [Segment Anything](https://github.com/facebookresearch/segment-anything) para una segmentación semántica de alta precisión.
- Este paso genera máscaras detalladas para cada objeto, permitiendo una separación precisa del objeto y el fondo.

#### 5.4 Análisis de Color mediante K-means

- Se aplica el algoritmo K-means sobre las regiones segmentadas para identificar los colores dominantes.
- El número de clusters (k) se ajusta dinámicamente basándose en la información obtenida de la ontología en el paso 5.1.
- Este enfoque permite una detección precisa de los colores exactos presentes en el objeto, incluso en casos de objetos multicolor o con patrones complejos.

Esta pipeline integrada proporciona una solución completa para la detección y análisis de color de objetos en imágenes, combinando técnicas de vanguardia en detección de objetos, segmentación semántica y análisis de clusters. La flexibilidad del sistema permite su adaptación a diversos escenarios y tipos de objetos, garantizando resultados precisos y fiables.

## Fase II: Pipeline para Estimación de Pose y Propiedades (Mediados de Abril 2024 - Mayo 2024)

### 1. Arquitectura para Estimación de Pose

[Pendiente de implementación]

### 2. Identificación de Propiedades de Interés

[Pendiente de implementación]

### 3. Estrategias de Estimación de Propiedades

Se planea desarrollar métodos basados en histogramas o k-means para la estimación de propiedades relevantes de los objetos.

### 4. Validación de Resultados

Se realizarán experimentos adicionales con objetos reales para validar la precisión de las estimaciones de pose y propiedades.

## Próximos Pasos

- Implementar y optimizar la arquitectura para estimación de pose.
- Desarrollar algoritmos para la identificación y estimación de propiedades de los objetos.
- Realizar experimentos exhaustivos para validar los resultados de la Fase II.

## Contribuciones

Las contribuciones a este proyecto son bienvenidas. Por favor, abra un issue para discutir los cambios propuestos antes de realizar un pull request.

## Licencia

[Pendiente de especificar]
