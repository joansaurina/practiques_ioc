# Prácticas IOC: Detección de Objetos y Análisis de Propiedades
## Joan Saurina

Este repositorio contiene el trabajo realizado para las prácticas del Instituto de Organización y Control (IOC), enfocadas en la detección de objetos y el análisis de sus propiedades utilizando técnicas de aprendizaje profundo y visión por computador.

## Fase I: Pipeline para Identificación de Objetos (Marzo 2024 - Mediados de Abril 2024)

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
