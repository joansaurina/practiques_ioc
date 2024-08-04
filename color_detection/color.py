import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import binary_fill_holes
from skimage.morphology import dilation, disk
import sys

from SAM import segment_image_with_sam

def propagate_mask(mask):
    """
    Propaga la máscara para rellenar huecos.
    """
    filled_mask = binary_fill_holes(mask).astype(np.uint8)
    dilated_mask = dilation(filled_mask, disk(3))
    return dilated_mask

def train_kmeans_on_largest_mask(image_path, normalized_box, device, model_type, k):
    """
    Entrena un modelo KMeans con k clusters en los colores de la máscara más grande.
    
    Parámetros:
    - image_path: Ruta de la imagen.
    - normalized_box: Bounding box normalizada [x_center, y_center, width, height].
    - device: Dispositivo a utilizar ('cpu', 'cuda:0', 'cuda:1').
    - model_type: Tipo de modelo SAM ('vit_b', 'vit_l', 'vit_h').
    - k: Número de clusters para KMeans.
    
    Retorna:
    - centroids: Colores de cada centroide.
    """
    # Segmentar la imagen y obtener las máscaras
    cropped_image_rgb, sam_result = segment_image_with_sam(image_path, normalized_box, device, model_type)
    
    # Seleccionar la máscara más grande
    largest_mask = max(sam_result, key=lambda x: x['area'])['segmentation']

    # Propagar y rellenar la máscara
    propagated_mask = propagate_mask(largest_mask)
    
    # Aplicar la máscara a la imagen recortada
    masked_image = cropped_image_rgb * np.stack([propagated_mask]*3, axis=-1)
    
    # Extraer los colores de la imagen enmascarada
    colors = masked_image[propagated_mask > 0]

    # Entrenar KMeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(colors)
    
    # Obtener los centroides de los colores
    centroids = kmeans.cluster_centers_

    return centroids

def main(args):
    image_path = args[0]
    normalized_box = args[1]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_type = 'vit_h'
    k = args[2]
    
    centroids = train_kmeans_on_largest_mask(image_path, normalized_box, device, model_type, k)
    print("Centroides de los colores:", centroids)
 
# Ejemplo de uso
if __name__ == "__main__":
    main(sys.argv[1:])