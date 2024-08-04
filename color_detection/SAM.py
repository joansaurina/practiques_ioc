import torch
import cv2
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt

from ...segment.segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from ...segment.segment_anything import sam_model_registry

def segment_image_with_sam(image_path, normalized_box, device, model_type):
    """
    # Ejemplo de uso
    normalized_box = [0.497059, 0.584683, 0.175294, 0.114875]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_type = vit_h = big, vit_l = mid, vit_b = small
    image_path = '*'
    cropped_image_rgb, detections, annotated_image_rgb = segment_image_with_sam(image_path, normalized_box, device, model_type)
    """
    if not isinstance(normalized_box, (list, np.ndarray)) or len(normalized_box) != 4:
        raise ValueError("normalized_box debe ser una lista o un numpy array de longitud 4.")
    
    if device not in ['cpu', 'cuda:0', 'cuda:1']:
        raise ValueError("device debe ser 'cpu', 'cuda:0' o 'cuda:1'.")
    
    if model_type not in ['vit_b', 'vit_l', 'vit_h']:
        raise ValueError("model_type debe ser 'vit_b', 'vit_l' o 'vit_h'.")
    
    DEVICE = torch.device(device)
    MODEL_TYPE = model_type
    checkpoints = {
        'vit_b': 'C:/Users/joans/practicas/project/fase1/project/segment_facebook/weights/sam_small.pth',
        'vit_l': 'C:/Users/joans/practicas/project/fase1/project/segment_facebook/weights/sam_mid.pth',
        'default': 'C:/Users/joans/practicas/project/fase1/project/segment_facebook/weights/sam_big.pth'
    }
    checkpoint_path = checkpoints.get(model_type, checkpoints['default'])
    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint_path)
    sam.to(device=DEVICE)


    # Carga la imagen
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen desde la ruta: {image_path}")
    
    height, width, _ = image_bgr.shape

    # Recorta la imagen al tamaño de la bounding box
    box = np.array([
        (normalized_box[0] * width) - (normalized_box[2] * width / 2),  # x_min
        (normalized_box[1] * height) - (normalized_box[3] * height / 2), # y_min
        normalized_box[2] * width,  # width
        normalized_box[3] * height  # height
    ]).astype(int)
    x_min, y_min, box_width, box_height = box
    x_max = x_min + box_width
    y_max = y_min + box_height
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)
    cropped_image = image_bgr[y_min:y_max, x_min:x_max]

    # Convierte la imagen a RGB
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

    # Pasar imagen por el modelo
    mask_generator = SamAutomaticMaskGenerator(sam)
    sam_result = mask_generator.generate(cropped_image_rgb)

    return cropped_image_rgb, sam_result
