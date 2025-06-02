from opencv_custom.selectivesearchsegmentation_opencv_custom import SelectiveSearchOpenCVCustom
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
import os

def prepocess_img(img_path):
    # Carregar a imagem e converter para RGB
    img = Image.open(img_path).convert('RGB')

    # Definir as transformações: converter para tensor e escalar para [0, 255]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converte para [0.0, 1.0]
        transforms.Lambda(lambda x: x * 255),  # Escala para [0.0, 255.0]
    ])

    # Aplicar as transformações e converter para uint8
    img_rgb = transform(img).byte()

    # Converter de RGB para BGR
    img_bgr = img_rgb[[2, 1, 0], :, :]  # Reordena os canais

    # Permutar as dimensões para (height, width, channels)
    img_bgr_hw3 = img_bgr.permute(1, 2, 0)

    # Adicionar a dimensão do batch: (1, height, width, channels)
    img_bgr1hw3_255 = img_bgr_hw3.unsqueeze(0)

    # Garantir que o tensor seja contíguo
    img_bgr1hw3_255 = img_bgr1hw3_255.contiguous()
    return img_bgr1hw3_255

def calcular_perimetro_normalizado(mask):
    mask_uint8 = mask.astype(np.uint8)
    contornos, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimetro = sum(cv2.arcLength(contorno, True) for contorno in contornos)
    area = np.sum(mask_uint8)
    if area == 0:
        return 0
    return perimetro / area

def select_softness(mascaras, idx):
    """
    Encontra o bbox correspondente à máscara com o perímetro normalizado
    na posição 'idx' após ordenação decrescente dos perímetros.
    """
    # Calcular os perímetros normalizados
    perimetros = [calcular_perimetro_normalizado(m) for m in mascaras]
    
    # Ordenar os perímetros em ordem decrescente
    indices_ordenados = np.argsort(perimetros)[::-1]
    perimetros_sorted = np.array(perimetros)[indices_ordenados]
    
    # Verificar se o índice fornecido está dentro do intervalo válido
    if idx < 0 or idx >= len(perimetros_sorted):
        raise IndexError(f"Índice {idx} fora do intervalo válido (0 a {len(perimetros_sorted)-1})")
    
    # Obter o índice original da máscara
    idx_original = indices_ordenados[idx]
    return mascaras[idx_original]

def extract_bboxes_per_classe(mascara):
    """
    Extrai bounding boxes de uma máscara multiclasses.

    Parâmetros:
        mascara (np.ndarray): Máscara 2D com diferentes valores inteiros representando classes.

    Retorna:
        List[Tuple[int, int, int, int]]: Lista de bounding boxes no formato (x, y, w, h).
    """
    bboxes = []
    # Identificar os valores únicos na máscara, excluindo o fundo (0)
    classes = np.unique(mascara)
    classes = classes[classes != 0]

    for classe in classes:
        # Criar uma máscara binária para a classe atual
        mask_binaria = (mascara == classe).astype(np.uint8)

        # Encontrar contornos na máscara binária
        contornos, _ = cv2.findContours(mask_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            x, y, w, h = cv2.boundingRect(contorno)
            bboxes.append((x, y, w, h))
    return bboxes

def load_selective_search( preset = 'fast', remove_duplicate_boxes = False, lib_path = 'opencv_custom/selectivesearchsegmentation_opencv_custom_.so', max_num_rects = 4096, max_num_planes = 16, max_num_bit = 64, base_k = 0, inc_k = 0, sigma = 0):
    algo = SelectiveSearchOpenCVCustom(
    preset=preset,
    remove_duplicate_boxes=remove_duplicate_boxes,
    lib_path=lib_path,
    max_num_rects = max_num_rects, max_num_planes = max_num_planes, 
    max_num_bit = max_num_bit, base_k = base_k, inc_k = inc_k, sigma = sigma)
    return algo

