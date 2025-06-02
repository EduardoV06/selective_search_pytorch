from opencv_custom.selectivesearchsegmentation_opencv_custom import SelectiveSearchOpenCVCustom
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
import os

def prepocess_img(img_path):
    img = Image.open(img_path).convert('RGB')

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: x * 255),  
    ])

    img_rgb = transform(img).byte()

    img_bgr = img_rgb[[2, 1, 0], :, :] 
    
    img_bgr_hw3 = img_bgr.permute(1, 2, 0)

    img_bgr1hw3_255 = img_bgr_hw3.unsqueeze(0)

    img_bgr1hw3_255 = img_bgr1hw3_255.contiguous()
    return img_bgr1hw3_255

def calculate_normalized_perimeter(mask):
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    area = np.sum(mask_uint8)
    if area == 0:
        return 0
    return perimeter / area

def select_by_softness(masks, idx):

    perimeters = [calculate_normalized_perimeter(m) for m in masks]
    
    sorted_indices = np.argsort(perimeters)[::-1]
    perimeters_sorted = np.array(perimeters)[sorted_indices]
    
    if idx < 0 or idx >= len(perimeters_sorted):
        raise IndexError(f"Index {idx} out of valid range (0 to {len(perimeters_sorted)-1})")
    
    # Get the original index of the mask
    return masks[sorted_indices]

def extract_bboxes_per_class(mask):

    bboxes = []
    classes = np.unique(mask)
    classes = classes[classes != 0]

    for cls in classes:
        binary_mask = (mask == cls).astype(np.uint8)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
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
